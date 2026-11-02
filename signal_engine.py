"""
signal_engine.py - Phase 2: 5m resampling, ATR(14), sessions, Asian levels,
sweep / confirmation detection with signal lifetime.

Reads  data/btc_1m.parquet   (Phase 1 output, 1m bars, UTC)
Writes data/signals.parquet  (5m bars + signal columns)

TIMEFRAMES: input is 1m; everything below runs on 5m candles. ATR(14) is
computed on those 5m candles.

TIME CONVENTION (the root of look-ahead safety)
-----------------------------------------------
A row's `timestamp` is the bar's OPEN time (label="left"). The bar covers
[timestamp, timestamp + 5m) and its close is only KNOWN at timestamp + 5m.
Every flag on row t is computed from rows <= t only, i.e. it is knowable at
the close of bar t. A trade reacting to row t therefore enters at the open of
row t+1 (Phase 3). tests/test_signal.py proves causality by truncating the
data and checking that earlier rows do not change.

SESSIONS: fixed-UTC hour windows, NOT DST-aware (see config.py).

LEVELS: the Asian high/low of a UTC day is only exposed AFTER the Asian
session has finished (from 08:00 on). During the Asian session the level is
null - exposing it earlier would leak the session's future highs/lows.
"""
from __future__ import annotations

import json
import sys

import polars as pl

import config as cfg
from utils.logging_setup import get_logger, setup_logging
from utils.validation import DataValidationError, raise_if_invalid, validate_ohlcv

logger = get_logger(__name__)

SIDES = ("short", "long")
OUTCOME_UNRESOLVED = "unresolved"


# ==========================================================================
# 1. Resample 1m -> 5m
# ==========================================================================
def resample_5m(df_1m: pl.DataFrame) -> pl.DataFrame:
    """
    1m -> 5m with group_by_dynamic. Requires `timestamp` sorted ascending
    (guaranteed by Phase 1 validation).

    Buckets are [t, t+5m), labelled by their LEFT edge (the open time).
    A bucket that does not contain exactly 5 one-minute bars is PARTIAL - it
    means a 1m bar is missing (a gap) - and is dropped, never completed.
    """
    bars = df_1m.group_by_dynamic(
        index_column="timestamp",
        every=cfg.RESAMPLE_EVERY,
        period=cfg.RESAMPLE_PERIOD,
        closed="left",
        label="left",
    ).agg(
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
        pl.len().alias("n_source"),
    )

    partial = bars.filter(pl.col("n_source") != cfg.SOURCE_BARS_PER_BAR)
    for row in partial.iter_rows(named=True):
        logger.warning("dropped partial 5m bar %s: %d of %d source 1m bars (gap boundary)",
                       row["timestamp"], row["n_source"], cfg.SOURCE_BARS_PER_BAR)
    logger.info("resampled %d 1m rows -> %d complete 5m bars; dropped %d partial 5m bar(s)",
                df_1m.height, bars.height - partial.height, partial.height)

    return bars.filter(pl.col("n_source") == cfg.SOURCE_BARS_PER_BAR).drop("n_source")


# ==========================================================================
# 2. ATR(14) on 5m candles
# ==========================================================================
def add_atr(df: pl.DataFrame) -> pl.DataFrame:
    """
    true_range = max(high-low, |high-prev_close|, |low-prev_close|)
    atr_14     = 14-period SIMPLE rolling mean of true_range (5m candles).

    Gap-strict: `prev_close` is only trusted if the previous row is exactly
    one 5m bar earlier. Otherwise true_range is null (we never compare against
    a close from before a hole). The rolling mean needs 14 non-null values,
    so ATR is null for the first 14 rows and for 14 bars after any gap.
    """
    prev_close = pl.col("close").shift(1)
    contiguous = pl.col("timestamp").diff() == cfg.BAR_DURATION   # null on row 0 -> treated as False
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return df.with_columns(
        pl.when(contiguous).then(true_range).otherwise(None).alias("true_range")
    ).with_columns(
        pl.col("true_range")
        .rolling_mean(window_size=cfg.ATR_PERIOD, min_samples=cfg.ATR_PERIOD)
        .alias(cfg.ATR_COLUMN)
    )


# ==========================================================================
# 3. Session tags (+ helper columns)
# ==========================================================================
def _in_window(hour: pl.Expr, window: tuple[int, int]) -> pl.Expr:
    return hour.is_between(window[0], window[1], closed="left")      # start <= hour < end


def add_session_tags(df: pl.DataFrame) -> pl.DataFrame:
    """
    session_tag       : reporting label of the bar (exclusive; overlap is its own tag).
    lifetime_session  : which session "owns" the bar for first-sweep / expiry
                        bookkeeping (London / NY / null). Overlap belongs to London.
    date              : UTC calendar day (grouping key).
    """
    hour = pl.col("timestamp").dt.hour()

    # when/then chains are evaluated top to bottom, first match wins. The overlap
    # test MUST come before London and NY, otherwise 13:00-16:00 would be
    # tagged plain "London".
    tag = (
        pl.when(_in_window(hour, cfg.ASIAN_WINDOW)).then(pl.lit(cfg.TAG_ASIAN))
        .when(_in_window(hour, cfg.OVERLAP_WINDOW)).then(pl.lit(cfg.TAG_OVERLAP))
        .when(_in_window(hour, cfg.LONDON_WINDOW)).then(pl.lit(cfg.TAG_LONDON))
        .when(_in_window(hour, cfg.NY_WINDOW)).then(pl.lit(cfg.TAG_NY))
        .otherwise(pl.lit(cfg.TAG_OFF))
    )

    lifetime = pl.lit(None, dtype=pl.String)
    session_end_hour = pl.lit(None, dtype=pl.Int64)
    for name, (start, end) in reversed(list(cfg.LIFETIME_WINDOWS.items())):
        inside = _in_window(hour, (start, end))
        lifetime = pl.when(inside).then(pl.lit(name)).otherwise(lifetime)
        session_end_hour = pl.when(inside).then(pl.lit(end, dtype=pl.Int64)).otherwise(session_end_hour)

    # The scheduled LAST bar of a lifetime session is the one whose close lands
    # exactly on the session end (e.g. the 15:55 bar for a 16:00 end).
    bar_close = pl.col("timestamp") + cfg.BAR_DURATION
    is_last_bar = (
        (bar_close.dt.hour() == session_end_hour) & (bar_close.dt.minute() == 0)
    ).fill_null(False)

    return df.with_columns(
        pl.col("timestamp").dt.date().alias("date"),
        tag.alias("session_tag"),
        lifetime.alias("lifetime_session"),
        is_last_bar.alias("is_session_last_bar"),
    )


# ==========================================================================
# 4. Asian high / low - broadcast per UTC day WITHOUT future leakage
# ==========================================================================
def add_asian_levels(df: pl.DataFrame) -> pl.DataFrame:
    """
    asian_high / asian_low for each UTC day, visible only after the Asian
    session has closed and only if all 96 Asian 5m bars exist.
    """
    hour = pl.col("timestamp").dt.hour()
    is_asian = _in_window(hour, cfg.ASIAN_WINDOW)

    # Visible only once the Asian session is over (08:00 and later).
    # Before that the full-day max would include bars that have not happened yet.
    session_finished = hour >= cfg.ASIAN_WINDOW[1]
    # A day with missing/dropped Asian bars has an unreliable range -> no level.
    session_complete = is_asian.sum().over("date") == cfg.ASIAN_EXPECTED_BARS
    usable = session_finished & session_complete

    # `.filter(is_asian)` restricts the aggregation to Asian bars only;
    # `.over("date")` evaluates it once per UTC day and broadcasts the single
    # result to every row of that day. The when/then mask then hides it on
    # rows where it must not be known yet.
    out = df.with_columns(
        pl.when(usable).then(pl.col("high").filter(is_asian).max().over("date")).alias("asian_high"),
        pl.when(usable).then(pl.col("low").filter(is_asian).min().over("date")).alias("asian_low"),
    )

    incomplete = (
        df.group_by("date").agg(is_asian.sum().alias("n_asian"))
        .filter(pl.col("n_asian") != cfg.ASIAN_EXPECTED_BARS)
        .sort("date")
    )
    for row in incomplete.iter_rows(named=True):
        logger.warning("Asian range unusable on %s: %d of %d Asian 5m bars present -> no signals that day",
                       row["date"], row["n_asian"], cfg.ASIAN_EXPECTED_BARS)
    return out


# ==========================================================================
# 5. Sweeps, confirmation, lifetime
# ==========================================================================
def _raw_sweep_conditions(mode: str) -> tuple[pl.Expr, pl.Expr]:
    """(short_condition, long_condition) for the chosen sweep_mode. Pure bar-level patterns."""
    high, low, close = pl.col("high"), pl.col("low"), pl.col("close")
    a_high, a_low = pl.col("asian_high"), pl.col("asian_low")

    wick_short = (high > a_high) & (close < a_high)     # pierced above, closed back below
    wick_long = (low < a_low) & (close > a_low)         # pierced below, closed back above
    beyond_short = close > a_high                       # closed beyond (no close-back needed)
    beyond_long = close < a_low

    if mode == cfg.SWEEP_MODE_WICK:
        return wick_short, wick_long
    if mode == cfg.SWEEP_MODE_CLOSE:
        return beyond_short, beyond_long
    if mode == cfg.SWEEP_MODE_BOTH:
        return wick_short | beyond_short, wick_long | beyond_long
    raise ValueError(f"unknown sweep_mode {mode!r}; expected one of {cfg.SWEEP_MODES}")


def _apply_side(df: pl.DataFrame, side: str, raw_pattern: pl.Expr, inside_level: pl.Expr) -> pl.DataFrame:
    """
    Signal state machine for ONE level (short -> asian_high, long -> asian_low),
    implemented with causal cumulative sums inside each (date, lifetime_session)
    group. Every cum_sum only ever looks at rows up to and including the
    current one, so nothing leaks from the future.
    """
    p = f"_{side}_"
    group = ["date", "lifetime_session"]

    def cum(expr: pl.Expr) -> pl.Expr:                  # running count of True values inside the group
        return expr.cast(pl.Int32).cum_sum().over(group)

    c = pl.col
    tradable_bar = c("_eligible")

    # 1) raw sweep patterns on eligible bars only
    df = df.with_columns(
        (raw_pattern.fill_null(False) & tradable_bar).alias(p + "raw"),
        (raw_pattern.fill_null(False) & ~tradable_bar & c("lifetime_session").is_not_null()
         & c("asian_high").is_not_null() & ~c("_contiguous")).alias(p + "excluded"),
    )
    # 2) FIRST sweep per (day, session, level); later ones are duplicates
    df = df.with_columns((c(p + "raw") & (cum(c(p + "raw")) == 1)).alias(p + "first"))
    df = df.with_columns(
        (c(p + "raw") & ~c(p + "first")).alias(p + "dup"),
        (cum(c(p + "first")) > 0).alias(p + "swept"),                       # sweep bar and everything after
    )
    df = df.with_columns((c(p + "swept") & ~c(p + "first")).alias(p + "swept_before"))   # strictly after sweep bar

    # 3) a gap after the sweep invalidates the pending signal
    df = df.with_columns((c(p + "swept_before") & ~c("_contiguous")).alias(p + "gap_bar"))
    df = df.with_columns((cum(c(p + "gap_bar")) > 0).alias(p + "gap_seen"))

    # 4) confirmation: close back inside, on a later (or, by config, the same) bar,
    #    with no gap since the sweep. Only the FIRST such bar confirms.
    confirm_window = c(p + "swept") if cfg.CONFIRM_ON_SWEEP_BAR else c(p + "swept_before")
    df = df.with_columns(
        (confirm_window & inside_level.fill_null(False) & tradable_bar & ~c(p + "gap_seen")).alias(p + "cand")
    )
    df = df.with_columns((c(p + "cand") & (cum(c(p + "cand")) == 1)).alias(p + "confirm"))
    df = df.with_columns((cum(c(p + "confirm")) > 0).alias(p + "confirmed"))

    # 5) gap-invalidated (rejected) and expired
    df = df.with_columns(
        (c(p + "gap_bar") & (cum(c(p + "gap_bar")) == 1) & ~c(p + "confirmed")).alias(p + "kill")
    )
    df = df.with_columns((cum(c(p + "kill")) > 0).alias(p + "killed"))
    df = df.with_columns(
        (c("is_session_last_bar") & c(p + "swept") & ~c(p + "confirmed") & ~c(p + "killed")).alias(p + "expire"),
        (c(p + "swept") & ~c(p + "confirmed") & ~c(p + "killed")).alias(p + "pending"),
    )

    # 6) per-bar status for this side (first matching branch wins)
    df = df.with_columns(
        pl.when(c(p + "confirm")).then(pl.lit(cfg.STATUS_CONFIRMED))
        .when(c(p + "kill")).then(pl.lit(cfg.STATUS_REJECTED))
        .when(c(p + "expire")).then(pl.lit(cfg.STATUS_EXPIRED))
        .when(c(p + "dup")).then(pl.lit(cfg.STATUS_REJECTED))
        .when(c(p + "pending")).then(pl.lit(cfg.STATUS_PENDING))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(p + "status")
    )

    # 7) reporting helpers (causal forward-fill of sweep time/tag; outcome is a
    #    group-level look-back used ONLY for the log, never exposed as a feature)
    resolved_any = (c(p + "confirm") | c(p + "kill") | c(p + "expire")).any().over(group)
    df = df.with_columns(
        pl.when(c(p + "first")).then(c("timestamp")).forward_fill().over(group).alias(p + "sweep_ts"),
        pl.when(c(p + "first")).then(c("session_tag")).forward_fill().over(group).alias(p + "sweep_tag"),
        pl.when(c(p + "confirm").any().over(group)).then(pl.lit(cfg.STATUS_CONFIRMED))
        .when(c(p + "kill").any().over(group)).then(pl.lit(cfg.STATUS_REJECTED))
        .when(c(p + "expire").any().over(group)).then(pl.lit(cfg.STATUS_EXPIRED))
        .when(c(p + "first").any().over(group) & ~resolved_any).then(pl.lit(OUTCOME_UNRESOLVED))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(p + "outcome"),
        pl.when(c(p + "confirm") | c(p + "kill") | c(p + "expire")).then(c("timestamp")).max().over(group)
        .alias(p + "outcome_ts"),
    )
    return df


def detect_signals(df: pl.DataFrame, sweep_mode: str | None = None) -> pl.DataFrame:
    """Adds eligibility, per-side state machines and the combined signal_status."""
    mode = sweep_mode or cfg.SWEEP_MODE
    short_pattern, long_pattern = _raw_sweep_conditions(mode)      # validates the mode early

    contiguous = (pl.col("timestamp").diff() == cfg.BAR_DURATION).fill_null(False)
    df = df.with_columns(contiguous.alias("_contiguous"))
    # A bar may carry a sweep/confirmation only if: it directly follows the
    # previous 5m bar (no gap before it), today's Asian level exists, and it
    # lies inside a lifetime session (London or NY-after-London).
    df = df.with_columns(
        (pl.col("_contiguous") & pl.col("asian_high").is_not_null()
         & pl.col("lifetime_session").is_not_null()).alias("_eligible")
    )

    df = _apply_side(df, "short", short_pattern, pl.col("close") < pl.col("asian_high"))
    df = _apply_side(df, "long", long_pattern, pl.col("close") > pl.col("asian_low"))

    # Combine the two sides into one column. Flags stay exact per side; only the
    # label can collide, and then the precedence in config decides.
    status_s, status_l = pl.col("_short_status"), pl.col("_long_status")
    combined = pl.coalesce([
        pl.when((status_s == s) | (status_l == s)).then(pl.lit(s)) for s in cfg.STATUS_PRECEDENCE
    ])
    return df.with_columns(
        pl.col("_short_first").alias("sweep_short"),
        pl.col("_long_first").alias("sweep_long"),
        pl.col("_short_confirm").alias("confirm_short"),
        pl.col("_long_confirm").alias("confirm_long"),
        combined.alias("signal_status"),
    )


def compute_signals(df_5m: pl.DataFrame, sweep_mode: str | None = None) -> pl.DataFrame:
    """5m OHLCV -> full analysis frame (output columns + internal `_*` columns)."""
    df = add_atr(df_5m)
    df = add_session_tags(df)
    df = add_asian_levels(df)
    return detect_signals(df, sweep_mode)


def to_output(df: pl.DataFrame) -> pl.DataFrame:
    """Exactly the columns of data/signals.parquet, in spec order."""
    return df.select(cfg.SIGNAL_COLUMNS)


# ==========================================================================
# 6. Logging / reporting
# ==========================================================================
def build_signal_log(df: pl.DataFrame) -> pl.DataFrame:
    """One row per signal (first sweep): side, lifetime session, sweep time/tag, outcome."""
    frames = []
    for side in SIDES:
        p = f"_{side}_"
        frames.append(
            df.filter(pl.col(p + "first")).select(
                pl.lit(side).alias("side"),
                pl.col("lifetime_session"),
                pl.col("timestamp").alias("sweep_ts"),
                pl.col("session_tag").alias("sweep_tag"),
                pl.col(p + "outcome").alias("outcome"),
                pl.col(p + "outcome_ts").alias("outcome_ts"),
            )
        )
    return pl.concat(frames).sort("sweep_ts")


def log_signal_events(df: pl.DataFrame) -> None:
    """Expired / gap-rejected / excluded / duplicate events, then a per-tag summary."""
    for side in SIDES:
        p = f"_{side}_"
        for r in df.filter(pl.col(p + "expire")).iter_rows(named=True):
            logger.info("EXPIRED %s signal: sweep at %s (%s) never closed back inside before session end %s",
                        side, r[p + "sweep_ts"], r[p + "sweep_tag"], r["timestamp"] + cfg.BAR_DURATION)
        for r in df.filter(pl.col(p + "kill")).iter_rows(named=True):
            logger.warning("REJECTED %s signal: gap at %s while pending (sweep at %s)",
                           side, r["timestamp"], r[p + "sweep_ts"])
        for r in df.filter(pl.col(p + "excluded")).iter_rows(named=True):
            logger.warning("EXCLUDED %s sweep pattern on %s: bar follows a gap, not eligible for signals",
                           side, r["timestamp"])
        n_dup = df.filter(pl.col(p + "dup")).height
        logger.info("%s side: %d later sweep(s) ignored (first-sweep-only rule) -> status 'rejected'", side, n_dup)

    log = build_signal_log(df)
    for r in log.filter(pl.col("outcome") == OUTCOME_UNRESOLVED).iter_rows(named=True):
        logger.warning("UNRESOLVED %s signal swept at %s: session's last scheduled bar is missing",
                       r["side"], r["sweep_ts"])

    logger.info("signals by sweep-bar tag (reported separately; overlap is its own row):")
    summary = (
        log.group_by("sweep_tag")
        .agg(
            pl.len().alias("signals"),
            (pl.col("outcome") == cfg.STATUS_CONFIRMED).sum().alias("confirmed"),
            (pl.col("outcome") == cfg.STATUS_EXPIRED).sum().alias("expired"),
            (pl.col("outcome") == cfg.STATUS_REJECTED).sum().alias("gap_rejected"),
        )
        .sort("sweep_tag")
    )
    for r in summary.iter_rows(named=True):
        logger.info("  %-18s signals=%d confirmed=%d expired=%d gap_rejected=%d",
                    r["sweep_tag"], r["signals"], r["confirmed"], r["expired"], r["gap_rejected"])
    if not summary.height:
        logger.info("  no signals")


# ==========================================================================
# 7. Entry point
# ==========================================================================
def main() -> int:
    setup_logging()
    logger.info("Phase 2 started: sweep_mode=%s confirm_on_sweep_bar=%s", cfg.SWEEP_MODE, cfg.CONFIRM_ON_SWEEP_BAR)

    df_1m = pl.read_parquet(cfg.PARQUET_PATH)
    try:
        raise_if_invalid(validate_ohlcv(df_1m, label="btc_1m.parquet"))
    except DataValidationError:
        logger.critical("ABORT: input parquet failed validation")
        return cfg.EXIT_ABORTED

    df_5m = resample_5m(df_1m)
    full = compute_signals(df_5m)
    out = to_output(full)

    log_signal_events(full)
    cfg.SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(cfg.SIGNALS_PATH)
    logger.info("signals written: %s (%d rows)", cfg.SIGNALS_PATH, out.height)
    # Sidecar so the backtester knows which settings produced these signals.
    cfg.SIGNALS_META_PATH.write_text(json.dumps(
        {"sweep_mode": cfg.SWEEP_MODE, "confirm_on_sweep_bar": cfg.CONFIRM_ON_SWEEP_BAR}), encoding="utf-8")
    return cfg.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
