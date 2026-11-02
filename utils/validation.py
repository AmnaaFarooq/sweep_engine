"""
validation.py - data-integrity checks and gap detection for 1m OHLCV data.

Design rules
------------
* validate_ohlcv() never raises on bad data. It returns a ValidationResult that
  lists EVERY violation found, so one run shows all problems at once.
  The caller decides to abort via raise_if_invalid().
* Nothing here repairs, fills, sorts or drops data. Bad data is reported;
  missing bars are reported. No candle is ever fabricated.

Gap detection mechanism (see find_gaps):
    consecutive timestamp difference  vs  expected bar length (60 s).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import polars as pl

import config as cfg
from utils.logging_setup import get_logger

logger = get_logger(__name__)

# Check identifiers (strings, so tests and reports can reference them).
CHECK_SCHEMA = "missing_columns"
CHECK_EMPTY = "empty_frame"
CHECK_TZ = "timestamp_not_utc_datetime"
CHECK_NULLS = "null_values"
CHECK_NON_FINITE = "non_finite_values"
CHECK_DUPLICATE = "duplicate_timestamps"
CHECK_NON_MONOTONIC = "non_monotonic_timestamps"
CHECK_MISALIGNED = "misaligned_timestamps"
CHECK_OHLC = "ohlc_integrity"
CHECK_PRICE_POSITIVE = "non_positive_price"
CHECK_VOLUME = "negative_volume"

_PRICE_COLUMNS = ["open", "high", "low", "close"]
_NUMERIC_COLUMNS = _PRICE_COLUMNS + ["volume"]


@dataclass(frozen=True)
class Violation:
    check: str
    count: int
    detail: str


@dataclass
class ValidationResult:
    label: str = ""
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def checks(self) -> set[str]:
        return {v.check for v in self.violations}

    def describe(self) -> str:
        if self.ok:
            return f"[{self.label}] validation passed"
        lines = [f"[{self.label}] validation FAILED ({len(self.violations)} check(s)):"]
        lines += [f"  - {v.check}: {v.count} row(s); {v.detail}" for v in self.violations]
        return "\n".join(lines)


class DataValidationError(Exception):
    """Raised by raise_if_invalid() when a ValidationResult has violations."""

    def __init__(self, result: ValidationResult):
        super().__init__(result.describe())
        self.result = result


def _sample(df: pl.DataFrame, mask: pl.Expr) -> str:
    """First few offending timestamps (with row index) for a violation message."""
    rows = (
        df.with_row_index("_row")
        .filter(mask)
        .select("_row", "timestamp")
        .head(cfg.VALIDATION_SAMPLE_SIZE)
        .iter_rows()
    )
    return "e.g. " + ", ".join(f"row {r} @ {ts}" for r, ts in rows)


def validate_ohlcv(df: pl.DataFrame, label: str = "") -> ValidationResult:
    """
    Validate a candidate OHLCV frame IN ITS ORIGINAL ROW ORDER.

    Checks: schema, UTC datetime dtype, nulls, NaN/inf, duplicate timestamps,
    strictly increasing timestamps, whole-minute alignment,
    high >= max(open, close), low <= min(open, close), prices > 0, volume >= 0.
    """
    result = ValidationResult(label=label)

    def add(check: str, count: int, detail: str) -> None:
        result.violations.append(Violation(check, int(count), detail))

    # ---- structural prerequisites: if these fail, later checks are meaningless
    missing = [c for c in cfg.OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        add(CHECK_SCHEMA, len(missing), f"missing columns: {missing}")
        return result
    if df.height == 0:
        add(CHECK_EMPTY, 0, "frame has no rows")
        return result
    ts_dtype = df.schema["timestamp"]
    if not (isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone == cfg.TIMEZONE):
        add(CHECK_TZ, df.height, f"timestamp dtype is {ts_dtype}, expected Datetime[{cfg.TIMEZONE}]")
        return result

    ts = pl.col("timestamp")

    # ---- nulls and non-finite numbers (comparisons silently ignore these)
    null_rows = df.filter(pl.any_horizontal(pl.col(cfg.OUTPUT_COLUMNS).is_null()))
    if null_rows.height:
        add(CHECK_NULLS, null_rows.height, "null in one of " + ", ".join(cfg.OUTPUT_COLUMNS))
    non_finite = pl.any_horizontal(pl.col(_NUMERIC_COLUMNS).is_finite().not_())
    n_non_finite = df.filter(non_finite).height
    if n_non_finite:
        add(CHECK_NON_FINITE, n_non_finite, "NaN or +/-inf in price/volume. " + _sample(df, non_finite))

    # ---- timestamps
    n_dup = df.height - df["timestamp"].n_unique()
    if n_dup:
        dup_ts = df.filter(ts.is_duplicated()).select(ts.unique().sort().head(cfg.VALIDATION_SAMPLE_SIZE))
        add(CHECK_DUPLICATE, n_dup, f"repeated timestamps, e.g. {dup_ts['timestamp'].to_list()}")

    # Going backwards in time. (Equal consecutive timestamps are duplicates and
    # are reported above; together the two checks enforce STRICT increase.)
    backwards = ts.diff() < timedelta(0)
    n_back = df.filter(backwards).height
    if n_back:
        add(CHECK_NON_MONOTONIC, n_back, "timestamp earlier than previous row. " + _sample(df, backwards))

    step_ms = cfg.EXPECTED_BAR_SECONDS * cfg.MS_PER_SECOND
    misaligned = (ts.dt.epoch(cfg.TIME_UNIT) % step_ms) != 0
    n_mis = df.filter(misaligned).height
    if n_mis:
        add(CHECK_MISALIGNED, n_mis, "timestamp not on a whole-bar boundary. " + _sample(df, misaligned))

    # ---- price / volume integrity
    max_oc = pl.max_horizontal("open", "close")
    min_oc = pl.min_horizontal("open", "close")
    ohlc_bad = (pl.col("high") < max_oc) | (pl.col("low") > min_oc)
    n_ohlc = df.filter(ohlc_bad).height
    if n_ohlc:
        add(CHECK_OHLC, n_ohlc, "need high >= max(open,close) and low <= min(open,close). " + _sample(df, ohlc_bad))

    nonpos = pl.any_horizontal(pl.col(_PRICE_COLUMNS) <= 0)
    n_nonpos = df.filter(nonpos).height
    if n_nonpos:
        add(CHECK_PRICE_POSITIVE, n_nonpos, "price <= 0. " + _sample(df, nonpos))

    neg_vol = pl.col("volume") < 0
    n_negvol = df.filter(neg_vol).height
    if n_negvol:
        add(CHECK_VOLUME, n_negvol, "volume < 0. " + _sample(df, neg_vol))

    return result


def raise_if_invalid(result: ValidationResult) -> None:
    """Log the outcome; raise DataValidationError if any check failed."""
    if result.ok:
        logger.info(result.describe())
        return
    logger.error(result.describe())
    raise DataValidationError(result)


# --------------------------------------------------------------------------
# Gap detection. Inputs must already have passed validate_ohlcv()
# (strictly increasing, aligned timestamps) - otherwise the arithmetic is moot.
# --------------------------------------------------------------------------
def find_gaps(df: pl.DataFrame) -> pl.DataFrame:
    """
    One row per gap (a run of consecutive missing bars).

    Mechanism: next_timestamp - previous_timestamp is computed for every pair of
    adjacent rows. In a perfect 1m series every difference is exactly 60 s.
    Any difference > 60 s means bars are missing between the two rows:
        missing_bars = difference / 60 s - 1
    """
    step_ms = cfg.EXPECTED_BAR_SECONDS * cfg.MS_PER_SECOND
    pairs = df.select(
        pl.col("timestamp").shift(1).alias("prev_timestamp"),
        pl.col("close").shift(1).alias("prev_close"),
        pl.col("timestamp").alias("next_timestamp"),
        pl.col("open").alias("next_open"),
    ).drop_nulls("prev_timestamp")                      # first row has no predecessor

    return (
        pairs.with_columns(
            pl.col("prev_timestamp").dt.epoch(cfg.TIME_UNIT).alias("_prev_ms"),
            pl.col("next_timestamp").dt.epoch(cfg.TIME_UNIT).alias("_next_ms"),
        )
        .with_columns(((pl.col("_next_ms") - pl.col("_prev_ms")) // step_ms - 1).alias("gap_missing_bars"))
        .filter(pl.col("gap_missing_bars") > 0)
        .with_row_index("gap_id")
    )


def build_gap_report(df: pl.DataFrame) -> pl.DataFrame:
    """
    One row per MISSING 1m bar, with the bars on either side of its gap.
    Columns: missing_timestamp, gap_id, gap_missing_bars,
             prev_timestamp, prev_close, next_timestamp, next_open.
    Nothing is filled - this only lists what is absent.
    """
    step_ms = cfg.EXPECTED_BAR_SECONDS * cfg.MS_PER_SECOND
    gaps = find_gaps(df)
    return (
        gaps.with_columns(
            # every expected bar-open strictly between prev and next (end exclusive)
            pl.int_ranges(pl.col("_prev_ms") + step_ms, pl.col("_next_ms"), step_ms).alias("_missing_ms")
        )
        .explode("_missing_ms")
        .with_columns(
            pl.from_epoch(pl.col("_missing_ms"), time_unit=cfg.TIME_UNIT)
            .cast(pl.Datetime(cfg.TIME_UNIT))
            .dt.replace_time_zone(cfg.TIMEZONE)
            .alias("missing_timestamp")
        )
        .select(
            "missing_timestamp", "gap_id", "gap_missing_bars",
            "prev_timestamp", "prev_close", "next_timestamp", "next_open",
        )
    )


def flag_after_gap(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a boolean column `after_gap`: True for the first bar after each gap.
    Signal generation must skip these bars. Bars are never altered or added.
    """
    step = timedelta(seconds=cfg.EXPECTED_BAR_SECONDS)
    return df.with_columns(
        (pl.col("timestamp").diff() > step).fill_null(False).alias(cfg.AFTER_GAP_COLUMN)
    )
