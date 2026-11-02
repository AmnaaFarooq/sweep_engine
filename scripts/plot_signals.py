"""
plot_signals.py - visual validation of Phase 2 signals.

Reads data/signals.parquet and writes ONE html page (data/signal_charts.html)
with one interactive candlestick chart per selected UTC day. All times are UTC.

Overlays
  * Asian high / low   : dotted while the range is still FORMING (00:00-08:00,
                         drawn only as a visual aid), solid from 08:00 when the
                         level becomes known to the engine
  * sweep markers      : circles at the sweep bar (red = short sweep at the bar's
                         high, green = long sweep at the bar's low)
  * confirmation       : triangles at the confirm bar (down = short, up = long)
  * extras             : orange x = expired signal, grey x = rejected
                         (duplicate sweep ignored / gap while pending)
  * session shading    : Asian / London / London_NY_overlap / NY, taken straight
                         from the session_tag column

Usage (from the project root):
    python scripts/plot_signals.py                       # 10 random days, seed from config
    python scripts/plot_signals.py --seed 7              # different random days
    python scripts/plot_signals.py --require-signal      # only days that contain a sweep
    python scripts/plot_signals.py --dates 2025-03-04 2025-03-05
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # so `import config` works from scripts/

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

import config as cfg
from utils.logging_setup import get_logger, setup_logging

logger = get_logger("plot_signals")


def load_signals(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path).sort("timestamp")
    # Plotly draws tz-aware datetimes in their own zone; strip tz AFTER the data is UTC
    # so the axis shows plain UTC wall-clock time.
    return df.with_columns(
        pl.col("timestamp").dt.replace_time_zone(None).alias("ts"),
        pl.col("timestamp").dt.date().alias("date"),
    )


def pick_days(df: pl.DataFrame, n: int, seed: int, require_signal: bool, explicit: list[date] | None) -> list[date]:
    if explicit:
        return sorted(explicit)
    pool = df
    if require_signal:
        pool = df.filter(pl.col("sweep_short") | pl.col("sweep_long"))
    days = sorted(pool["date"].unique().to_list())
    if len(days) < n:
        logger.warning("only %d eligible day(s) available, plotting all of them", len(days))
        return days
    return sorted(random.Random(seed).sample(days, n))


def session_segments(day: pl.DataFrame) -> list[tuple[str, object, object]]:
    """Consecutive runs of the same session_tag -> (tag, x_start, x_end)."""
    segments, start, prev_tag, last_ts = [], None, None, None
    for ts, tag in zip(day["ts"].to_list(), day["session_tag"].to_list()):
        if tag != prev_tag:
            if prev_tag is not None:
                segments.append((prev_tag, start, ts))           # a run ends where the next begins
            start, prev_tag = ts, tag
        last_ts = ts
    if prev_tag is not None:
        segments.append((prev_tag, start, last_ts + cfg.BAR_DURATION))
    return segments


def build_day_figure(day: pl.DataFrame, day_date: date) -> go.Figure:
    fig = go.Figure()
    ts = day["ts"].to_list()
    day_high, day_low = day["high"].max(), day["low"].min()
    offset = (day_high - day_low) * cfg.PLOT_MARKER_OFFSET_FRACTION

    # --- session shading (below the candles)
    for tag, x0, x1 in session_segments(day):
        color = cfg.PLOT_SESSION_COLORS.get(tag)
        if color:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0, layer="below",
                          annotation_text=tag, annotation_position="top left", annotation_font_size=10)

    # --- candles
    fig.add_trace(go.Candlestick(
        x=ts, open=day["open"], high=day["high"], low=day["low"], close=day["close"],
        name="5m candles", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    # --- Asian high / low
    known = day.filter(pl.col("asian_high").is_not_null())
    if known.height:
        a_high, a_low = known["asian_high"][0], known["asian_low"][0]
        x_known, x_end = known["ts"][0], ts[-1] + cfg.BAR_DURATION
        for name, level, color in (("Asian high", a_high, cfg.PLOT_COLOR_LEVEL_HIGH),
                                   ("Asian low", a_low, cfg.PLOT_COLOR_LEVEL_LOW)):
            fig.add_trace(go.Scatter(x=[ts[0], x_known], y=[level, level], mode="lines", name=f"{name} (forming)",
                                     line=dict(color=color, dash="dot", width=1), legendgroup=name, showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[x_known, x_end], y=[level, level], mode="lines", name=f"{name} {level:,.1f}",
                                     line=dict(color=color, width=2), legendgroup=name))
    else:
        fig.add_annotation(text="Asian range unavailable (incomplete Asian session) - no signals this day",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))

    # --- markers
    def markers(mask, y, name, symbol, color, size=cfg.PLOT_MARKER_SIZE):
        sel = day.filter(mask)
        if not sel.height:
            return
        fig.add_trace(go.Scatter(
            x=sel["ts"].to_list(), y=(y(sel) if callable(y) else y), mode="markers", name=name,
            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1, color="black")),
            text=[f"{t} UTC | status={s}" for t, s in zip(sel["ts"].to_list(), sel["signal_status"].to_list())],
            hoverinfo="text",
        ))

    markers(pl.col("sweep_short"), lambda s: s["high"], "sweep (short)", "circle", cfg.PLOT_COLOR_SHORT)
    markers(pl.col("sweep_long"), lambda s: s["low"], "sweep (long)", "circle", cfg.PLOT_COLOR_LONG)
    markers(pl.col("confirm_short"), lambda s: s["high"] + offset, "confirm (short)", "triangle-down", cfg.PLOT_COLOR_SHORT)
    markers(pl.col("confirm_long"), lambda s: s["low"] - offset, "confirm (long)", "triangle-up", cfg.PLOT_COLOR_LONG)
    markers(pl.col("signal_status") == cfg.STATUS_EXPIRED, lambda s: s["close"], "expired", "x", "orange")
    markers(pl.col("signal_status") == cfg.STATUS_REJECTED, lambda s: s["close"], "rejected", "x", "gray")

    fig.update_layout(
        title=f"{day_date} (UTC) - sweep_mode={cfg.SWEEP_MODE}",
        xaxis_title="time (UTC)", yaxis_title="price (USDT)",
        xaxis_rangeslider_visible=False, height=cfg.PLOT_HEIGHT_PX,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=90),
    )
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot sweep/confirmation signals for random UTC days.")
    ap.add_argument("--input", type=Path, default=cfg.SIGNALS_PATH)
    ap.add_argument("--output", type=Path, default=cfg.PLOT_OUTPUT_PATH)
    ap.add_argument("--days", type=int, default=cfg.PLOT_DAYS)
    ap.add_argument("--seed", type=int, default=cfg.PLOT_SEED)
    ap.add_argument("--require-signal", action="store_true", help="sample only days containing at least one sweep")
    ap.add_argument("--dates", nargs="*", type=date.fromisoformat, help="explicit UTC dates (YYYY-MM-DD)")
    args = ap.parse_args()

    setup_logging(log_to_file=False)
    df = load_signals(args.input)
    days = pick_days(df, args.days, args.seed, args.require_signal, args.dates)
    logger.info("plotting %d day(s): %s", len(days), ", ".join(map(str, days)))

    html = ["<html><head><meta charset='utf-8'><title>Signal validation charts</title></head><body>"]
    js_included = False
    for d in days:
        day = df.filter(pl.col("date") == d)
        if day.is_empty():
            logger.warning("no data for %s, skipped", d)
            continue
        n_sweeps = day.filter(pl.col("sweep_short") | pl.col("sweep_long")).height
        n_conf = day.filter(pl.col("confirm_short") | pl.col("confirm_long")).height
        logger.info("%s: %d sweep(s), %d confirmation(s)", d, n_sweeps, n_conf)
        fig = build_day_figure(day, d)
        html.append(pio.to_html(fig, full_html=False, include_plotlyjs=not js_included))   # embed plotly.js once
        js_included = True
    html.append("</body></html>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(html), encoding="utf-8")
    logger.info("charts written: %s  (open it in a browser)", args.output)
    return cfg.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
