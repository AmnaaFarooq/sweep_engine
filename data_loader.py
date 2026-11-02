"""
data_loader.py - Phase 1: historical 1m kline ingestion for Binance USD-M BTCUSDT.

Pipeline (one pass per month, then one global pass):
    download zip (cached) -> parse CSV -> normalise timestamps to UTC
    -> validate month -> ... -> concat -> validate whole series
    -> gap report (nothing filled) -> flag bars after gaps
    -> write parquet + gaps_report.csv + load_summary.txt

Source: Binance Data Vision monthly zips. No ccxt fallback is used.
Timeframe at this stage: 1m (resampling to 5m happens in signal_engine, Phase 2).

Run from the project root:   python data_loader.py
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import requests

import config as cfg
from utils.logging_setup import get_logger, setup_logging
from utils.validation import (
    DataValidationError,
    build_gap_report,
    find_gaps,
    flag_after_gap,
    raise_if_invalid,
    validate_ohlcv,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Month helpers
# --------------------------------------------------------------------------
def month_range(start: date, end: date) -> list[date]:
    """First-of-month dates from start to end, inclusive."""
    months, current = [], date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while current <= last:
        months.append(current)
        current = next_month(current)
    return months


def next_month(d: date) -> date:
    return date(d.year + d.month // cfg.MONTHS_PER_YEAR, d.month % cfg.MONTHS_PER_YEAR + 1, 1)


def month_tag(month: date) -> str:
    return f"{month.year:04d}-{month.month:02d}"


def zip_name(month: date) -> str:
    return f"{cfg.SYMBOL}-{cfg.INTERVAL}-{month_tag(month)}.zip"


def zip_url(month: date) -> str:
    return f"{cfg.BINANCE_BASE_URL}/{cfg.SYMBOL}/{cfg.INTERVAL}/{zip_name(month)}"


def expected_rows(month: date) -> int:
    """Number of 1m bars a complete calendar month would contain."""
    seconds = (next_month(month) - month).days * cfg.SECONDS_PER_DAY
    return seconds // cfg.EXPECTED_BAR_SECONDS


# --------------------------------------------------------------------------
# Download (with cache)
# --------------------------------------------------------------------------
def _zip_is_healthy(path: Path) -> bool:
    try:
        if path.stat().st_size == 0 or not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None        # None means every member's CRC is fine
    except (OSError, zipfile.BadZipFile):
        return False


def download_month(session: requests.Session, month: date) -> Path | None:
    """
    Return the local zip path for `month`, downloading it if not cached.
    Returns None if the file does not exist on the server (HTTP 404) or all
    retries failed. Every attempt is logged.
    """
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = cfg.RAW_DIR / zip_name(month)
    url = zip_url(month)

    if dest.exists():
        if _zip_is_healthy(dest):
            logger.info("cache hit, skipping download: %s", dest.name)
            return dest
        logger.warning("cached file is corrupt, deleting and re-downloading: %s", dest.name)
        dest.unlink()

    part = dest.with_name(dest.name + cfg.PART_SUFFIX)
    delay = cfg.RETRY_BACKOFF_SECONDS
    for attempt in range(1, cfg.MAX_DOWNLOAD_RETRIES + 1):
        logger.info("download attempt %d/%d: %s", attempt, cfg.MAX_DOWNLOAD_RETRIES, url)
        try:
            with session.get(url, stream=True, timeout=cfg.DOWNLOAD_TIMEOUT_SECONDS) as resp:
                if resp.status_code == cfg.HTTP_NOT_FOUND:
                    logger.error("HTTP %d (file not published) for %s", resp.status_code, url)
                    return None                           # retrying a 404 is pointless
                resp.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=cfg.DOWNLOAD_CHUNK_BYTES):
                        fh.write(chunk)
            part.replace(dest)
            if not _zip_is_healthy(dest):
                raise zipfile.BadZipFile("downloaded file failed integrity test")
            logger.info("downloaded %s (%d bytes)", dest.name, dest.stat().st_size)
            return dest
        except (requests.RequestException, OSError, zipfile.BadZipFile) as exc:
            logger.warning("download attempt %d failed for %s: %s", attempt, dest.name, exc)
            for leftover in (part, dest):
                leftover.unlink(missing_ok=True)
            if attempt < cfg.MAX_DOWNLOAD_RETRIES:
                time.sleep(delay)
                delay *= cfg.RETRY_BACKOFF_FACTOR
    logger.error("giving up on %s after %d attempts", dest.name, cfg.MAX_DOWNLOAD_RETRIES)
    return None


# --------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------
def normalize_timestamps(df: pl.DataFrame, label: str) -> pl.DataFrame:
    """
    Convert raw epoch integers to Datetime[ms, UTC].
    Detects microsecond epochs (>= threshold) and converts them to ms, with a
    warning - Binance changed units on some datasets, so we never assume.
    """
    max_raw = df["timestamp"].max()
    if max_raw is not None and max_raw >= cfg.MICROSECOND_EPOCH_THRESHOLD:
        logger.warning("[%s] timestamps look like MICROseconds; converting to milliseconds", label)
        df = df.with_columns(pl.col("timestamp") // cfg.MICROS_PER_MILLI)
    return df.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit=cfg.TIME_UNIT)
        .cast(pl.Datetime(cfg.TIME_UNIT))              # pin precision explicitly
        .dt.replace_time_zone(cfg.TIMEZONE)
        .alias("timestamp")
    )


def read_month_zip(path: Path, label: str) -> pl.DataFrame:
    """
    Parse one monthly zip into [timestamp, open, high, low, close, volume].
    Row order is preserved exactly as in the file (validation checks it as-is).
    Handles both header and header-less CSVs.
    """
    with zipfile.ZipFile(path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"{path.name}: expected exactly one CSV, found {csv_names}")
        raw = zf.read(csv_names[0])

    first_field = raw.split(b"\n", 1)[0].split(b",", 1)[0].strip()
    has_header = not first_field.isdigit()          # a data row starts with a numeric open_time
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=False,
        skip_rows=int(has_header),
        new_columns=cfg.KLINE_COLUMNS,
        schema=cfg.KLINE_SCHEMA,
    )
    df = df.select(cfg.KEEP_COLUMNS).rename(cfg.RENAME_MAP)
    return normalize_timestamps(df, label)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
@dataclass
class MonthStat:
    month: date
    status: str = "ok"                 # ok | missing | invalid | parse_error
    rows: int = 0
    min_ts: object = None
    max_ts: object = None
    note: str = ""


@dataclass
class RunReport:
    months: list[MonthStat] = field(default_factory=list)
    aborts: list[str] = field(default_factory=list)
    total_rows: int = 0
    min_ts: object = None
    max_ts: object = None
    n_gaps: int = 0
    n_missing_bars: int = 0
    n_excluded_bars: int = 0

    @property
    def aborted(self) -> bool:
        return bool(self.aborts)


def write_summary(report: RunReport) -> None:
    status = "ABORTED" if report.aborted else "OK"
    lines = [
        "LOAD SUMMARY",
        f"generated (UTC)   : {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
        f"source            : {cfg.BINANCE_BASE_URL}/{cfg.SYMBOL}/{cfg.INTERVAL}/",
        f"requested months  : {month_tag(cfg.DATA_START)} .. {month_tag(cfg.DATA_END)}",
        f"STATUS            : {status}",
        "",
        f"{'month':<8} {'status':<12} {'rows':>8} {'expected':>9} {'short_by':>9}  min_ts .. max_ts",
    ]
    for m in report.months:
        exp = expected_rows(m.month)
        short = exp - m.rows if m.status == "ok" else exp
        span = f"{m.min_ts} .. {m.max_ts}" if m.min_ts is not None else "-"
        lines.append(f"{month_tag(m.month):<8} {m.status:<12} {m.rows:>8} {exp:>9} {short:>9}  {span}")
        if m.note:
            lines.append(f"         note: {m.note}")

    missing = [month_tag(m.month) for m in report.months if m.status == "missing"]
    lines += [
        "",
        f"total rows        : {report.total_rows}",
        f"min timestamp     : {report.min_ts}",
        f"max timestamp     : {report.max_ts}",
        f"gaps              : {report.n_gaps}",
        f"missing 1m bars   : {report.n_missing_bars}",
        f"excluded bars     : {report.n_excluded_bars}  (first bar after each gap; flagged '{cfg.AFTER_GAP_COLUMN}')",
        f"months not on server (404/failed): {missing if missing else 'none'}",
        "",
        "validation aborts :",
    ]
    lines += [f"  {a}" for a in report.aborts] if report.aborts else ["  none"]

    cfg.SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary written: %s", cfg.SUMMARY_PATH)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_dataset() -> tuple[pl.DataFrame | None, pl.DataFrame | None, RunReport]:
    """Download, parse, validate. Returns (clean_df, gap_report_df, RunReport)."""
    report = RunReport()
    frames: list[pl.DataFrame] = []

    with requests.Session() as session:
        for month in month_range(cfg.DATA_START, cfg.DATA_END):
            tag = month_tag(month)
            stat = MonthStat(month=month)
            report.months.append(stat)

            path = download_month(session, month)
            if path is None:
                stat.status, stat.note = "missing", "zip unavailable; a mid-series hole shows in the gap report, an edge month just shortens the series"
                logger.error("[%s] zip unavailable - continuing without this month", tag)
                continue

            try:
                df = read_month_zip(path, tag)
            except Exception as exc:                   # parse failure = unusable data = abort
                stat.status, stat.note = "parse_error", str(exc)
                report.aborts.append(f"[{tag}] could not parse: {exc}")
                logger.error("[%s] parse error: %s", tag, exc)
                continue

            result = validate_ohlcv(df, label=tag)
            try:
                raise_if_invalid(result)
            except DataValidationError:
                stat.status = "invalid"
                report.aborts.append(result.describe())
                continue

            stat.rows, stat.min_ts, stat.max_ts = df.height, df["timestamp"].min(), df["timestamp"].max()
            logger.info("[%s] %d rows, %s .. %s", tag, stat.rows, stat.min_ts, stat.max_ts)
            frames.append(df)

    if report.aborted:
        logger.critical("ABORT: %d month(s) failed validation/parsing; no parquet written", len(report.aborts))
        return None, None, report
    if not frames:
        report.aborts.append("no month could be loaded")
        logger.critical("ABORT: no data loaded")
        return None, None, report

    # Cross-month check: catches overlapping months / out-of-order boundaries.
    # Months are concatenated in calendar order and NOT sorted - order must already be right.
    full = pl.concat(frames)
    result = validate_ohlcv(full, label="ALL")
    try:
        raise_if_invalid(result)
    except DataValidationError:
        report.aborts.append(result.describe())
        logger.critical("ABORT: combined series failed validation; no parquet written")
        return None, None, report

    gaps = find_gaps(full)
    gap_report = build_gap_report(full)
    for g in gaps.iter_rows(named=True):
        logger.warning(
            "GAP #%d: %d missing 1m bar(s) between %s and %s; bar at %s excluded from signal generation (not filled)",
            g["gap_id"], g["gap_missing_bars"], g["prev_timestamp"], g["next_timestamp"], g["next_timestamp"],
        )

    clean = flag_after_gap(full)
    report.total_rows = clean.height
    report.min_ts, report.max_ts = clean["timestamp"].min(), clean["timestamp"].max()
    report.n_gaps = gaps.height
    report.n_missing_bars = gap_report.height
    report.n_excluded_bars = int(clean[cfg.AFTER_GAP_COLUMN].sum())
    return clean, gap_report, report


def save_outputs(clean: pl.DataFrame, gap_report: pl.DataFrame) -> None:
    cfg.PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.PARQUET_PATH.with_name(cfg.PARQUET_PATH.name + cfg.PART_SUFFIX)
    clean.write_parquet(tmp)
    tmp.replace(cfg.PARQUET_PATH)                  # atomic swap: never leaves a half-written file
    logger.info("parquet written: %s (%d rows)", cfg.PARQUET_PATH, clean.height)

    gap_report.write_csv(cfg.GAPS_PATH)
    logger.info("gap report written: %s (%d missing bars)", cfg.GAPS_PATH, gap_report.height)


def main() -> int:
    setup_logging()
    logger.info("Phase 1 load started: %s %s, %s .. %s",
                cfg.SYMBOL, cfg.INTERVAL, month_tag(cfg.DATA_START), month_tag(cfg.DATA_END))
    clean, gap_report, report = build_dataset()
    if clean is not None and gap_report is not None:
        save_outputs(clean, gap_report)
    write_summary(report)
    if report.aborted:
        return cfg.EXIT_ABORTED
    logger.info("Phase 1 load finished OK")
    return cfg.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
