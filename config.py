"""
config.py - single source of truth for every constant in the project.

Phase 1 (data ingestion) constants only. Later phases extend this file;
no other module may hard-code a tunable number (structural 0/1 such as
"compare against zero" or "n - 1" are not treated as magic numbers).

All timestamps in this project are UTC.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_zips"                 # download cache
PARQUET_PATH = DATA_DIR / "btc_1m.parquet"
GAPS_PATH = DATA_DIR / "gaps_report.csv"
SUMMARY_PATH = DATA_DIR / "load_summary.txt"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "pipeline.log"

# --------------------------------------------------------------------------
# Data source (Binance Data Vision, USD-M futures, monthly 1m klines)
# --------------------------------------------------------------------------
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
BINANCE_BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
# {base}/{symbol}/{interval}/{symbol}-{interval}-YYYY-MM.zip

# 36 COMPLETE months, inclusive, first-of-month dates. Monthly zips only exist
# for finished months, so the current month is never requested.
# Hard-coded (not "today minus 36 months") so every run is reproducible.
# 2023-11 .. 2026-10 inclusive = 36 months.
DATA_START = date(2023, 11, 1)
DATA_END = date(2026, 10, 1)

# --------------------------------------------------------------------------
# Download behaviour
# --------------------------------------------------------------------------
DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_DOWNLOAD_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2            # doubles each retry: 2s, 4s, 8s...
RETRY_BACKOFF_FACTOR = 2
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
HTTP_NOT_FOUND = 404
PART_SUFFIX = ".part"

# --------------------------------------------------------------------------
# Kline file format
# --------------------------------------------------------------------------
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
# Explicit schema: no inference. Columns we drop are read as strings so an odd
# value in an unused column can never break parsing.
KLINE_SCHEMA = {
    "open_time": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "close_time": pl.String,
    "quote_volume": pl.String,
    "trades": pl.String,
    "taker_buy_base": pl.String,
    "taker_buy_quote": pl.String,
    "ignore": pl.String,
}
KEEP_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
RENAME_MAP = {"open_time": "timestamp"}
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
AFTER_GAP_COLUMN = "after_gap"

# Binance moved some datasets from millisecond to MICROsecond epoch timestamps
# (spot, from 2025-01-01). Futures files are expected to be milliseconds, but we
# detect the unit per file rather than assume. Current epoch is ~1.7e12 in ms
# and ~1.7e15 in us, so anything >= 1e14 cannot be milliseconds.
MICROSECOND_EPOCH_THRESHOLD = 10**14
MICROS_PER_MILLI = 1000
TIME_UNIT = "ms"
TIMEZONE = "UTC"
MS_PER_SECOND = 1000
SECONDS_PER_DAY = 24 * 60 * 60
MONTHS_PER_YEAR = 12

# --------------------------------------------------------------------------
# Validation / gap detection
# --------------------------------------------------------------------------
EXPECTED_BAR_SECONDS = 60            # 1m bars
VALIDATION_SAMPLE_SIZE = 5           # example rows quoted per violation

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"   # rendered in UTC (see logging_setup)

# ==========================================================================
# PHASE 2 - resampling, ATR, sessions, sweeps
# ==========================================================================
SIGNALS_PATH = DATA_DIR / "signals.parquet"

# --------------------------------------------------------------------------
# Resampling 1m -> 5m (group_by_dynamic) and ATR
# --------------------------------------------------------------------------
BAR_MINUTES = 5
RESAMPLE_EVERY = "5m"
RESAMPLE_PERIOD = "5m"
BAR_DURATION = timedelta(minutes=BAR_MINUTES)
MINUTES_PER_HOUR = 60
SOURCE_BARS_PER_BAR = BAR_MINUTES * 60 // EXPECTED_BAR_SECONDS   # 1m bars per 5m bar = 5
ATR_PERIOD = 14                       # ATR(14) on 5m candles, simple rolling mean of true range
ATR_COLUMN = "atr_14"

# --------------------------------------------------------------------------
# Sessions - FIXED UTC HOURS, half-open [start, end).
# NOT DST-aware: real London/NY open times move by one hour when the UK/US
# change clocks. This is a deliberate fixed-UTC approximation.
# Windows must be whole hours (bars are 5m, so hour-granular windows are exact).
# --------------------------------------------------------------------------
ASIAN_WINDOW = (0, 8)
LONDON_WINDOW = (8, 16)
NY_WINDOW = (13, 21)
OVERLAP_WINDOW = (max(LONDON_WINDOW[0], NY_WINDOW[0]), min(LONDON_WINDOW[1], NY_WINDOW[1]))  # 13-16

TAG_ASIAN = "Asian"
TAG_LONDON = "London"
TAG_NY = "NY"
TAG_OVERLAP = "London_NY_overlap"
TAG_OFF = "Off_hours"

# Signal-lifetime sessions: a partition of the post-Asian day, so each Asian
# level gets exactly one "first sweep" per lifetime session.
# The overlap (13-16) belongs to London (the session that opened first), so
# NY's lifetime window starts when London's ends. The *reported* tag of a bar
# is still London_NY_overlap - lifetime ownership is only for first-sweep /
# expiry bookkeeping.
LIFETIME_WINDOWS = {
    TAG_LONDON: LONDON_WINDOW,
    TAG_NY: (max(NY_WINDOW[0], LONDON_WINDOW[1]), NY_WINDOW[1]),
}
ASIAN_EXPECTED_BARS = (ASIAN_WINDOW[1] - ASIAN_WINDOW[0]) * MINUTES_PER_HOUR // BAR_MINUTES   # 96

# --------------------------------------------------------------------------
# Sweep logic
# --------------------------------------------------------------------------
SWEEP_MODE_WICK = "wick_only"
SWEEP_MODE_CLOSE = "close_beyond"
SWEEP_MODE_BOTH = "both"
SWEEP_MODES = (SWEEP_MODE_WICK, SWEEP_MODE_CLOSE, SWEEP_MODE_BOTH)
SWEEP_MODE = SWEEP_MODE_WICK

# False: confirmation must be a LATER bar than the sweep bar ("sweep already
#        happened" - literal reading of the spec).
# True : in wick modes the sweep bar itself already closed back inside the
#        level, so it may confirm itself (classic "sweep candle closes back in").
CONFIRM_ON_SWEEP_BAR = False

STATUS_PENDING = "pending"
STATUS_CONFIRMED = "confirmed"
STATUS_EXPIRED = "expired"
STATUS_REJECTED = "rejected"
# When the short-side and long-side status collide on one bar, show the first match.
STATUS_PRECEDENCE = (STATUS_CONFIRMED, STATUS_EXPIRED, STATUS_REJECTED, STATUS_PENDING)

SIGNAL_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume", ATR_COLUMN, "session_tag",
    "asian_high", "asian_low", "sweep_short", "sweep_long", "confirm_short", "confirm_long",
    "signal_status",
]

# --------------------------------------------------------------------------
# Validation chart (scripts/plot_signals.py)
# --------------------------------------------------------------------------
PLOT_DAYS = 10
PLOT_SEED = 42
PLOT_OUTPUT_PATH = DATA_DIR / "signal_charts.html"
PLOT_HEIGHT_PX = 620
PLOT_MARKER_SIZE = 11
PLOT_MARKER_OFFSET_FRACTION = 0.03    # triangle offset as a fraction of the day's price range
PLOT_SESSION_COLORS = {
    TAG_ASIAN: "rgba(90,110,255,0.10)",
    TAG_LONDON: "rgba(255,165,0,0.13)",
    TAG_OVERLAP: "rgba(190,0,190,0.15)",
    TAG_NY: "rgba(0,170,90,0.12)",
}
PLOT_COLOR_SHORT = "#d62728"
PLOT_COLOR_LONG = "#1f9d55"
PLOT_COLOR_LEVEL_HIGH = "#8b0000"
PLOT_COLOR_LEVEL_LOW = "#006400"

# ==========================================================================
# PHASE 3 - backtester and risk engine
# ==========================================================================
TRADES_PARQUET_PATH = DATA_DIR / "trades.parquet"
TRADES_CSV_PATH = DATA_DIR / "trades.csv"
FUNDING_PATH = DATA_DIR / "funding_rates.parquet"     # optional, see FUNDING_*_COLUMN below
SIGNALS_META_PATH = DATA_DIR / "signals_meta.json"    # written by signal_engine (which sweep_mode made the signals)

# --------------------------------------------------------------------------
# Capital, risk, sizing
# --------------------------------------------------------------------------
STARTING_CAPITAL = 10_000.0
RISK_PCT = 0.01
RISK_USD = STARTING_CAPITAL * RISK_PCT        # fixed $100 per trade (not compounded)
MAX_LEVERAGE = 25                             # margin cap only; size is never leverage-driven
R_MULTIPLE = 2.0                              # take-profit distance = R_MULTIPLE * stop distance

# --------------------------------------------------------------------------
# Stop placement
# --------------------------------------------------------------------------
STOP_BUFFER_PCT = 0.001                       # stop_buffer = max(0.1% of entry, 0.25 * ATR(14) on 5m)
STOP_BUFFER_ATR_MULT = 0.25

# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------
TAKER_FEE_RATE = 0.0005                       # 0.05% per side, on filled notional
SLIPPAGE_BPS = 2                              # per side, always against the trader
BPS_PER_UNIT = 10_000
SLIPPAGE_RATE = SLIPPAGE_BPS / BPS_PER_UNIT

# Funding settles at these UTC hours (minute 0). Positive rate: longs pay shorts.
FUNDING_HOURS = (0, 8, 16)
FUNDING_ESTIMATE_RATE = 0.0001                # 0.01% per 8h, used when no historical rate is available
# Expected schema of data/funding_rates.parquet (optional file):
#   timestamp     Datetime (UTC) - settlement time, truncated to the minute on load
#   funding_rate  Float64        - fraction per 8h, e.g. 0.0001 = 0.01%
FUNDING_TIME_COLUMN = "timestamp"
FUNDING_RATE_COLUMN = "funding_rate"

# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
ONE_MINUTE = timedelta(seconds=EXPECTED_BAR_SECONDS)
# Flat at the end of the LAST session (NY, 21:00 UTC): no overnight positions.
# The forced exit fills at the close of the final 1m candle before this hour.
FORCE_EXIT_HOUR = NY_WINDOW[1]
PRICE_MATCH_REL_TOL = 1e-9                    # 5m open must equal the first 1m open of that bar

DIR_LONG = "long"
DIR_SHORT = "short"
EXIT_TP = "tp"
EXIT_SL = "sl"
EXIT_BOTH_SL_FIRST = "both_sl_first"
EXIT_SESSION_END = "session_end"

TRADE_COLUMNS = [
    "trade_id", "signal_time", "entry_time", "exit_time", "direction", "session_tag", "sweep_mode",
    "sweep_price", "entry_price", "stop_price", "tp_price", "size", "stop_distance", "risk_usd",
    "pnl_gross", "fees", "slippage_cost", "funding_paid", "pnl_net", "r_multiple",
    "sl_tp_both_hit", "funding_estimated", "exit_reason",
    # extras (audit trail for scripts/verify_trade.py)
    "exit_price", "entry_fill_price", "exit_fill_price", "n_funding_boundaries", "margin_capped", "size_uncapped",
]

# ==========================================================================
# PHASE 4 - dashboard and reconciliation
# ==========================================================================
APP_TITLE = "Session Liquidity Sweep - BTCUSDT perpetual backtest"
DASH_LOW_SAMPLE_TRADES = 30           # below this many trades a number is shown with a small-sample warning
DASH_HIST_BIN_SIZE_R = 0.25           # R-multiple histogram bin width
DASH_CHART_HEIGHT_PX = 560
DASH_BREAKDOWN_HEIGHT_PX = 380
DASH_RECON_DECIMALS = 6               # decimals in the reconciliation block (dashboard AND scripts/reconcile.py)
DASH_SESSION_GROUPS = [TAG_LONDON, TAG_OVERLAP, TAG_NY]       # always shown, even with zero trades
DASH_DIRECTION_GROUPS = [DIR_LONG, DIR_SHORT]
DASH_NOT_COMPUTED = "not yet computed"
DASH_PLACEHOLDERS = ["Market Regime", "Volatility State", "Higher-Timeframe Alignment"]
DASH_COLOR_POS = "#26a69a"
DASH_COLOR_NEG = "#ef5350"
DASH_COLOR_NEUTRAL = "#5c6bc0"
DASH_COLOR_DRAWDOWN = "rgba(239,83,80,0.35)"

# --------------------------------------------------------------------------
# Process exit codes
# --------------------------------------------------------------------------
EXIT_OK = 0
EXIT_ABORTED = 1
