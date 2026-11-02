"""
backtester.py - Phase 3: event-driven backtest and risk engine.

Reads  data/signals.parquet  (5m bars + signal flags, Phase 2)
       data/btc_1m.parquet   (1m candles, Phase 1)
       data/funding_rates.parquet (optional)
Writes data/trades.parquet and data/trades.csv

TIMEFRAMES: signals are on 5m candles (ATR(14) on 5m); fills are resolved on 1m candles.

EVENT LOOP AND NO-LOOK-AHEAD
----------------------------
One chronological loop over 5m bars. All market data is read through
MarketView, which owns a clock (`now`) that only moves forward. A bar/candle
is visible only once it has CLOSED (open timestamp + length <= now); reading
anything later raises LookAheadError. Per 5m bar starting at T:
  1. now = T      : an order created after the previous bar closed fills at this
                    bar's OPEN (the open price is known the instant the bar starts).
  2. position open: now steps through the five 1m candles; each candle's
                    stop/TP outcome is judged only after that candle closed.
  3. now = T + 5m : the 5m bar has closed, so ITS flags become visible. A
                    confirmation creates an order for the NEXT bar's open.

COST MODEL (all per trade; sign = +1 long, -1 short)
  entry_fill    = entry * (1 + sign*slip)         exit_fill = exit * (1 - sign*slip)
  pnl_gross     = sign * size * (exit - entry)                     (unslipped prices)
  slippage_cost = size * (|entry_fill-entry| + |exit_fill-exit|)
  fees          = taker_fee * size * (entry_fill + exit_fill)       (filled notional)
  funding_paid  = sum over crossed boundaries of sign * size * price_at_boundary * rate
                  (positive = paid, negative = received; longs pay when rate > 0)
  pnl_net       = pnl_gross - slippage_cost - fees - funding_paid
  r_multiple    = pnl_net / risk_usd
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import numpy as np
import polars as pl

import config as cfg
from utils.logging_setup import get_logger, setup_logging
from utils.validation import DataValidationError, raise_if_invalid, validate_ohlcv

logger = get_logger(__name__)

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
SIDE_FLAGS = ((cfg.DIR_SHORT, "short"), (cfg.DIR_LONG, "long"))     # evaluation order on a shared bar


class LookAheadError(RuntimeError):
    """Raised if code touches market data that has not closed yet at the current clock time."""


class DataGapError(RuntimeError):
    """A 1m candle needed while a position is open is missing."""


# ==========================================================================
# Pure math (unit-tested)
# ==========================================================================
def sign_of(direction: str) -> int:
    return 1 if direction == cfg.DIR_LONG else -1


@dataclass(frozen=True)
class SizeResult:
    size: float
    size_by_risk: float
    size_cap: float
    capped: bool
    risk_actual: float          # dollars lost if the stop fills at the stop price (before costs)


def compute_position_size(entry_price: float, stop_price: float, risk_usd: float,
                          equity: float, max_leverage: float) -> SizeResult:
    """
    size = min(risk_usd / stop_distance, equity * max_leverage / entry_price).
    Risk-based, never leverage-based: leverage only appears as a CAP.
    """
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        raise ValueError("stop distance must be > 0")
    size_by_risk = risk_usd / stop_distance
    size_cap = equity * max_leverage / entry_price
    size = min(size_by_risk, size_cap)
    return SizeResult(size, size_by_risk, size_cap, size_cap < size_by_risk, size * stop_distance)


def compute_stop_tp(direction: str, entry_price: float, sweep_price: float, atr: float) -> tuple[float, float, float]:
    """(stop, take_profit, stop_buffer). Stop sits stop_buffer beyond the sweep extreme; TP is R_MULTIPLE away."""
    buffer = max(cfg.STOP_BUFFER_PCT * entry_price, cfg.STOP_BUFFER_ATR_MULT * atr)
    sign = sign_of(direction)
    stop = sweep_price - sign * buffer                  # long: below the sweep low; short: above the sweep high
    tp = entry_price + sign * cfg.R_MULTIPLE * abs(entry_price - stop)
    return stop, tp, buffer


def stop_is_valid(direction: str, entry_price: float, stop_price: float) -> bool:
    """A long stop must be below entry, a short stop above it (a gap past the sweep extreme can break this)."""
    return stop_price < entry_price if direction == cfg.DIR_LONG else stop_price > entry_price


def resolve_candle(direction: str, stop: float, tp: float, high: float, low: float) -> tuple[str | None, bool]:
    """
    Outcome of ONE 1m candle for an open position -> (exit_reason or None, sl_tp_both_hit).
    Touching a level counts (<= / >=). If the candle's range reaches both, the
    stop is assumed to fill first (pessimistic).
    """
    if direction == cfg.DIR_LONG:
        sl_hit, tp_hit = low <= stop, high >= tp
    else:
        sl_hit, tp_hit = high >= stop, low <= tp
    if sl_hit and tp_hit:
        return cfg.EXIT_BOTH_SL_FIRST, True
    if sl_hit:
        return cfg.EXIT_SL, False
    if tp_hit:
        return cfg.EXIT_TP, False
    return None, False


def exit_price_for(reason: str, stop: float, tp: float, candle_close: float) -> float:
    if reason in (cfg.EXIT_SL, cfg.EXIT_BOTH_SL_FIRST):
        return stop
    if reason == cfg.EXIT_TP:
        return tp
    return candle_close                                  # session_end: close of the last candle


def compute_costs(direction: str, size: float, entry: float, exit_: float) -> dict:
    sign, slip = sign_of(direction), cfg.SLIPPAGE_RATE
    entry_fill = entry * (1 + sign * slip)               # long pays up, short sells lower
    exit_fill = exit_ * (1 - sign * slip)                # long sells lower, short buys higher
    return {
        "entry_fill": entry_fill,
        "exit_fill": exit_fill,
        "pnl_gross": sign * size * (exit_ - entry),
        "slippage_cost": size * (abs(entry_fill - entry) + abs(exit_fill - exit_)),
        "fees": cfg.TAKER_FEE_RATE * size * (entry_fill + exit_fill),
    }


def funding_boundaries(entry_ts: datetime, exit_ts: datetime) -> list[datetime]:
    """
    Funding settlement instants b with  entry_ts < b <= exit_ts.
    `exit_ts` is the OPEN timestamp of the exit candle: the position was open at
    that candle's open, so a boundary equal to it is charged. A boundary equal to
    entry_ts is not (the position did not exist before it).
    """
    out: list[datetime] = []
    day = entry_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= exit_ts:
        for hour in cfg.FUNDING_HOURS:
            boundary = day + timedelta(hours=hour)
            if entry_ts < boundary <= exit_ts:
                out.append(boundary)
        day += timedelta(days=1)
    return out


class FundingSchedule:
    """Historical funding rates keyed by settlement time; falls back to the estimate."""

    def __init__(self, rates: dict[datetime, float] | None = None):
        self._rates = rates

    @property
    def has_history(self) -> bool:
        return self._rates is not None

    def rate_at(self, boundary: datetime) -> tuple[float, bool]:
        """(rate, estimated?)"""
        if self._rates is not None and boundary in self._rates:
            return self._rates[boundary], False
        return cfg.FUNDING_ESTIMATE_RATE, True

    @classmethod
    def from_parquet(cls, path) -> "FundingSchedule":
        if not path.exists():
            logger.warning("no %s -> funding will be ESTIMATED at %.4f%% per 8h (funding_estimated=True)",
                           path.name, cfg.FUNDING_ESTIMATE_RATE * 100)
            return cls(None)
        df = pl.read_parquet(path)
        missing = [c for c in (cfg.FUNDING_TIME_COLUMN, cfg.FUNDING_RATE_COLUMN) if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} lacks columns {missing}")
        df = df.with_columns(pl.col(cfg.FUNDING_TIME_COLUMN).dt.truncate("1m").alias(cfg.FUNDING_TIME_COLUMN))
        rates = dict(zip(df[cfg.FUNDING_TIME_COLUMN].to_list(), df[cfg.FUNDING_RATE_COLUMN].to_list()))
        logger.info("historical funding loaded: %d settlements from %s", len(rates), path.name)
        return cls(rates)


def compute_funding(direction: str, size: float, priced_boundaries: list[tuple[datetime, float]],
                    schedule: FundingSchedule) -> tuple[float, bool, list[tuple[datetime, float, float]]]:
    """
    -> (funding_paid, funding_estimated, [(boundary, rate, charge), ...]).
    `priced_boundaries` = (boundary time, price at that time). Charged exactly once per boundary.
    """
    sign, paid, events = sign_of(direction), 0.0, []
    estimated = not schedule.has_history
    for boundary, price in priced_boundaries:
        rate, was_estimate = schedule.rate_at(boundary)
        estimated = estimated or was_estimate
        charge = sign * size * price * rate
        paid += charge
        events.append((boundary, rate, charge))
    return paid, estimated, events


# ==========================================================================
# MarketView - the only door to market data; enforces visibility
# ==========================================================================
class Row5(NamedTuple):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    atr: float | None
    session_tag: str
    sweep_short: bool
    sweep_long: bool
    confirm_short: bool
    confirm_long: bool
    status: str | None


class MarketView:
    """
    Holds all 5m rows and 1m candles but only returns what has CLOSED by `now`.
    `advance()` moves the clock forward (never back). The event queue (the list
    of bar open times) is exposed through event_time(); prices are not.
    """

    def __init__(self, signals: pl.DataFrame, candles_1m: pl.DataFrame):
        s = signals
        self._ts5 = s["timestamp"].to_list()
        self._o5, self._h5, self._l5, self._c5 = (s[c].to_list() for c in ("open", "high", "low", "close"))
        self._atr5 = s[cfg.ATR_COLUMN].to_list()
        self._tag5 = s["session_tag"].to_list()
        self._flags5 = tuple(s[c].to_list() for c in ("sweep_short", "sweep_long", "confirm_short", "confirm_long"))
        self._status5 = s["signal_status"].to_list()

        self._ts1 = candles_1m["timestamp"].dt.epoch(cfg.TIME_UNIT).to_numpy()
        self._o1, self._h1, self._l1, self._c1 = (candles_1m[c].to_numpy() for c in ("open", "high", "low", "close"))
        self.now: datetime = EPOCH

    # ---- clock
    def advance(self, t: datetime) -> None:
        if t < self.now:
            raise LookAheadError(f"clock moved backwards: {t} < {self.now}")
        self.now = t

    @property
    def n_bars(self) -> int:
        return len(self._ts5)

    def event_time(self, i: int) -> datetime:
        return self._ts5[i]

    def _require_closed(self, close_time: datetime, what: str) -> None:
        if close_time > self.now:
            raise LookAheadError(f"{what} closes at {close_time} but clock is {self.now}")

    # ---- 5m
    def bar_open(self, i: int) -> float:
        if self._ts5[i] > self.now:
            raise LookAheadError(f"open of 5m bar {self._ts5[i]} read before it started (now={self.now})")
        return self._o5[i]

    def closed_bar(self, i: int) -> Row5:
        self._require_closed(self._ts5[i] + cfg.BAR_DURATION, f"5m bar {self._ts5[i]}")
        f = self._flags5
        return Row5(self._ts5[i], self._o5[i], self._h5[i], self._l5[i], self._c5[i], self._atr5[i],
                    self._tag5[i], f[0][i], f[1][i], f[2][i], f[3][i], self._status5[i])

    def extreme_between(self, direction: str, first: int, last: int) -> float:
        """Sweep extreme: highest high (short) / lowest low (long) over closed 5m bars first..last inclusive."""
        self._require_closed(self._ts5[last] + cfg.BAR_DURATION, f"5m bar {self._ts5[last]}")
        if direction == cfg.DIR_SHORT:
            return max(self._h5[first:last + 1])
        return min(self._l5[first:last + 1])

    def tag_at(self, i: int) -> str:
        self._require_closed(self._ts5[i] + cfg.BAR_DURATION, f"5m bar {self._ts5[i]}")
        return self._tag5[i]

    # ---- 1m
    @property
    def n_candles(self) -> int:
        return len(self._ts1)

    def idx_1m(self, ts: datetime) -> int | None:
        """Index of the 1m candle opening exactly at ts, or None. (Locates; reveals no prices.)"""
        ms = int(round(ts.timestamp() * cfg.MS_PER_SECOND))
        j = int(np.searchsorted(self._ts1, ms))
        return j if j < len(self._ts1) and self._ts1[j] == ms else None

    def candle_ts(self, idx: int) -> datetime:
        ts = EPOCH + timedelta(milliseconds=int(self._ts1[idx]))
        if ts > self.now:
            raise LookAheadError(f"1m candle {ts} has not started (now={self.now})")
        return ts

    def open_1m(self, idx: int) -> float:
        self.candle_ts(idx)                                # raises if the candle has not started
        return float(self._o1[idx])

    def closed_1m(self, idx: int) -> tuple[datetime, float, float, float, float]:
        ts = EPOCH + timedelta(milliseconds=int(self._ts1[idx]))
        self._require_closed(ts + cfg.ONE_MINUTE, f"1m candle {ts}")
        return ts, float(self._o1[idx]), float(self._h1[idx]), float(self._l1[idx]), float(self._c1[idx])


# ==========================================================================
# Orders, positions, results
# ==========================================================================
@dataclass
class Order:
    direction: str
    confirm_ts: datetime            # open label of the confirmation 5m bar
    entry_ts: datetime              # = confirm_ts + 5m: the next bar's open
    sweep_price: float
    atr: float
    session_tag: str                # tag of the SWEEP bar (reported separately, incl. London_NY_overlap)


@dataclass
class Position:
    direction: str
    signal_ts: datetime
    entry_ts: datetime
    entry_price: float
    stop: float
    tp: float
    size: float
    size_uncapped: float
    stop_distance: float
    risk_actual: float
    margin_capped: bool
    sweep_price: float
    session_tag: str


@dataclass
class ExitEvent:
    ts: datetime                    # open timestamp of the 1m candle that triggered the fill
    reason: str
    both_hit: bool
    price: float


@dataclass
class BacktestResult:
    trades: pl.DataFrame
    rejections: list[dict] = field(default_factory=list)
    discarded: list[dict] = field(default_factory=list)


def _is_force_exit_candle(candle_ts: datetime) -> bool:
    end = candle_ts + cfg.ONE_MINUTE
    return end.hour == cfg.FORCE_EXIT_HOUR and end.minute == 0


def walk_bar(view: MarketView, pos: Position, bar_ts: datetime) -> ExitEvent | None:
    """
    Walk the five 1m candles of the 5m bar starting at bar_ts, in order. The
    clock is advanced to each candle's CLOSE before its high/low are used.
    """
    idx0 = view.idx_1m(bar_ts)
    if idx0 is None:
        raise DataGapError(f"no 1m candle at {bar_ts}")
    for k in range(cfg.SOURCE_BARS_PER_BAR):
        expected = bar_ts + k * cfg.ONE_MINUTE
        idx = idx0 + k
        if idx >= view.n_candles or view.candle_ts(idx) != expected:
            raise DataGapError(f"missing 1m candle at {expected}")
        view.advance(expected + cfg.ONE_MINUTE)                     # candle k has now closed
        _, _, high, low, close = view.closed_1m(idx)
        reason, both = resolve_candle(pos.direction, pos.stop, pos.tp, high, low)
        if reason is not None:
            return ExitEvent(expected, reason, both, exit_price_for(reason, pos.stop, pos.tp, close))
        if _is_force_exit_candle(expected):
            return ExitEvent(expected, cfg.EXIT_SESSION_END, False, close)
    return None


# ==========================================================================
# The event loop
# ==========================================================================
def _reject(rejections: list[dict], ts: datetime, direction: str, reason: str) -> None:
    rejections.append({"signal_time": ts, "direction": direction, "reason": reason})
    logger.warning("REJECTED order: %s signal confirmed on bar %s - %s", direction, ts, reason)


def _open_position(view: MarketView, order: Order, bar_i: int, equity: float,
                   rejections: list[dict]) -> Position | None:
    T = view.event_time(bar_i)
    # ENTRY RULE: the fill is the OPEN of the bar AFTER the confirmation bar - never the confirmation close.
    assert T == order.confirm_ts + cfg.BAR_DURATION, "entry bar must be the 5m bar immediately after the confirmation bar"
    assert T > order.confirm_ts
    entry_price = view.bar_open(bar_i)

    if T.hour >= cfg.FORCE_EXIT_HOUR:
        _reject(rejections, order.confirm_ts, order.direction, f"entry bar {T} is at/after the {cfg.FORCE_EXIT_HOUR}:00 UTC forced-exit time")
        return None
    idx1 = view.idx_1m(T)
    if idx1 is None:
        _reject(rejections, order.confirm_ts, order.direction, f"no 1m candle at entry time {T}")
        return None
    open_1m = view.open_1m(idx1)
    if not math.isclose(entry_price, open_1m, rel_tol=cfg.PRICE_MATCH_REL_TOL):
        _reject(rejections, order.confirm_ts, order.direction,
                f"5m open {entry_price} != first 1m open {open_1m} (signals/candles from different data?)")
        return None

    stop, tp, buffer = compute_stop_tp(order.direction, entry_price, order.sweep_price, order.atr)
    if not stop_is_valid(order.direction, entry_price, stop):
        _reject(rejections, order.confirm_ts, order.direction,
                f"stop {stop:.2f} is not beyond entry {entry_price:.2f} (price already past the sweep extreme)")
        return None
    if equity <= 0:
        _reject(rejections, order.confirm_ts, order.direction, f"no equity left ({equity:.2f})")
        return None

    sizing = compute_position_size(entry_price, stop, cfg.RISK_USD, equity, cfg.MAX_LEVERAGE)
    if sizing.capped:
        logger.warning("MARGIN CAP reduced size on %s %s: %.6f -> %.6f (risk at stop $%.2f instead of $%.2f)",
                       order.direction, T, sizing.size_by_risk, sizing.size, sizing.risk_actual, cfg.RISK_USD)
    return Position(
        direction=order.direction, signal_ts=order.confirm_ts, entry_ts=T, entry_price=entry_price,
        stop=stop, tp=tp, size=sizing.size, size_uncapped=sizing.size_by_risk,
        stop_distance=abs(entry_price - stop), risk_actual=sizing.risk_actual, margin_capped=sizing.capped,
        sweep_price=order.sweep_price, session_tag=order.session_tag,
    )


def _close_trade(view: MarketView, pos: Position, ev: ExitEvent, funding: FundingSchedule,
                 sweep_mode: str, trade_id: int) -> dict:
    costs = compute_costs(pos.direction, pos.size, pos.entry_price, ev.price)

    boundaries = funding_boundaries(pos.entry_ts, ev.ts)
    priced = []
    for b in boundaries:
        idx = view.idx_1m(b)
        if idx is None:
            raise DataGapError(f"no 1m candle at funding boundary {b}")
        priced.append((b, view.open_1m(idx)))                       # boundary <= exit candle, so already visible
    funding_paid, funding_estimated, events = compute_funding(pos.direction, pos.size, priced, funding)
    for b, rate, charge in events:
        logger.info("FUNDING trade %d: %s boundary %s rate=%.6f charge=%+.4f", trade_id, pos.direction, b, rate, charge)

    pnl_net = costs["pnl_gross"] - costs["slippage_cost"] - costs["fees"] - funding_paid
    if ev.both_hit:
        logger.warning("SL/TP COLLISION trade %d: one 1m candle at %s spans stop %.2f and tp %.2f -> stop assumed first",
                       trade_id, ev.ts, pos.stop, pos.tp)
    row = {
        "trade_id": trade_id, "signal_time": pos.signal_ts, "entry_time": pos.entry_ts, "exit_time": ev.ts,
        "direction": pos.direction, "session_tag": pos.session_tag, "sweep_mode": sweep_mode,
        "sweep_price": pos.sweep_price, "entry_price": pos.entry_price, "stop_price": pos.stop, "tp_price": pos.tp,
        "size": pos.size, "stop_distance": pos.stop_distance, "risk_usd": pos.risk_actual,
        "pnl_gross": costs["pnl_gross"], "fees": costs["fees"], "slippage_cost": costs["slippage_cost"],
        "funding_paid": funding_paid, "pnl_net": pnl_net, "r_multiple": pnl_net / pos.risk_actual,
        "sl_tp_both_hit": ev.both_hit, "funding_estimated": funding_estimated, "exit_reason": ev.reason,
        "exit_price": ev.price, "entry_fill_price": costs["entry_fill"], "exit_fill_price": costs["exit_fill"],
        "n_funding_boundaries": len(boundaries), "margin_capped": pos.margin_capped, "size_uncapped": pos.size_uncapped,
    }
    logger.info("TRADE %d %s %s -> %s exit=%s pnl_net=%+.2f (%+.2fR)", trade_id, pos.direction, pos.entry_ts,
                ev.ts, ev.reason, pnl_net, row["r_multiple"])
    return row


def run_backtest(signals: pl.DataFrame, candles_1m: pl.DataFrame, funding: FundingSchedule,
                 sweep_mode: str, starting_capital: float = cfg.STARTING_CAPITAL) -> BacktestResult:
    view = MarketView(signals, candles_1m)
    rows: list[dict] = []
    rejections: list[dict] = []
    discarded: list[dict] = []
    equity = starting_capital
    pending: Order | None = None
    position: Position | None = None
    last_sweep: dict[str, int | None] = {cfg.DIR_SHORT: None, cfg.DIR_LONG: None}   # index of latest CLOSED sweep bar
    prev_ts: datetime | None = None

    for i in range(view.n_bars):
        T = view.event_time(i)
        view.advance(T)                                              # bar i opens: its open price is now known

        # A hole in the 5m series while a position is open means unknown price action.
        if position is not None and prev_ts is not None and T != prev_ts + cfg.BAR_DURATION:
            logger.error("DISCARDED trade %s %s: 5m series jumps from %s to %s while position open (data gap)",
                         position.direction, position.entry_ts, prev_ts, T)
            discarded.append({"entry_time": position.entry_ts, "direction": position.direction, "reason": "data gap"})
            position = None

        # (1) fill the order created after the previous bar closed
        if pending is not None:
            if pending.entry_ts == T and position is None:
                position = _open_position(view, pending, i, equity, rejections)
            elif pending.entry_ts != T:
                _reject(rejections, pending.confirm_ts, pending.direction, f"entry bar {pending.entry_ts} missing from data")
            pending = None

        # (2) manage the open position through this bar's 1m candles
        if position is not None:
            try:
                ev = walk_bar(view, position, T)
                if ev is not None:
                    row = _close_trade(view, position, ev, funding, sweep_mode, len(rows) + 1)
                    rows.append(row)
                    equity += row["pnl_net"]
                    position = None
            except DataGapError as exc:
                logger.error("DISCARDED trade %s %s: %s", position.direction, position.entry_ts, exc)
                discarded.append({"entry_time": position.entry_ts, "direction": position.direction, "reason": str(exc)})
                position = None

        # (3) bar i closes: its flags become visible
        view.advance(T + cfg.BAR_DURATION)
        bar = view.closed_bar(i)
        if bar.sweep_short:
            last_sweep[cfg.DIR_SHORT] = i
        if bar.sweep_long:
            last_sweep[cfg.DIR_LONG] = i

        for direction, _ in SIDE_FLAGS:
            confirmed = bar.confirm_short if direction == cfg.DIR_SHORT else bar.confirm_long
            if not confirmed:
                continue
            assert bar.status == cfg.STATUS_CONFIRMED, f"confirm flag without 'confirmed' status at {bar.ts}"
            if position is not None or pending is not None:
                _reject(rejections, bar.ts, direction, "another position or order is already active (one at a time)")
                continue
            sweep_i = last_sweep[direction]
            if sweep_i is None or view.event_time(sweep_i).date() != bar.ts.date():
                _reject(rejections, bar.ts, direction, "no sweep bar found for this confirmation (inconsistent signals)")
                continue
            if bar.atr is None:
                _reject(rejections, bar.ts, direction, "ATR(14) unavailable at confirmation bar - cannot size stop buffer")
                continue
            pending = Order(direction, bar.ts, bar.ts + cfg.BAR_DURATION,
                            view.extreme_between(direction, sweep_i, i), bar.atr, view.tag_at(sweep_i))
        prev_ts = T

    if pending is not None:
        _reject(rejections, pending.confirm_ts, pending.direction, "data ended before the entry bar")
    if position is not None:
        logger.error("DISCARDED trade %s %s: data ended while position open", position.direction, position.entry_ts)
        discarded.append({"entry_time": position.entry_ts, "direction": position.direction, "reason": "data ended"})

    return BacktestResult(_trades_frame(rows), rejections, discarded)


# ==========================================================================
# Output / reporting
# ==========================================================================
_DT = pl.Datetime(cfg.TIME_UNIT, cfg.TIMEZONE)
TRADE_SCHEMA = {
    "trade_id": pl.Int64, "signal_time": _DT, "entry_time": _DT, "exit_time": _DT,
    "direction": pl.String, "session_tag": pl.String, "sweep_mode": pl.String,
    "sweep_price": pl.Float64, "entry_price": pl.Float64, "stop_price": pl.Float64, "tp_price": pl.Float64,
    "size": pl.Float64, "stop_distance": pl.Float64, "risk_usd": pl.Float64,
    "pnl_gross": pl.Float64, "fees": pl.Float64, "slippage_cost": pl.Float64, "funding_paid": pl.Float64,
    "pnl_net": pl.Float64, "r_multiple": pl.Float64, "sl_tp_both_hit": pl.Boolean,
    "funding_estimated": pl.Boolean, "exit_reason": pl.String,
    "exit_price": pl.Float64, "entry_fill_price": pl.Float64, "exit_fill_price": pl.Float64,
    "n_funding_boundaries": pl.Int64, "margin_capped": pl.Boolean, "size_uncapped": pl.Float64,
}


def _trades_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=TRADE_SCHEMA).select(cfg.TRADE_COLUMNS)


def log_summary(result: BacktestResult) -> None:
    t = result.trades
    logger.info("backtest finished: %d trades, %d rejected orders, %d discarded (data gap)",
                t.height, len(result.rejections), len(result.discarded))
    if result.rejections:
        reasons = pl.DataFrame(result.rejections).group_by("reason").len().sort("len", descending=True)
        for r in reasons.iter_rows(named=True):
            logger.info("  rejected x%d: %s", r["len"], r["reason"])
    if t.is_empty():
        return
    logger.info("net pnl $%.2f | win rate %.1f%% | avg R %.3f | SL/TP collisions %d | margin-capped %d | funding estimated on %d",
                t["pnl_net"].sum(), 100 * (t["pnl_net"] > 0).mean(), t["r_multiple"].mean(),
                t["sl_tp_both_hit"].sum(), t["margin_capped"].sum(), t["funding_estimated"].sum())
    by_tag = (t.group_by("session_tag")
              .agg(pl.len().alias("n"), (pl.col("pnl_net") > 0).mean().alias("win"),
                   pl.col("pnl_net").sum().alias("net"), pl.col("r_multiple").mean().alias("avg_r"))
              .sort("session_tag"))
    logger.info("by sweep-bar session tag (overlap reported separately):")
    for r in by_tag.iter_rows(named=True):
        logger.info("  %-18s n=%d win=%.1f%% net=$%.2f avgR=%.3f", r["session_tag"], r["n"], 100 * r["win"], r["net"], r["avg_r"])


def _sweep_mode_used() -> str:
    if cfg.SIGNALS_META_PATH.exists():
        meta = json.loads(cfg.SIGNALS_META_PATH.read_text(encoding="utf-8"))
        if meta.get("sweep_mode") != cfg.SWEEP_MODE:
            logger.warning("signals were generated with sweep_mode=%s but config says %s; using the signals' mode",
                           meta.get("sweep_mode"), cfg.SWEEP_MODE)
        return meta["sweep_mode"]
    logger.warning("%s missing; assuming signals were generated with config SWEEP_MODE=%s",
                   cfg.SIGNALS_META_PATH.name, cfg.SWEEP_MODE)
    return cfg.SWEEP_MODE


def main() -> int:
    setup_logging()
    logger.info("Phase 3 started: risk $%.2f/trade, %.1fR target, fee %.3f%%, slippage %d bps/side, force-exit %02d:00 UTC",
                cfg.RISK_USD, cfg.R_MULTIPLE, cfg.TAKER_FEE_RATE * 100, cfg.SLIPPAGE_BPS, cfg.FORCE_EXIT_HOUR)
    signals = pl.read_parquet(cfg.SIGNALS_PATH)
    candles = pl.read_parquet(cfg.PARQUET_PATH).select(cfg.OUTPUT_COLUMNS)
    try:
        raise_if_invalid(validate_ohlcv(signals, label="signals.parquet"))
        raise_if_invalid(validate_ohlcv(candles, label="btc_1m.parquet"))
    except DataValidationError:
        logger.critical("ABORT: input failed validation")
        return cfg.EXIT_ABORTED

    result = run_backtest(signals, candles, FundingSchedule.from_parquet(cfg.FUNDING_PATH), _sweep_mode_used())
    cfg.TRADES_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.trades.write_parquet(cfg.TRADES_PARQUET_PATH)
    result.trades.write_csv(cfg.TRADES_CSV_PATH)
    logger.info("trades written: %s and %s", cfg.TRADES_PARQUET_PATH, cfg.TRADES_CSV_PATH)
    log_summary(result)
    return cfg.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
