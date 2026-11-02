"""
verify_trade.py - independent audit of logged trades. It does NOT import backtester.py.

For a trade_id it re-derives, from the RAW files (signals.parquet, btc_1m.parquet,
funding_rates.parquet if present) and the constants in config.py:
  signal chain  : confirmation bar, sweep bar, sweep extreme (max high / min low from sweep to confirmation)
  entry         : 5m open of the bar after confirmation == first 1m open at entry_time
  stop / tp     : buffer = max(0.1% of entry, 0.25 * ATR14[confirmation bar]); tp = entry -/+ 2R
  size          : min(risk / stop distance, equity * 25 / entry), equity rebuilt from earlier trades
  1m walk       : from the entry candle, first candle touching stop / tp / forced-exit, stop-first on collisions
  costs         : slippage, fees, funding boundaries (+ the 1m open used at each), pnl_net, R
and prints every number next to the logged value with PASS / FAIL. Any FAIL -> exit code 1.

Usage (project root):
    python scripts/verify_trade.py 17
    python scripts/verify_trade.py --random 5 --seed 3
    python scripts/verify_trade.py 17 --context 5        # more candles around entry/exit
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

import config as cfg

LINE = "-" * 100


class Audit:
    def __init__(self):
        self.failures: list[str] = []

    def check(self, label: str, logged, recomputed, tol: float = 1e-9) -> None:
        if isinstance(logged, (int, float)) and not isinstance(logged, bool) and isinstance(recomputed, (int, float)):
            ok = math.isclose(logged, recomputed, rel_tol=tol, abs_tol=tol)
            shown = f"logged {logged:,.8f} | recomputed {recomputed:,.8f}"
        else:
            ok = logged == recomputed
            shown = f"logged {logged} | recomputed {recomputed}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<26} {shown}")
        if not ok:
            self.failures.append(label)


def fmt_candle(ts, o, h, l, c, note="") -> str:
    return f"    {ts:%Y-%m-%d %H:%M} | O {o:>12,.2f} H {h:>12,.2f} L {l:>12,.2f} C {c:>12,.2f} | {note}"


def load_inputs():
    for path in (cfg.TRADES_PARQUET_PATH, cfg.SIGNALS_PATH, cfg.PARQUET_PATH):
        if not path.exists():
            sys.exit(f"missing {path} - run the earlier phases first")
    trades = pl.read_parquet(cfg.TRADES_PARQUET_PATH)
    signals = pl.read_parquet(cfg.SIGNALS_PATH)
    candles = pl.read_parquet(cfg.PARQUET_PATH)
    funding = None
    if cfg.FUNDING_PATH.exists():
        f = pl.read_parquet(cfg.FUNDING_PATH).with_columns(pl.col(cfg.FUNDING_TIME_COLUMN).dt.truncate("1m"))
        funding = dict(zip(f[cfg.FUNDING_TIME_COLUMN].to_list(), f[cfg.FUNDING_RATE_COLUMN].to_list()))
    return trades, signals, candles, funding


def verify(trade_id: int, trades, signals, candles, funding, context: int) -> bool:
    rows = trades.filter(pl.col("trade_id") == trade_id)
    if rows.is_empty():
        print(f"trade {trade_id} not found (valid ids: 1..{trades.height})")
        return False
    t = rows.row(0, named=True)
    direction = t["direction"]
    sign = 1 if direction == cfg.DIR_LONG else -1
    audit = Audit()
    bar, minute = cfg.BAR_DURATION, cfg.ONE_MINUTE

    print("\n" + "=" * 100)
    print(f"TRADE {trade_id}: {direction.upper()}  session_tag={t['session_tag']}  sweep_mode={t['sweep_mode']}  exit_reason={t['exit_reason']}")
    print("=" * 100)

    # ------------------------------------------------------------------ 1. signal chain
    print("1) SIGNAL CHAIN (from signals.parquet)")
    sig = signals.sort("timestamp")
    confirm_row = sig.filter(pl.col("timestamp") == t["signal_time"])
    if confirm_row.is_empty():
        print("  [FAIL] confirmation bar not found in signals.parquet")
        return False
    c = confirm_row.row(0, named=True)
    flag = f"confirm_{direction}"
    print(f"   confirmation bar {t['signal_time']:%Y-%m-%d %H:%M} UTC  close={c['close']:,.2f}  atr_14={c[cfg.ATR_COLUMN]}  {flag}={c[flag]}")
    audit.check("confirm flag set", c[flag], True)

    sweep_flag = f"sweep_{direction}"
    candidates = sig.filter((pl.col("timestamp") <= t["signal_time"]) & pl.col(sweep_flag)).tail(1)
    if candidates.is_empty():
        print("  [FAIL] no sweep bar found before the confirmation")
        return False
    s = candidates.row(0, named=True)
    span = sig.filter((pl.col("timestamp") >= s["timestamp"]) & (pl.col("timestamp") <= t["signal_time"]))
    extreme = span["high"].max() if direction == cfg.DIR_SHORT else span["low"].min()
    level_name = "asian_high" if direction == cfg.DIR_SHORT else "asian_low"
    print(f"   sweep bar        {s['timestamp']:%Y-%m-%d %H:%M} UTC  H={s['high']:,.2f} L={s['low']:,.2f} C={s['close']:,.2f}  {level_name}={s[level_name]:,.2f}  tag={s['session_tag']}")
    print(f"   sweep extreme = {'max high' if direction == cfg.DIR_SHORT else 'min low'} over {len(span)} 5m bar(s) sweep..confirmation")
    audit.check("sweep_price", t["sweep_price"], extreme)
    audit.check("session_tag (sweep bar)", t["session_tag"], s["session_tag"])
    audit.check("signal_time + 5m = entry", t["entry_time"], t["signal_time"] + bar)

    # ------------------------------------------------------------------ 2. entry
    print("\n2) ENTRY (open of the bar AFTER the confirmation bar - never the confirmation close)")
    entry_bar = sig.filter(pl.col("timestamp") == t["entry_time"])
    window = candles.filter((pl.col("timestamp") >= t["entry_time"]) & (pl.col("timestamp") <= t["entry_time"] + timedelta(days=1)))
    first = window.head(1).row(0, named=True)
    print(f"   confirmation close (NOT used) = {c['close']:,.2f}")
    print(fmt_candle(first["timestamp"], first["open"], first["high"], first["low"], first["close"], "<== ENTRY candle (first 1m of entry bar)"))
    audit.check("1m open at entry_time", t["entry_price"], first["open"])
    if not entry_bar.is_empty():
        audit.check("5m open at entry_time", t["entry_price"], entry_bar["open"][0])
    audit.check("entry candle timestamp", first["timestamp"], t["entry_time"])

    # ------------------------------------------------------------------ 3. stop / tp
    print("\n3) STOP / TAKE-PROFIT")
    entry = first["open"]
    atr = c[cfg.ATR_COLUMN]
    pct_term, atr_term = cfg.STOP_BUFFER_PCT * entry, cfg.STOP_BUFFER_ATR_MULT * atr
    buffer = max(pct_term, atr_term)
    stop = extreme + buffer if direction == cfg.DIR_SHORT else extreme - buffer
    dist = abs(entry - stop)
    tp = entry + sign * cfg.R_MULTIPLE * dist
    print(f"   buffer = max(0.1% x entry = {pct_term:.4f}, 0.25 x ATR14 = {atr_term:.4f}) = {buffer:.4f}")
    print(f"   stop = sweep extreme {extreme:,.2f} {'+' if direction == cfg.DIR_SHORT else '-'} buffer = {stop:,.4f}   |entry - stop| = {dist:,.4f}")
    print(f"   tp   = entry {'+' if sign > 0 else '-'} {cfg.R_MULTIPLE:g} x distance = {tp:,.4f}")
    audit.check("stop_price", t["stop_price"], stop)
    audit.check("tp_price", t["tp_price"], tp)
    audit.check("stop_distance", t["stop_distance"], dist)

    # ------------------------------------------------------------------ 4. sizing
    print("\n4) SIZING (risk-based; margin only caps)")
    prior = trades.filter(pl.col("trade_id") < trade_id)["pnl_net"].sum()
    equity = cfg.STARTING_CAPITAL + prior
    size_risk = cfg.RISK_USD / dist
    size_cap = equity * cfg.MAX_LEVERAGE / entry
    size = min(size_risk, size_cap)
    print(f"   equity before trade = {cfg.STARTING_CAPITAL:,.2f} + realised {prior:+,.2f} = {equity:,.2f}")
    print(f"   size by risk  = {cfg.RISK_USD:g} / {dist:.4f} = {size_risk:.6f}     margin cap = {equity:,.2f} x {cfg.MAX_LEVERAGE} / {entry:,.2f} = {size_cap:.6f}")
    audit.check("size", t["size"], size)
    audit.check("margin_capped", t["margin_capped"], size_cap < size_risk)
    audit.check("risk_usd (at stop)", t["risk_usd"], size * dist, tol=1e-9)

    # ------------------------------------------------------------------ 5. independent 1m walk
    print("\n5) 1m WALK from the entry candle (independent re-simulation)")
    walk = window.sort("timestamp").iter_rows(named=True)
    seen, exit_info, expected_ts = [], None, t["entry_time"]
    first_sl = first_tp = None
    for row in walk:
        ts = row["timestamp"]
        if ts != expected_ts:
            print(f"  [FAIL] 1m candle missing at {expected_ts} while position open")
            audit.failures.append("1m gap")
            break
        sl_touch = (row["low"] <= stop) if sign > 0 else (row["high"] >= stop)
        tp_touch = (row["high"] >= tp) if sign > 0 else (row["low"] <= tp)
        first_sl = first_sl or (ts if sl_touch else None)
        first_tp = first_tp or (ts if tp_touch else None)
        note = ("SL TOUCHED " if sl_touch else "") + ("TP TOUCHED" if tp_touch else "")
        seen.append((ts, row["open"], row["high"], row["low"], row["close"], note.strip()))
        end = ts + minute
        if sl_touch or tp_touch:
            reason = cfg.EXIT_BOTH_SL_FIRST if (sl_touch and tp_touch) else (cfg.EXIT_SL if sl_touch else cfg.EXIT_TP)
            price = stop if sl_touch else tp
            exit_info = (ts, reason, price, sl_touch and tp_touch)
            break
        if end.hour == cfg.FORCE_EXIT_HOUR and end.minute == 0:
            exit_info = (ts, cfg.EXIT_SESSION_END, row["close"], False)
            break
        expected_ts = ts + minute

    if exit_info is None:
        print("  [FAIL] independent walk found no exit")
        return False
    exit_ts, reason, exit_price, both = exit_info
    after = candles.filter(pl.col("timestamp") > exit_ts).head(1).row(0, named=True)

    n = len(seen)
    shown = set(range(min(context, n))) | set(range(max(0, n - context - 1), n))
    prev = -1
    for k in sorted(shown):
        if k != prev + 1:
            print(f"    ... {k - prev - 1} candle(s) with neither stop nor tp touched ...")
        ts, o, h, l, cl, note = seen[k]
        tag = "<== ENTRY candle" if k == 0 else ""
        if k == n - 1:
            tag = f"<== EXIT TRIGGER ({reason})"
        print(fmt_candle(ts, o, h, l, cl, (note + " " + tag).strip()))
        prev = k
    print(fmt_candle(after["timestamp"], after["open"], after["high"], after["low"], after["close"], "(next candle, for context)"))
    print(f"   stop level {stop:,.4f}: first touched by {first_sl:%H:%M} candle" if first_sl else f"   stop level {stop:,.4f}: never touched before exit")
    print(f"   tp   level {tp:,.4f}: first touched by {first_tp:%H:%M} candle" if first_tp else f"   tp   level {tp:,.4f}: never touched before exit")
    audit.check("exit_time", t["exit_time"], exit_ts)
    audit.check("exit_reason", t["exit_reason"], reason)
    audit.check("exit_price", t["exit_price"], exit_price)
    audit.check("sl_tp_both_hit", t["sl_tp_both_hit"], both)

    # ------------------------------------------------------------------ 6. costs, funding, pnl
    print("\n6) COSTS AND PNL")
    slip = cfg.SLIPPAGE_BPS / cfg.BPS_PER_UNIT
    entry_fill = entry * (1 + sign * slip)
    exit_fill = exit_price * (1 - sign * slip)
    gross = sign * size * (exit_price - entry)
    slip_cost = size * (abs(entry_fill - entry) + abs(exit_fill - exit_price))
    fees = cfg.TAKER_FEE_RATE * size * (entry_fill + exit_fill)
    print(f"   fills: entry {entry:,.4f} -> {entry_fill:,.4f} | exit {exit_price:,.4f} -> {exit_fill:,.4f}   (slippage {cfg.SLIPPAGE_BPS} bps against you)")

    boundaries = []
    day = t["entry_time"].replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= exit_ts:
        for h in cfg.FUNDING_HOURS:
            b = day + timedelta(hours=h)
            if t["entry_time"] < b <= exit_ts:
                boundaries.append(b)
        day += timedelta(days=1)
    funding_paid, estimated = 0.0, funding is None
    for b in boundaries:
        px = candles.filter(pl.col("timestamp") == b)["open"][0]
        if funding is not None and b in funding:
            rate = funding[b]
        else:
            rate, estimated = cfg.FUNDING_ESTIMATE_RATE, True
        charge = sign * size * px * rate
        funding_paid += charge
        print(f"   funding boundary {b:%Y-%m-%d %H:%M}: 1m open {px:,.2f} x size {size:.6f} x rate {rate:.6f} x {'long pays' if sign > 0 else 'short receives'} = {charge:+.6f}")
    if not boundaries:
        print("   funding: no 00/08/16 UTC boundary between entry and exit")

    pnl_net = gross - slip_cost - fees - funding_paid
    print(f"   pnl_gross {gross:+.6f} - slippage {slip_cost:.6f} - fees {fees:.6f} - funding {funding_paid:+.6f} = pnl_net {pnl_net:+.6f}   ({pnl_net / (size * dist):+.4f} R)")
    audit.check("entry_fill_price", t["entry_fill_price"], entry_fill)
    audit.check("exit_fill_price", t["exit_fill_price"], exit_fill)
    audit.check("pnl_gross", t["pnl_gross"], gross, tol=1e-6)
    audit.check("slippage_cost", t["slippage_cost"], slip_cost, tol=1e-6)
    audit.check("fees", t["fees"], fees, tol=1e-6)
    audit.check("n_funding_boundaries", t["n_funding_boundaries"], len(boundaries))
    audit.check("funding_paid", t["funding_paid"], funding_paid, tol=1e-6)
    audit.check("funding_estimated", t["funding_estimated"], estimated)
    audit.check("pnl_net", t["pnl_net"], pnl_net, tol=1e-6)
    audit.check("r_multiple", t["r_multiple"], pnl_net / (size * dist), tol=1e-9)

    print(LINE)
    if audit.failures:
        print(f"RESULT: FAIL ({len(audit.failures)} mismatch(es): {', '.join(audit.failures)})")
        return False
    print("RESULT: PASS - every logged number was reproduced independently from raw data")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Independently verify logged trades against raw 1m data.")
    ap.add_argument("trade_id", nargs="?", type=int)
    ap.add_argument("--random", type=int, metavar="N", help="verify N randomly chosen trades")
    ap.add_argument("--seed", type=int, default=cfg.PLOT_SEED)
    ap.add_argument("--context", type=int, default=3, help="1m candles shown around entry and exit")
    args = ap.parse_args()
    if args.trade_id is None and not args.random:
        ap.error("give a trade_id or --random N")

    trades, signals, candles, funding = load_inputs()
    ids = [args.trade_id] if args.trade_id is not None else sorted(
        random.Random(args.seed).sample(trades["trade_id"].to_list(), min(args.random, trades.height)))
    results = [verify(i, trades, signals, candles, funding, args.context) for i in ids]
    print(f"\n{sum(results)}/{len(results)} trade(s) verified: {ids}")
    return cfg.EXIT_OK if all(results) else cfg.EXIT_ABORTED


if __name__ == "__main__":
    sys.exit(main())
