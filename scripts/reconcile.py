"""
reconcile.py - recompute the dashboard's headline numbers from data/trades.csv.

Deliberately independent of the dashboard code: standard library only (csv, math),
no Polars, no analytics.py. It also reads a different file (the CSV) from the one
the dashboard reads (the parquet) and cross-checks the two.

Usage (project root):
    python scripts/reconcile.py
    python scripts/reconcile.py --csv path/to/trades.csv

Compare its output line by line with the dashboard's
"Exact values for scripts/reconcile.py" box (Key metrics section). The first four
lines (Win Rate, Profit Factor, Net PnL, Max Drawdown) are the required ones; the
rest cover the remaining KPI tiles. Exit code 1 if the CSV is missing/unreadable
or disagrees with trades.parquet.

Definitions: see the docstring of analytics.py (they are the same by design).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg

DEC = cfg.DASH_RECON_DECIMALS


def fmt(x):
    if x is None:
        return "n/a"
    if math.isinf(x):
        return "inf"
    return f"{x:.{DEC}f}"


def read_trades(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    need = {"trade_id", "exit_time", "pnl_net", "r_multiple"}
    if rows and not need <= set(rows[0]):
        sys.exit(f"{path.name} lacks columns {sorted(need - set(rows[0]))}")
    for r in rows:
        r["trade_id"], r["pnl_net"], r["r_multiple"] = int(r["trade_id"]), float(r["pnl_net"]), float(r["r_multiple"])
    # realisation order: exit_time (ISO strings sort chronologically), then trade_id
    rows.sort(key=lambda r: (r["exit_time"], r["trade_id"]))
    return rows


def recompute(rows: list[dict], start: float) -> dict:
    pnl = [r["pnl_net"] for r in rows]
    n = len(pnl)
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]
    gross_profit, gross_loss = math.fsum(wins), -math.fsum(losses)

    if n == 0:
        pf = None
    elif gross_loss == 0:
        pf = math.inf if gross_profit > 0 else None
    else:
        pf = gross_profit / gross_loss

    equity = peak = start
    max_dd_usd = max_dd_pct = 0.0
    for p in pnl:
        equity += p
        peak = max(peak, equity)
        max_dd_usd = max(max_dd_usd, peak - equity)
        max_dd_pct = max(max_dd_pct, (peak - equity) / peak * 100)

    def mean(xs):
        return math.fsum(xs) / len(xs) if xs else None

    net = math.fsum(pnl)
    return {
        "n": n, "wins": len(wins), "win_rate": (len(wins) / n * 100) if n else None, "pf": pf,
        "net": net, "net_pct": net / start * 100, "dd_usd": max_dd_usd, "dd_pct": max_dd_pct,
        "exp_r": mean([r["r_multiple"] for r in rows]),
        "win_r": mean([r["r_multiple"] for r in rows if r["pnl_net"] > 0]),
        "loss_r": mean([r["r_multiple"] for r in rows if r["pnl_net"] < 0]),
    }


def render(m: dict, start: float) -> str:
    return "\n".join([
        f"Win Rate        : {fmt(m['win_rate'])} %  ({m['wins']} wins / {m['n']} trades)",
        f"Profit Factor   : {fmt(m['pf'])}",
        f"Net PnL         : {fmt(m['net'])} USD  ({fmt(m['net_pct'])} % of {start:.2f} starting capital)",
        f"Max Drawdown    : {fmt(m['dd_usd'])} USD  /  {fmt(m['dd_pct'])} % of peak equity",
        f"Total Trades    : {m['n']}",
        f"Expectancy (R)  : {fmt(m['exp_r'])}",
        f"Average Win R   : {fmt(m['win_r'])}",
        f"Average Loss R  : {fmt(m['loss_r'])}",
    ])


def cross_check_parquet(rows: list[dict]) -> bool:
    """The dashboard reads the parquet; make sure the CSV describes the same trades."""
    if not cfg.TRADES_PARQUET_PATH.exists():
        print(f"\n(cross-check skipped: {cfg.TRADES_PARQUET_PATH.name} not found)")
        return True
    import polars as pl                                    # only for this check
    pq = pl.read_parquet(cfg.TRADES_PARQUET_PATH)
    pq_pnl = math.fsum(pq["pnl_net"].to_list())
    csv_pnl = math.fsum(r["pnl_net"] for r in rows)
    ids_match = sorted(pq["trade_id"].to_list()) == sorted(r["trade_id"] for r in rows)
    ok = pq.height == len(rows) and ids_match and math.isclose(pq_pnl, csv_pnl, rel_tol=1e-12, abs_tol=1e-9)
    print(f"\nCSV vs parquet : {'MATCH' if ok else 'MISMATCH'}  (csv {len(rows)} trades / pnl {csv_pnl:.6f}; "
          f"parquet {pq.height} trades / pnl {pq_pnl:.6f})")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute dashboard KPIs from trades.csv.")
    ap.add_argument("--csv", type=Path, default=cfg.TRADES_CSV_PATH)
    args = ap.parse_args(argv)
    if not args.csv.exists():
        print(f"{args.csv} not found - run `python backtester.py` first")
        return cfg.EXIT_ABORTED

    rows = read_trades(args.csv)
    print(f"Reconciliation from {args.csv.name}  ({len(rows)} rows, starting capital {cfg.STARTING_CAPITAL:.2f})\n")
    print(render(recompute(rows, cfg.STARTING_CAPITAL), cfg.STARTING_CAPITAL))
    return cfg.EXIT_OK if cross_check_parquet(rows) else cfg.EXIT_ABORTED


if __name__ == "__main__":
    sys.exit(main())
