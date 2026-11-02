"""
analytics.py - every dashboard number, computed from the trade log ALONE.

Pure functions, no Streamlit, no market data: input is the trades table
(data/trades.parquet), nothing else. app.py only draws what this returns, and
scripts/reconcile.py recomputes the headline numbers independently from
trades.csv so the two can be compared line by line.

DEFINITIONS (identical in scripts/reconcile.py)
  order of trades   : by (exit_time, trade_id) - the order in which pnl is realised
  win / loss        : pnl_net > 0 / pnl_net < 0 (a trade with pnl_net == 0 is neither, but still counts as a trade)
  win rate          : wins / total trades
  profit factor     : sum(positive pnl_net) / |sum(negative pnl_net)|   (inf if no losses, None if no trades)
  net pnl           : sum(pnl_net);  net pnl %  = net pnl / starting capital   (fixed capital, no compounding)
  expectancy (R)    : mean(r_multiple) over ALL trades
  average win R/loss R : mean(r_multiple) over winning / losing trades (loss R is negative)
  equity            : starting capital + cumulative pnl_net, one point per trade at its exit_time
  drawdown          : (running peak - equity), peak starts at starting capital
  max drawdown      : the largest drawdown in $ and, independently, the largest as % of the peak
                      (the two can occur at different times)
Sums use math.fsum so results do not depend on summation order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

import config as cfg

REQUIRED_COLUMNS = ["trade_id", "entry_time", "exit_time", "direction", "session_tag",
                    "pnl_net", "r_multiple", "exit_reason"]


@dataclass(frozen=True)
class Kpis:
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate_pct: float | None
    profit_factor: float | None          # math.inf when there are no losing trades
    gross_profit: float
    gross_loss: float                    # positive number
    net_pnl: float
    net_pnl_pct: float
    expectancy_r: float | None
    avg_win_r: float | None
    avg_loss_r: float | None
    max_dd_usd: float
    max_dd_pct: float
    starting_capital: float


def missing_columns(trades: pl.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in trades.columns]


def ordered(trades: pl.DataFrame) -> pl.DataFrame:
    return trades.sort(["exit_time", "trade_id"])


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def compute_kpis(trades: pl.DataFrame, starting_capital: float = cfg.STARTING_CAPITAL) -> Kpis:
    rows = ordered(trades)
    pnl = rows["pnl_net"].to_list()
    r = rows["r_multiple"].to_list()
    n = len(pnl)

    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]
    gross_profit, gross_loss = math.fsum(wins), -math.fsum(losses)
    if not n:
        profit_factor = None
    elif gross_loss == 0:
        profit_factor = math.inf if gross_profit > 0 else None
    else:
        profit_factor = gross_profit / gross_loss
    net = math.fsum(pnl)

    equity, peak, dd_usd, dd_pct = starting_capital, starting_capital, 0.0, 0.0
    for p in pnl:
        equity += p
        peak = max(peak, equity)
        dd_usd = max(dd_usd, peak - equity)
        dd_pct = max(dd_pct, (peak - equity) / peak * 100)

    return Kpis(
        n_trades=n, n_wins=len(wins), n_losses=len(losses),
        win_rate_pct=(len(wins) / n * 100) if n else None,
        profit_factor=profit_factor, gross_profit=gross_profit, gross_loss=gross_loss,
        net_pnl=net, net_pnl_pct=net / starting_capital * 100,
        expectancy_r=_mean(r),
        avg_win_r=_mean([rr for p, rr in zip(pnl, r) if p > 0]),
        avg_loss_r=_mean([rr for p, rr in zip(pnl, r) if p < 0]),
        max_dd_usd=dd_usd, max_dd_pct=dd_pct, starting_capital=starting_capital,
    )


def equity_curve(trades: pl.DataFrame, starting_capital: float = cfg.STARTING_CAPITAL) -> pl.DataFrame:
    """
    Columns: time, equity, peak, drawdown_usd, drawdown_pct (<= 0, % of peak).
    First row is the starting point (starting capital at the first trade's entry_time).
    """
    rows = ordered(trades)
    if rows.is_empty():
        return pl.DataFrame(schema={"time": pl.Datetime(cfg.TIME_UNIT, cfg.TIMEZONE), "equity": pl.Float64,
                                    "peak": pl.Float64, "drawdown_usd": pl.Float64, "drawdown_pct": pl.Float64})
    times = [rows["entry_time"][0]] + rows["exit_time"].to_list()
    equity, level = [], starting_capital
    equity.append(level)
    for p in rows["pnl_net"].to_list():
        level += p
        equity.append(level)
    curve = pl.DataFrame({"time": times, "equity": equity}, schema_overrides={"time": rows.schema["exit_time"]})
    return curve.with_columns(pl.col("equity").cum_max().alias("peak")).with_columns(
        (pl.col("peak") - pl.col("equity")).alias("drawdown_usd"),
        (-(pl.col("peak") - pl.col("equity")) / pl.col("peak") * 100).alias("drawdown_pct"),
    )


def breakdown(trades: pl.DataFrame, by: str, categories: list | None = None) -> pl.DataFrame:
    """
    One row per group: trades, wins, win_rate_pct, expectancy_r, net_pnl.
    Every category in `categories` is present even with zero trades (rates are null, not 0).
    """
    agg = trades.group_by(by).agg(
        pl.len().alias("trades"),
        (pl.col("pnl_net") > 0).sum().alias("wins"),
        pl.col("r_multiple").mean().alias("expectancy_r"),
        pl.col("pnl_net").sum().alias("net_pnl"),
    )
    present = agg[by].to_list()
    keys = list(categories or []) + sorted(k for k in present if k not in (categories or []))
    base = pl.DataFrame({by: keys}, schema={by: agg.schema[by]}) if keys else agg.select(by).clear()
    out = base.join(agg, on=by, how="left").with_columns(
        pl.col("trades").fill_null(0).cast(pl.Int64),
        pl.col("wins").fill_null(0).cast(pl.Int64),
        pl.col("net_pnl").fill_null(0.0),
    )
    out = out.with_columns(
        pl.when(pl.col("trades") > 0).then(pl.col("wins") / pl.col("trades") * 100).alias("win_rate_pct")
    )
    # keep the requested order, then any extras
    order = {k: i for i, k in enumerate(keys)}
    return out.sort(pl.col(by).replace_strict(order, default=len(order))).select(
        by, "trades", "wins", "win_rate_pct", "expectancy_r", "net_pnl")


def yearly(trades: pl.DataFrame) -> pl.DataFrame:
    """Per calendar year of exit_time (UTC), including quiet years between the first and last."""
    if trades.is_empty():
        return pl.DataFrame(schema={"year": pl.Int32, "trades": pl.Int64, "wins": pl.Int64,
                                    "win_rate_pct": pl.Float64, "expectancy_r": pl.Float64, "net_pnl": pl.Float64})
    t = trades.with_columns(pl.col("exit_time").dt.year().alias("year"))
    years = list(range(t["year"].min(), t["year"].max() + 1))
    return breakdown(t, "year", years)


def diagnostics(trades: pl.DataFrame) -> dict:
    """Honesty counters. A value of None means the column is not in the trade log (never faked)."""
    def count_true(col: str) -> int | None:
        return int(trades[col].sum()) if col in trades.columns else None

    out = {
        "n_trades": trades.height,
        "both_hit": count_true("sl_tp_both_hit"),
        "funding_estimated": count_true("funding_estimated"),
        "margin_capped": count_true("margin_capped"),
        "exit_reasons": dict(trades.group_by("exit_reason").len().sort("exit_reason").iter_rows()),
        "sweep_modes": sorted(trades["sweep_mode"].unique().to_list()) if "sweep_mode" in trades.columns else None,
    }
    cost_cols = ("fees", "slippage_cost", "funding_paid", "pnl_gross", "risk_usd")
    if trades.height and all(c in trades.columns for c in cost_cols):
        out["total_fees"] = trades["fees"].sum()
        out["total_slippage"] = trades["slippage_cost"].sum()
        out["total_funding"] = trades["funding_paid"].sum()
        out["total_gross"] = trades["pnl_gross"].sum()
        out["avg_cost_r"] = ((trades["fees"] + trades["slippage_cost"] + trades["funding_paid"]) / trades["risk_usd"]).mean()
    return out


# --------------------------------------------------------------------------
# Reconciliation block - the SAME text is printed by scripts/reconcile.py
# --------------------------------------------------------------------------
def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    if math.isinf(x):
        return "inf"
    return f"{x:.{cfg.DASH_RECON_DECIMALS}f}"


def format_reconciliation(k: Kpis) -> str:
    return "\n".join([
        f"Win Rate        : {_fmt(k.win_rate_pct)} %  ({k.n_wins} wins / {k.n_trades} trades)",
        f"Profit Factor   : {_fmt(k.profit_factor)}",
        f"Net PnL         : {_fmt(k.net_pnl)} USD  ({_fmt(k.net_pnl_pct)} % of {k.starting_capital:.2f} starting capital)",
        f"Max Drawdown    : {_fmt(k.max_dd_usd)} USD  /  {_fmt(k.max_dd_pct)} % of peak equity",
        f"Total Trades    : {k.n_trades}",
        f"Expectancy (R)  : {_fmt(k.expectancy_r)}",
        f"Average Win R   : {_fmt(k.avg_win_r)}",
        f"Average Loss R  : {_fmt(k.avg_loss_r)}",
    ])
