"""
app.py - Phase 4 Streamlit dashboard.   Launch:  streamlit run app.py

Reads data/trades.parquet ONLY. Every number comes from analytics.py, which
uses the trade log alone (no market data, no re-simulation). KPIs and charts
always use the FULL, unfiltered log so they reconcile with scripts/reconcile.py;
the filters in the trade-log section affect that table and its CSV download only.
"""
from __future__ import annotations

import math

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

import analytics as an
import config as cfg
from utils.logging_setup import get_logger, setup_logging

setup_logging(log_to_file=False)
logger = get_logger("app")


# ==========================================================================
# helpers
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_trades(path: str, mtime: float) -> pl.DataFrame:      # mtime is part of the cache key
    logger.info("loading trades from %s", path)
    return pl.read_parquet(path)


def money(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def fmt(x: float | None, spec: str = ".2f", suffix: str = "") -> str:
    if x is None:
        return "n/a"
    if math.isinf(x):
        return "inf"
    return f"{x:{spec}}{suffix}"


def colors(values: list) -> list[str]:
    return [cfg.DASH_COLOR_NEUTRAL if v is None else (cfg.DASH_COLOR_POS if v >= 0 else cfg.DASH_COLOR_NEG) for v in values]


# ==========================================================================
# figures
# ==========================================================================
def equity_figure(curve: pl.DataFrame, start: float) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=("Equity (starting capital + cumulative pnl_net)", "Drawdown (% of peak equity)"))
    x = curve["time"].dt.replace_time_zone(None).to_list()       # naive UTC wall-clock for plotly
    fig.add_trace(go.Scatter(x=x, y=curve["equity"].to_list(), mode="lines", name="Equity",
                             line=dict(color=cfg.DASH_COLOR_NEUTRAL, width=2),
                             hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>equity $%{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=start, line_dash="dot", line_color="gray", annotation_text=f"start ${start:,.0f}", row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=curve["drawdown_pct"].to_list(), mode="lines", name="Drawdown",
                             fill="tozeroy", fillcolor=cfg.DASH_COLOR_DRAWDOWN, line=dict(color=cfg.DASH_COLOR_NEG, width=1),
                             hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>drawdown %{y:.2f}%<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="% of peak", row=2, col=1)
    fig.update_xaxes(title_text="exit time (UTC)", row=2, col=1)
    fig.update_layout(height=cfg.DASH_CHART_HEIGHT_PX, showlegend=False, margin=dict(t=50, b=40))
    return fig


def breakdown_figure(table: pl.DataFrame, by: str) -> go.Figure:
    """Three panels (the metrics have different units): Trades | Win Rate | Expectancy, one bar per group."""
    panels = [("trades", "Trades", "{:d}"), ("win_rate_pct", "Win rate (%)", "{:.1f}%"), ("expectancy_r", "Expectancy (R)", "{:+.2f}")]
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[1] for p in panels])
    groups = [str(g) for g in table[by].to_list()]
    for i, (col, _, label) in enumerate(panels, start=1):
        values = table[col].to_list()
        fig.add_trace(go.Bar(
            x=groups, y=values, name=col,
            marker_color=cfg.DASH_COLOR_NEUTRAL if col != "expectancy_r" else colors(values),
            text=["n/a" if v is None else label.format(v) for v in values], textposition="outside",
            customdata=table["trades"].to_list(), hovertemplate="%{x}<br>%{y}<br>n = %{customdata} trades<extra></extra>",
        ), row=1, col=i)
    fig.update_layout(height=cfg.DASH_BREAKDOWN_HEIGHT_PX, showlegend=False, margin=dict(t=50, b=30))
    return fig


def r_histogram(trades: pl.DataFrame, expectancy: float | None) -> go.Figure:
    r = trades["r_multiple"].to_list()
    fig = go.Figure(go.Histogram(x=r, xbins=dict(size=cfg.DASH_HIST_BIN_SIZE_R), marker_color=cfg.DASH_COLOR_NEUTRAL,
                                 hovertemplate="R in %{x}<br>%{y} trades<extra></extra>"))
    fig.add_vline(x=0, line_color="gray", line_dash="dot")
    if expectancy is not None:
        fig.add_vline(x=expectancy, line_color=cfg.DASH_COLOR_NEG, annotation_text=f"mean {expectancy:+.2f}R")
    fig.update_layout(height=cfg.DASH_BREAKDOWN_HEIGHT_PX, xaxis_title="r_multiple (net of costs)",
                      yaxis_title="trades", bargap=0.05, margin=dict(t=30, b=30))
    return fig


def table_view(table: pl.DataFrame) -> pl.DataFrame:
    return table.with_columns(
        (pl.col("trades") < cfg.DASH_LOW_SAMPLE_TRADES).alias(f"low sample (<{cfg.DASH_LOW_SAMPLE_TRADES})"))


# ==========================================================================
# page
# ==========================================================================
st.set_page_config(page_title="Sweep backtest", layout="wide")
st.title(cfg.APP_TITLE)

path = cfg.TRADES_PARQUET_PATH
if not path.exists():
    st.error(f"`{path}` not found. Run `python backtester.py` first (see README.md).")
    st.stop()

trades = load_trades(str(path), path.stat().st_mtime)
bad = an.missing_columns(trades)
if bad:
    st.error(f"`{path.name}` is missing required columns: {bad}. Re-run `python backtester.py`.")
    st.stop()

k = an.compute_kpis(trades, cfg.STARTING_CAPITAL)
diag = an.diagnostics(trades)

st.caption(
    f"Source: `{path.name}` - {k.n_trades} trades"
    + (f", {trades['entry_time'].min():%Y-%m-%d} to {trades['exit_time'].max():%Y-%m-%d} UTC" if k.n_trades else "")
    + (f" - sweep_mode: {', '.join(diag['sweep_modes'])}" if diag.get("sweep_modes") else "")
    + f" - starting capital ${cfg.STARTING_CAPITAL:,.0f}, fixed ${cfg.RISK_USD:,.0f} risk per trade (no compounding)."
)
if k.n_trades == 0:
    st.warning("The trade log is empty - nothing to analyse. Check the backtester log for rejected orders.")
elif k.n_trades < cfg.DASH_LOW_SAMPLE_TRADES:
    st.warning(f"Only {k.n_trades} trades (< {cfg.DASH_LOW_SAMPLE_TRADES}): treat every statistic below as noise.")
if diag["funding_estimated"] and diag["funding_estimated"] == diag["n_trades"]:
    st.warning("Funding was ESTIMATED (0.01% per 8h) for every trade - no historical funding file was used.")

# ---------------------------------------------------------------- KPI header
st.header("Key metrics")
st.caption("All values are computed from the trade log alone, on the full unfiltered log. "
           "Wins = pnl_net > 0 (after fees, slippage, funding).")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Win Rate", fmt(k.win_rate_pct, ".2f", "%"), help="winning trades / total trades")
c2.metric("Profit Factor", fmt(k.profit_factor, ".2f"), help="gross profit / gross loss (inf = no losing trades)")
c3.metric("Total Trades", f"{k.n_trades:,}")
c4.metric("Net PnL", money(k.net_pnl), delta=f"{k.net_pnl_pct:+.2f}%" if k.n_trades else None,
          help=f"sum of pnl_net; % is of the ${cfg.STARTING_CAPITAL:,.0f} starting capital")
c5, c6, c7, c8 = st.columns(4)
c5.metric("Expectancy (R)", fmt(k.expectancy_r, "+.3f", " R"), help="mean r_multiple over all trades")
c6.metric("Average Win R", fmt(k.avg_win_r, "+.3f", " R"))
c7.metric("Average Loss R", fmt(k.avg_loss_r, "+.3f", " R"), help="negative; includes fees and slippage, so it can be worse than -1R")
c8.metric("Max Drawdown", f"{money(-k.max_dd_usd) if k.max_dd_usd else '$0.00'} / {k.max_dd_pct:.2f}%",
          help="largest peak-to-trough fall in $, and (independently) in % of the peak; they can occur at different times")

with st.expander("Exact values for scripts/reconcile.py"):
    st.code(an.format_reconciliation(k), language="text")

if k.n_trades:
    # ------------------------------------------------------------ equity
    st.header("Equity curve")
    curve = an.equity_curve(trades, cfg.STARTING_CAPITAL)
    st.plotly_chart(equity_figure(curve, cfg.STARTING_CAPITAL), width="stretch")

    # ------------------------------------------------------------ breakdowns
    st.header("Session breakdown")
    st.caption("Grouped by the session tag of the sweep bar. `London_NY_overlap` is its own bucket and is always shown.")
    by_session = an.breakdown(trades, "session_tag", cfg.DASH_SESSION_GROUPS)
    st.plotly_chart(breakdown_figure(by_session, "session_tag"), width="stretch")
    st.dataframe(table_view(by_session), hide_index=True, width="stretch")

    st.header("Long vs short")
    by_dir = an.breakdown(trades, "direction", cfg.DASH_DIRECTION_GROUPS)
    st.plotly_chart(breakdown_figure(by_dir, "direction"), width="stretch")
    st.dataframe(table_view(by_dir), hide_index=True, width="stretch")

    st.header("Per calendar year")
    st.caption("Year of exit_time (UTC), i.e. when the pnl was realised. Years with no trades are shown as zero.")
    by_year = an.yearly(trades)
    st.plotly_chart(breakdown_figure(by_year, "year"), width="stretch")
    st.dataframe(table_view(by_year), hide_index=True, width="stretch")

    # ------------------------------------------------------------ R histogram
    st.header("R-multiple distribution")
    st.plotly_chart(r_histogram(trades, k.expectancy_r), width="stretch")

    # ------------------------------------------------------------ trade log
    st.header("Trade log")
    st.caption("Filters apply to this table and its download only - the metrics above always use the full log.")
    f1, f2, f3 = st.columns(3)
    sel_dir = f1.multiselect("direction", sorted(trades["direction"].unique().to_list()), default=sorted(trades["direction"].unique().to_list()))
    sel_tag = f2.multiselect("session_tag", sorted(trades["session_tag"].unique().to_list()), default=sorted(trades["session_tag"].unique().to_list()))
    sel_exit = f3.multiselect("exit_reason", sorted(trades["exit_reason"].unique().to_list()), default=sorted(trades["exit_reason"].unique().to_list()))
    shown = trades.filter(pl.col("direction").is_in(sel_dir) & pl.col("session_tag").is_in(sel_tag) & pl.col("exit_reason").is_in(sel_exit))
    st.caption(f"Showing {shown.height:,} of {trades.height:,} trades.")
    st.dataframe(shown, hide_index=True, width="stretch")
    st.download_button("Download shown trades as CSV", data=shown.write_csv(), file_name="trades_filtered.csv", mime="text/csv")

# ---------------------------------------------------------------- diagnostics
st.header("Diagnostics")
st.caption("Honesty checks: how often the simulation had to make a pessimistic call, estimate, or alter risk.")
d1, d2, d3, d4 = st.columns(4)
d1.metric("SL and TP in same 1m candle", fmt(diag["both_hit"], "d"), help="stop assumed first (pessimistic); exit_reason = both_sl_first")
d2.metric("Funding estimated", f"{fmt(diag['funding_estimated'], 'd')} of {diag['n_trades']}",
          help="no historical rate for at least one boundary, so 0.01% per 8h was used")
d3.metric("Margin cap reduced size", fmt(diag["margin_capped"], "d"),
          help="size cut below the risk-based size, so risk at the stop was less than $100")
d4.metric("Exit reasons", ", ".join(f"{r}: {n}" for r, n in diag["exit_reasons"].items()) or "n/a")
for name in ("both_hit", "funding_estimated", "margin_capped"):
    if diag[name] is None:
        st.info(f"Column for `{name}` is not in the trade log - cannot be determined (not assumed to be zero).")
if diag["margin_capped"]:
    st.dataframe(trades.filter(pl.col("margin_capped")).select("trade_id", "entry_time", "direction", "size_uncapped", "size", "risk_usd"),
                 hide_index=True, width="stretch")
if "avg_cost_r" in diag:
    st.subheader("Cost drag")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Gross PnL", money(diag["total_gross"]))
    g2.metric("Fees", money(-diag["total_fees"]))
    g3.metric("Slippage", money(-diag["total_slippage"]))
    g4.metric("Funding paid", money(-diag["total_funding"]), help="negative cost = funding received")
    g5.metric("Avg cost per trade", f"{diag['avg_cost_r']:.2f} R", help="(fees + slippage + funding) / risk_usd")

# ---------------------------------------------------------------- placeholders
st.header("Coming in Phase 5")
st.caption("Placeholders only - not yet wired, no values are being fabricated.")
for col, label in zip(st.columns(len(cfg.DASH_PLACEHOLDERS)), cfg.DASH_PLACEHOLDERS):
    with col:
        st.metric(label, cfg.DASH_NOT_COMPUTED)
        st.caption("coming in phase 5, not yet wired")
