"""
Phase 4 tests - dashboard maths and reconciliation (synthetic trade logs, hand-computed answers).

Reference log (6 trades, risk $100 each so r = pnl/100), in exit order:
  id  exit        dir    tag      pnl     R
  1   2024-03-01  long   London   +200   +2.0
  2   2024-03-02  short  London   -100   -1.0
  3   2024-03-03  short  NY       +150   +1.5
  4   2026-01-05  long   London    -50   -0.5
  5   2026-01-06  short  NY       -120   -1.2
  6   2026-01-07  long   London      0    0.0     (neither a win nor a loss)
equity: 10000 -> 10200, 10100, 10250, 10200, 10080, 10080
"""
import importlib.util
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

import analytics as an
import config as cfg

UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
DT = pl.Datetime(cfg.TIME_UNIT, cfg.TIMEZONE)

REF = [  # (exit date, direction, session_tag, pnl)
    (datetime(2024, 3, 1, 12, tzinfo=UTC), "long", "London", 200.0),
    (datetime(2024, 3, 2, 12, tzinfo=UTC), "short", "London", -100.0),
    (datetime(2024, 3, 3, 18, tzinfo=UTC), "short", "NY", 150.0),
    (datetime(2026, 1, 5, 12, tzinfo=UTC), "long", "London", -50.0),
    (datetime(2026, 1, 6, 18, tzinfo=UTC), "short", "NY", -120.0),
    (datetime(2026, 1, 7, 12, tzinfo=UTC), "long", "London", 0.0),
]


def make_trades(spec=REF, **extra):
    rows = []
    for i, (exit_ts, direction, tag, pnl) in enumerate(spec, start=1):
        rows.append({
            "trade_id": i, "entry_time": exit_ts - timedelta(hours=1), "exit_time": exit_ts,
            "direction": direction, "session_tag": tag, "sweep_mode": "wick_only",
            "pnl_gross": pnl + 10.0, "fees": 6.0, "slippage_cost": 3.0, "funding_paid": 1.0,
            "pnl_net": pnl, "r_multiple": pnl / 100.0, "risk_usd": 100.0,
            "sl_tp_both_hit": False, "funding_estimated": True, "exit_reason": "tp" if pnl > 0 else "sl",
            "margin_capped": False, "size_uncapped": 1.0, "size": 1.0,
        })
    df = pl.DataFrame(rows)
    return df.with_columns(pl.col("entry_time").cast(DT), pl.col("exit_time").cast(DT))


# ==========================================================================
# KPIs
# ==========================================================================
def test_kpis_match_hand_calculation():
    k = an.compute_kpis(make_trades())
    assert (k.n_trades, k.n_wins, k.n_losses) == (6, 2, 3)
    assert k.win_rate_pct == pytest.approx(2 / 6 * 100)                 # 33.333...
    assert k.gross_profit == 350.0 and k.gross_loss == 270.0
    assert k.profit_factor == pytest.approx(350 / 270)                  # 1.296296...
    assert k.net_pnl == 80.0 and k.net_pnl_pct == pytest.approx(0.8)    # 80 / 10_000
    assert k.expectancy_r == pytest.approx(0.8 / 6)                     # mean R over all 6 trades
    assert k.avg_win_r == pytest.approx(1.75)                           # (2.0 + 1.5) / 2
    assert k.avg_loss_r == pytest.approx(-0.9)                          # (-1.0 - 0.5 - 1.2) / 3
    assert k.max_dd_usd == pytest.approx(170.0)                         # peak 10250 -> trough 10080
    assert k.max_dd_pct == pytest.approx(170 / 10250 * 100)             # 1.6585...


def test_kpis_do_not_depend_on_row_order():
    shuffled = make_trades().sample(fraction=1.0, shuffle=True, seed=5)
    assert an.compute_kpis(shuffled) == an.compute_kpis(make_trades())


def test_drawdown_usd_and_pct_are_independent_maxima():
    # -500 from 10000 = 5.00% ;  then +5000 -> 14500 ;  -600 from 14500 = 4.14% but a bigger $ drop
    spec = [(datetime(2024, 1, d, tzinfo=UTC), "long", "London", p) for d, p in ((1, -500.0), (2, 5000.0), (3, -600.0))]
    k = an.compute_kpis(make_trades(spec))
    assert k.max_dd_usd == pytest.approx(600.0)
    assert k.max_dd_pct == pytest.approx(5.0)


def test_profit_factor_edge_cases():
    only_wins = [(datetime(2024, 1, d, tzinfo=UTC), "long", "London", 100.0) for d in (1, 2)]
    k = an.compute_kpis(make_trades(only_wins))
    assert math.isinf(k.profit_factor) and k.avg_loss_r is None and k.win_rate_pct == 100.0
    empty = an.compute_kpis(make_trades().clear())
    assert empty.n_trades == 0 and empty.profit_factor is None and empty.win_rate_pct is None
    assert empty.net_pnl == 0.0 and empty.max_dd_usd == 0.0


# ==========================================================================
# equity curve
# ==========================================================================
def test_equity_curve_starts_at_capital_and_ends_at_capital_plus_net():
    curve = an.equity_curve(make_trades())
    assert curve["equity"].to_list() == [10000.0, 10200.0, 10100.0, 10250.0, 10200.0, 10080.0, 10080.0]
    assert curve["peak"].to_list() == [10000.0, 10200.0, 10200.0, 10250.0, 10250.0, 10250.0, 10250.0]
    assert curve["drawdown_pct"].min() == pytest.approx(-170 / 10250 * 100)
    assert (curve["drawdown_pct"] <= 0).all()
    assert curve["equity"][-1] == cfg.STARTING_CAPITAL + an.compute_kpis(make_trades()).net_pnl
    assert curve["time"][0] == make_trades()["entry_time"][0]            # starting point sits at the first entry


# ==========================================================================
# breakdowns
# ==========================================================================
def row_of(table, by, key):
    return table.filter(pl.col(by) == key).row(0, named=True)


def test_session_breakdown_keeps_empty_overlap_bucket_with_null_rates():
    t = an.breakdown(make_trades(), "session_tag", cfg.DASH_SESSION_GROUPS)
    assert t["session_tag"].to_list() == ["London", "London_NY_overlap", "NY"]
    lon, ovl, ny = (row_of(t, "session_tag", k) for k in t["session_tag"].to_list())
    assert (lon["trades"], lon["wins"], lon["win_rate_pct"], lon["expectancy_r"], lon["net_pnl"]) == (4, 1, 25.0, pytest.approx(0.125), 50.0)
    assert (ny["trades"], ny["win_rate_pct"], ny["expectancy_r"], ny["net_pnl"]) == (2, 50.0, pytest.approx(0.15), 30.0)
    assert ovl["trades"] == 0 and ovl["win_rate_pct"] is None and ovl["expectancy_r"] is None and ovl["net_pnl"] == 0.0


def test_overlap_trades_are_counted_in_their_own_bucket():
    spec = list(REF) + [(datetime(2026, 1, 8, 12, tzinfo=UTC), "long", "London_NY_overlap", 120.0)]
    t = an.breakdown(make_trades(spec), "session_tag", cfg.DASH_SESSION_GROUPS)
    ovl = row_of(t, "session_tag", "London_NY_overlap")
    assert (ovl["trades"], ovl["win_rate_pct"], ovl["net_pnl"]) == (1, 100.0, 120.0)
    assert row_of(t, "session_tag", "London")["trades"] == 4                 # not absorbed into London


def test_direction_breakdown():
    t = an.breakdown(make_trades(), "direction", cfg.DASH_DIRECTION_GROUPS)
    assert t["direction"].to_list() == ["long", "short"]
    lg, sh = row_of(t, "direction", "long"), row_of(t, "direction", "short")
    assert (lg["trades"], lg["net_pnl"], lg["expectancy_r"]) == (3, 150.0, pytest.approx(0.5))
    assert (sh["trades"], sh["net_pnl"], sh["expectancy_r"]) == (3, -70.0, pytest.approx(-0.7 / 3))
    assert lg["win_rate_pct"] == pytest.approx(100 / 3) and sh["win_rate_pct"] == pytest.approx(100 / 3)


def test_yearly_breakdown_includes_the_quiet_year():
    t = an.yearly(make_trades())
    assert t["year"].to_list() == [2024, 2025, 2026]
    y24, y25, y26 = (row_of(t, "year", y) for y in (2024, 2025, 2026))
    assert (y24["trades"], y24["win_rate_pct"], y24["net_pnl"]) == (3, pytest.approx(200 / 3), 250.0)
    assert y25["trades"] == 0 and y25["win_rate_pct"] is None and y25["net_pnl"] == 0.0
    assert (y26["trades"], y26["win_rate_pct"], y26["net_pnl"]) == (3, 0.0, -170.0)


# ==========================================================================
# diagnostics
# ==========================================================================
def test_diagnostics_counts_and_missing_columns_are_not_faked():
    df = make_trades().with_columns(
        pl.when(pl.col("trade_id") == 2).then(True).otherwise(False).alias("sl_tp_both_hit"),
        pl.when(pl.col("trade_id") <= 2).then(True).otherwise(False).alias("margin_capped"),
    )
    d = an.diagnostics(df)
    assert d["both_hit"] == 1 and d["margin_capped"] == 2 and d["funding_estimated"] == 6
    assert d["exit_reasons"] == {"sl": 4, "tp": 2}                         # pnl > 0 -> tp, otherwise sl (see make_trades)
    assert d["avg_cost_r"] == pytest.approx((6 + 3 + 1) / 100)            # (fees+slippage+funding)/risk
    old_log = make_trades().drop("margin_capped")
    assert an.diagnostics(old_log)["margin_capped"] is None               # unknown, NOT reported as 0


def test_missing_required_columns_detected():
    assert an.missing_columns(make_trades().drop("r_multiple")) == ["r_multiple"]


# ==========================================================================
# reconciliation: scripts/reconcile.py (independent code path) must equal analytics
# ==========================================================================
def load_reconcile():
    spec = importlib.util.spec_from_file_location("reconcile_script", ROOT / "scripts" / "reconcile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("spec", [REF, [(datetime(2024, 1, d, tzinfo=UTC), "long", "London", p)
                                         for d, p in ((1, -500.0), (2, 5000.0), (3, -600.0))],
                                  [(datetime(2024, 1, 1, tzinfo=UTC), "long", "London", 100.0)]])
def test_reconcile_script_text_equals_dashboard_block(tmp_path, spec):
    df = make_trades(spec)
    csv_path = tmp_path / "trades.csv"
    df.write_csv(csv_path)
    rec = load_reconcile()
    from_csv = rec.render(rec.recompute(rec.read_trades(csv_path), cfg.STARTING_CAPITAL), cfg.STARTING_CAPITAL)
    assert from_csv == an.format_reconciliation(an.compute_kpis(df))


def test_reconcile_main_flags_csv_that_disagrees_with_parquet(tmp_path, monkeypatch, capsys):
    df = make_trades()
    csv_path, pq_path = tmp_path / "trades.csv", tmp_path / "trades.parquet"
    df.write_csv(csv_path)
    df.write_parquet(pq_path)
    monkeypatch.setattr(cfg, "TRADES_PARQUET_PATH", pq_path)
    rec = load_reconcile()
    assert rec.main(["--csv", str(csv_path)]) == cfg.EXIT_OK
    assert "MATCH" in capsys.readouterr().out

    df.with_columns(pl.when(pl.col("trade_id") == 1).then(pl.col("pnl_net") + 1).otherwise(pl.col("pnl_net")).alias("pnl_net")).write_csv(csv_path)
    assert rec.main(["--csv", str(csv_path)]) == cfg.EXIT_ABORTED
    assert "MISMATCH" in capsys.readouterr().out
