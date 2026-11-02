"""Phase 4 tests - run the real Streamlit app headlessly (streamlit.testing AppTest) on synthetic trade logs."""
from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

import analytics as an
import config as cfg
from test_analytics import make_trades   # pytest puts tests/ on sys.path (rootdir-relative imports)

APP = Path(__file__).resolve().parents[1] / "app.py"


def run_app():
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    return at


def test_dashboard_renders_and_every_kpi_matches_the_analytics_block(tmp_path, monkeypatch):
    df = make_trades()
    path = tmp_path / "trades.parquet"
    df.write_parquet(path)
    monkeypatch.setattr(cfg, "TRADES_PARQUET_PATH", path)

    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    m = {x.label: x.value for x in at.metric}
    k = an.compute_kpis(df)
    assert m["Total Trades"] == "6"
    assert m["Win Rate"] == f"{k.win_rate_pct:.2f}%" == "33.33%"
    assert m["Profit Factor"] == "1.30"
    assert m["Net PnL"] == "$80.00"
    assert m["Expectancy (R)"] == "+0.133 R" and m["Average Win R"] == "+1.750 R" and m["Average Loss R"] == "-0.900 R"
    assert m["Max Drawdown"] == "-$170.00 / 1.66%"
    # the exact-values box is the reconcile text
    assert at.code[0].value == an.format_reconciliation(k)
    # placeholders are labelled and never show a number
    assert [m[n] for n in cfg.DASH_PLACEHOLDERS] == [cfg.DASH_NOT_COMPUTED] * 3
    # diagnostics
    assert m["SL and TP in same 1m candle"] == "0" and m["Funding estimated"] == "6 of 6" and m["Margin cap reduced size"] == "0"
    # net pnl delta is % of starting capital
    assert [x.delta for x in at.metric if x.label == "Net PnL"] == ["+0.80%"]


def test_empty_trade_log_does_not_crash_or_invent_numbers(tmp_path, monkeypatch):
    path = tmp_path / "trades.parquet"
    make_trades().clear().write_parquet(path)
    monkeypatch.setattr(cfg, "TRADES_PARQUET_PATH", path)
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    m = {x.label: x.value for x in at.metric}
    assert m["Total Trades"] == "0" and m["Win Rate"] == "n/a" and m["Profit Factor"] == "n/a"
    assert any("empty" in w.value for w in at.warning)


def test_missing_trade_file_shows_an_error_not_a_traceback(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "TRADES_PARQUET_PATH", tmp_path / "nope.parquet")
    at = run_app()
    assert not at.exception
    assert any("not found" in e.value for e in at.error)


def test_filters_only_change_the_table_not_the_kpis(tmp_path, monkeypatch):
    path = tmp_path / "trades.parquet"
    make_trades().write_parquet(path)
    monkeypatch.setattr(cfg, "TRADES_PARQUET_PATH", path)
    at = run_app()
    direction_filter = next(w for w in at.multiselect if w.label == "direction")
    direction_filter.unselect("long").run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["Total Trades"] == "6" and m["Net PnL"] == "$80.00"             # unchanged by the filter
