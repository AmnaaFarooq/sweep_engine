"""
Phase 3 tests - synthetic data only.

Market: default 1m candle = (open 95, high 96, low 94, close 95). 03:00 candle has high 100
(Asian high = 100), 05:00 candle has low 90 (Asian low = 90). Scenarios are built from 1m
candles and pushed through the REAL Phase 2 pipeline (resample -> signals), so the 5m
signals and 1m fills are coherent.

Spec constants are written out literally in the expected-value arithmetic on purpose
(fee 0.05%, slippage 2 bps, 2R, risk $100): if config drifts, these tests fail.
"""
import logging
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

import backtester as bt
import config as cfg
import signal_engine as se

UTC = timezone.utc
MIN = timedelta(minutes=1)
D1 = datetime(2024, 3, 4, tzinfo=UTC)
DEFAULT = (95.0, 96.0, 94.0, 95.0)


def at(hh, mm=0, day=0):
    return D1 + timedelta(days=day, hours=hh, minutes=mm)


def five(ts, o, h, l, c):
    """Five identical 1m candles -> a 5m bar with exactly this open/high/low/close."""
    return {ts + k * MIN: (o, h, l, c) for k in range(5)}


def make_market(overrides=None, drop=()):
    ov = {at(3): (95.0, 100.0, 94.0, 95.0), at(5): (95.0, 96.0, 90.0, 95.0)}
    ov.update(overrides or {})
    rows = []
    for i in range(24 * 60):
        ts = D1 + i * MIN
        if ts in drop:
            continue
        o, h, l, c = ov.get(ts, DEFAULT)
        rows.append((ts, o, h, l, c, 1.0))
    candles = pl.DataFrame(rows, schema=["timestamp", "open", "high", "low", "close", "volume"], orient="row")
    candles = candles.with_columns(pl.col("timestamp").cast(pl.Datetime(cfg.TIME_UNIT, cfg.TIMEZONE)))
    signals = se.to_output(se.compute_signals(se.resample_5m(candles)))
    return candles, signals


def backtest(candles, signals, funding=None):
    return bt.run_backtest(signals, candles, funding or bt.FundingSchedule(None), "wick_only")


# 09:00 5m bar: wick above 100, closes back inside at 99.5  -> short sweep.  09:05 bar closes 94.5 -> confirmation.
SHORT_SWEEP = {**five(at(9, 0), 99.0, 101.0, 98.0, 99.5), **five(at(9, 5), 95.0, 96.0, 94.0, 94.5)}


# ==========================================================================
# (a) sizing math
# ==========================================================================
def test_a_sizing_entry100_stop98_risk100_gives_size_50():
    r = bt.compute_position_size(entry_price=100.0, stop_price=98.0, risk_usd=100.0, equity=10_000.0, max_leverage=25)
    assert r.size_by_risk == pytest.approx(50.0)
    assert r.size == pytest.approx(50.0) and not r.capped       # cap = 10_000*25/100 = 2500, not binding
    assert r.risk_actual == pytest.approx(100.0)


def test_a_sizing_margin_cap_reduces_size_when_required():
    # equity 100 at 25x -> max notional 2500 -> max size 25 at price 100 (< 50 needed by risk)
    r = bt.compute_position_size(100.0, 98.0, 100.0, equity=100.0, max_leverage=25)
    assert r.size_by_risk == pytest.approx(50.0)
    assert r.size_cap == pytest.approx(25.0)
    assert r.size == pytest.approx(25.0) and r.capped
    assert r.risk_actual == pytest.approx(50.0)                 # risk really fell below $100 - flagged, not hidden
    # realistic BTC case: tiny stop (0.0167%) -> risk-sized notional would be 6x the 25x cap
    r = bt.compute_position_size(60_000.0, 59_990.0, 100.0, 10_000.0, 25)
    assert r.size_by_risk == pytest.approx(10.0) and r.size == pytest.approx(250_000 / 60_000) and r.capped


def test_a_sizing_rejects_zero_stop_distance():
    with pytest.raises(ValueError):
        bt.compute_position_size(100.0, 100.0, 100.0, 10_000.0, 25)


def test_a_stop_buffer_and_tp_for_both_directions():
    # ATR dominates: 0.25*8 = 2.0 > 0.001*100 = 0.1
    stop, tp, buf = bt.compute_stop_tp(cfg.DIR_SHORT, 100.0, sweep_price=104.0, atr=8.0)
    assert (buf, stop, tp) == (2.0, 106.0, 100.0 - 2 * 6.0)
    stop, tp, buf = bt.compute_stop_tp(cfg.DIR_LONG, 100.0, sweep_price=96.0, atr=8.0)
    assert (buf, stop, tp) == (2.0, 94.0, 100.0 + 2 * 6.0)
    # percent floor dominates: 0.001*100 = 0.1 > 0.25*0.2 = 0.05
    assert bt.compute_stop_tp(cfg.DIR_SHORT, 100.0, 104.0, atr=0.2)[2] == pytest.approx(0.1)


def test_risk_usd_is_one_percent_of_capital():
    assert cfg.RISK_USD == pytest.approx(100.0) and cfg.STARTING_CAPITAL == 10_000.0


# ==========================================================================
# (b) one synthetic trade: pnl_net after fees + slippage within 1e-6
# ==========================================================================
def short_tp_scenario():
    ov = dict(SHORT_SWEEP)
    ov.update({
        at(9, 10): (95.0, 95.5, 94.5, 95.0), at(9, 11): (95.0, 95.5, 94.5, 95.0),
        at(9, 12): (95.0, 95.5, 81.0, 82.0),                    # touches TP only
        at(9, 13): (82.0, 83.0, 81.0, 82.0), at(9, 14): (82.0, 83.0, 81.0, 82.0),
    })
    return make_market(ov)


def test_b_single_short_trade_pnl_net_matches_hand_calculation():
    candles, signals = short_tp_scenario()
    # hand check of the ATR the engine must see at the 09:05 confirmation bar:
    # 12 default bars TR=2 (24) + sweep bar TR=6 (|101-95|) + confirm bar TR=5.5 (|94-99.5|) = 35.5 ; /14
    atr = signals.filter(pl.col("timestamp") == at(9, 5))["atr_14"][0]
    assert atr == pytest.approx(35.5 / 14, abs=1e-12)

    res = backtest(candles, signals)
    assert res.trades.height == 1 and not res.rejections
    t = res.trades.row(0, named=True)

    # ---- independent hand arithmetic (short) ----
    buffer = max(0.001 * 95.0, 0.25 * (35.5 / 14))          # = 0.633928...  (ATR term wins)
    stop = 101.0 + buffer                                   # sweep high 101 + buffer
    dist = stop - 95.0
    size = 100.0 / dist
    tp = 95.0 - 2.0 * dist
    entry_fill = 95.0 * (1 - 0.0002)                        # short sells lower
    exit_fill = tp * (1 + 0.0002)                           # short buys back higher
    gross = size * (95.0 - tp)                              # = 200 = 2R
    slippage = size * (95.0 - entry_fill) + size * (exit_fill - tp)
    fees = 0.0005 * size * (entry_fill + exit_fill)
    expected_net = gross - slippage - fees                  # no funding boundary crossed

    assert t["direction"] == "short" and t["exit_reason"] == cfg.EXIT_TP and t["sl_tp_both_hit"] is False
    assert t["sweep_price"] == 101.0 and t["session_tag"] == cfg.TAG_LONDON and t["sweep_mode"] == "wick_only"
    assert t["signal_time"] == at(9, 5) and t["entry_time"] == at(9, 10) and t["exit_time"] == at(9, 12)
    assert t["entry_price"] == 95.0                          # next bar's OPEN ...
    assert t["entry_price"] != 94.5                          # ... not the confirmation bar's close
    assert t["stop_price"] == pytest.approx(stop, abs=1e-9) and t["tp_price"] == pytest.approx(tp, abs=1e-9)
    assert t["size"] == pytest.approx(size, abs=1e-9) and t["risk_usd"] == pytest.approx(100.0, abs=1e-9)
    assert t["pnl_gross"] == pytest.approx(200.0, abs=1e-6)
    assert t["slippage_cost"] == pytest.approx(slippage, abs=1e-6)
    assert t["fees"] == pytest.approx(fees, abs=1e-6)
    assert t["funding_paid"] == 0.0 and t["n_funding_boundaries"] == 0
    assert t["pnl_net"] == pytest.approx(expected_net, abs=1e-6)
    assert t["r_multiple"] == pytest.approx(expected_net / 100.0, abs=1e-9)
    assert list(res.trades.columns) == cfg.TRADE_COLUMNS


def test_b_single_long_trade_mirrors_the_short():
    ov = {**five(at(9, 0), 91.0, 92.0, 89.0, 90.5), **five(at(9, 5), 95.0, 96.0, 94.0, 94.5),
          at(9, 12): (95.0, 109.0, 94.5, 108.0)}              # touches TP only
    candles, signals = make_market(ov)
    t = backtest(candles, signals).trades.row(0, named=True)

    buffer = max(0.001 * 95.0, 0.25 * (35.5 / 14))
    stop = 89.0 - buffer                                     # sweep low 89 - buffer
    dist = 95.0 - stop
    size, tp = 100.0 / dist, 95.0 + 2.0 * dist
    entry_fill, exit_fill = 95.0 * (1 + 0.0002), tp * (1 - 0.0002)      # long pays up, sells lower
    expected = (size * (tp - 95.0)
                - (size * (entry_fill - 95.0) + size * (tp - exit_fill))
                - 0.0005 * size * (entry_fill + exit_fill))
    assert t["direction"] == "long" and t["exit_reason"] == cfg.EXIT_TP and t["sweep_price"] == 89.0
    assert t["pnl_net"] == pytest.approx(expected, abs=1e-6)


def test_session_end_exit_at_force_exit_hour():
    candles, signals = make_market(SHORT_SWEEP)              # nothing ever touches stop or TP
    res = backtest(candles, signals)
    t = res.trades.row(0, named=True)
    assert t["exit_reason"] == cfg.EXIT_SESSION_END
    assert t["exit_time"] == at(20, 59) and t["exit_price"] == 95.0     # close of the last candle before 21:00


# ==========================================================================
# (c) sl_tp_both_hit
# ==========================================================================
def test_c_candle_spanning_stop_and_tp_is_a_pessimistic_stop_loss():
    ov = dict(SHORT_SWEEP)
    ov[at(9, 12)] = (95.0, 102.5, 80.0, 90.0)                # high >= stop (101.63) AND low <= tp (81.73)
    candles, signals = make_market(ov)
    res = backtest(candles, signals)
    t = res.trades.row(0, named=True)
    assert t["sl_tp_both_hit"] is True and t["exit_reason"] == cfg.EXIT_BOTH_SL_FIRST
    assert t["exit_time"] == at(9, 12)
    assert t["exit_price"] == pytest.approx(t["stop_price"])          # filled at the STOP, not the tp
    assert t["pnl_gross"] < 0 and t["r_multiple"] < -1.0              # a full loss plus costs


def test_c_resolve_candle_unit_cases():
    L, S = cfg.DIR_LONG, cfg.DIR_SHORT
    assert bt.resolve_candle(L, stop=98, tp=104, high=105, low=97) == (cfg.EXIT_BOTH_SL_FIRST, True)
    assert bt.resolve_candle(S, stop=102, tp=96, high=103, low=95) == (cfg.EXIT_BOTH_SL_FIRST, True)
    assert bt.resolve_candle(L, 98, 104, high=105, low=99) == (cfg.EXIT_TP, False)
    assert bt.resolve_candle(L, 98, 104, high=103, low=98) == (cfg.EXIT_SL, False)         # touching counts
    assert bt.resolve_candle(S, 102, 96, high=101.9, low=96.0) == (cfg.EXIT_TP, False)
    assert bt.resolve_candle(S, 102, 96, high=101.9, low=96.1) == (None, False)


# ==========================================================================
# (d) funding: once per 8h boundary crossed
# ==========================================================================
def test_d_funding_boundaries_counted_exactly_once_each():
    f = bt.funding_boundaries
    assert f(at(15, 40), at(17, 0)) == [at(16)]                              # crosses 16:00 once
    assert f(at(7, 50), at(16, 10)) == [at(8), at(16)]                       # two boundaries
    assert f(at(23, 50), at(0, 10, day=1)) == [at(0, 0, day=1)]              # across midnight
    assert f(at(8, 10), at(15, 50)) == []                                    # none
    assert f(at(15, 40), at(16, 0)) == [at(16)]                              # exit candle opening AT the boundary counts
    assert f(at(16, 0), at(17, 0)) == []                                     # entry exactly at boundary does not
    assert f(at(1), at(0, 0, day=3)) == [at(8), at(16), at(0, day=1), at(8, day=1), at(16, day=1),
                                         at(0, day=2), at(8, day=2), at(16, day=2), at(0, day=3)]


def late_trade_scenario(direction):
    """Sweep 15:30 -> confirm 15:35 -> entry 15:40 -> TP only at 17:00, so 16:00 is crossed once."""
    if direction == "short":
        ov = {**five(at(15, 30), 99.0, 101.0, 98.0, 99.5), **five(at(15, 35), 95.0, 96.0, 94.0, 94.5),
              at(17, 0): (95.0, 95.5, 81.0, 82.0)}
    else:
        ov = {**five(at(15, 30), 91.0, 92.0, 89.0, 90.5), **five(at(15, 35), 95.0, 96.0, 94.0, 94.5),
              at(17, 0): (95.0, 109.0, 94.5, 108.0)}
    return make_market(ov)


@pytest.mark.parametrize("direction, sign", [("short", -1), ("long", +1)])
def test_d_estimated_funding_charged_once_with_correct_sign(direction, sign, caplog):
    candles, signals = late_trade_scenario(direction)
    with caplog.at_level(logging.INFO, logger="backtester"):
        res = backtest(candles, signals)
    t = res.trades.row(0, named=True)
    assert t["exit_reason"] == cfg.EXIT_TP and t["exit_time"] == at(17, 0)
    assert t["n_funding_boundaries"] == 1
    # notional uses the 1m open at the boundary (95); +0.01% rate => longs pay, shorts receive
    assert t["funding_paid"] == pytest.approx(sign * t["size"] * 95.0 * 0.0001, abs=1e-9)
    assert t["funding_estimated"] is True
    assert sum("FUNDING trade" in r.message for r in caplog.records) == 1
    net_without_funding = t["pnl_gross"] - t["slippage_cost"] - t["fees"]
    assert t["pnl_net"] == pytest.approx(net_without_funding - t["funding_paid"], abs=1e-9)


def test_d_historical_funding_rate_is_used_and_not_flagged_estimated():
    candles, signals = late_trade_scenario("short")
    schedule = bt.FundingSchedule({at(16): 0.0003})
    t = backtest(candles, signals, schedule).trades.row(0, named=True)
    assert t["funding_paid"] == pytest.approx(-1 * t["size"] * 95.0 * 0.0003, abs=1e-9)
    assert t["funding_estimated"] is False


def test_d_boundary_missing_from_history_falls_back_to_estimate_and_flags_it():
    candles, signals = late_trade_scenario("short")
    t = backtest(candles, signals, bt.FundingSchedule({at(8): 0.0003})).trades.row(0, named=True)   # no 16:00 entry
    assert t["funding_paid"] == pytest.approx(-1 * t["size"] * 95.0 * 0.0001, abs=1e-9)
    assert t["funding_estimated"] is True


# ==========================================================================
# Rejections, gaps, look-ahead guards
# ==========================================================================
def test_signal_while_a_position_is_open_is_rejected_not_stacked():
    ov = {**SHORT_SWEEP, **five(at(10, 0), 91.0, 92.0, 89.0, 90.5)}      # long sweep+confirm while short is open
    candles, signals = make_market(ov)
    res = backtest(candles, signals)
    assert res.trades.height == 1
    assert any("already active" in r["reason"] and r["direction"] == "long" for r in res.rejections)


def test_stop_not_beyond_entry_after_a_gap_is_rejected():
    ov = {**SHORT_SWEEP, **five(at(9, 10), 105.0, 106.0, 104.0, 105.0)}  # next bar opens ABOVE the stop (101.63)
    candles, signals = make_market(ov)
    res = backtest(candles, signals)
    assert res.trades.height == 0
    assert any("not beyond entry" in r["reason"] for r in res.rejections)


def test_trade_is_discarded_when_a_1m_candle_is_missing_while_open():
    candles, signals = make_market(SHORT_SWEEP, drop={at(9, 22)})        # 09:20 5m bar becomes partial -> dropped
    res = backtest(candles, signals)
    assert res.trades.height == 0 and len(res.discarded) == 1
    assert "data gap" in res.discarded[0]["reason"]


def test_missing_entry_bar_rejects_the_order():
    candles, signals = make_market(SHORT_SWEEP, drop={at(9, 12)})        # the entry bar 09:10 is partial -> dropped
    res = backtest(candles, signals)
    assert res.trades.height == 0
    assert any("missing from data" in r["reason"] for r in res.rejections)


def test_market_view_refuses_to_show_the_future():
    candles, signals = make_market()
    view = bt.MarketView(signals, candles)
    i = signals["timestamp"].to_list().index(at(9, 0))
    view.advance(at(9, 0))
    assert view.bar_open(i) == 95.0                                      # the open is known the instant the bar starts
    with pytest.raises(bt.LookAheadError):
        view.closed_bar(i)                                               # the bar has not closed yet
    j = view.idx_1m(at(9, 0))
    with pytest.raises(bt.LookAheadError):
        view.closed_1m(j)                                                # the 1m candle has not closed yet
    with pytest.raises(bt.LookAheadError):
        view.open_1m(view.idx_1m(at(9, 1)))                              # a candle that has not started
    view.advance(at(9, 5))
    assert view.closed_bar(i).close == 95.0                              # now it is visible
    with pytest.raises(bt.LookAheadError):
        view.advance(at(9, 0))                                           # the clock never runs backwards


def test_entry_rule_assertion_fires_if_entry_bar_is_not_the_next_bar():
    candles, signals = make_market(SHORT_SWEEP)
    view = bt.MarketView(signals, candles)
    i = signals["timestamp"].to_list().index(at(9, 10))
    view.advance(at(9, 10))
    wrong = bt.Order(cfg.DIR_SHORT, at(9, 0), at(9, 10), 101.0, 2.0, cfg.TAG_LONDON)   # confirm 09:00 -> entry must be 09:05
    with pytest.raises(AssertionError):
        bt._open_position(view, wrong, i, 10_000.0, [])
