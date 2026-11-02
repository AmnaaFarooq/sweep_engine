"""
Phase 2 tests - synthetic data only.

Synthetic day layout (every bar is 5m, UTC):
  * default bar: open=close=95, high=96, low=94  (well inside the Asian range)
  * 03:00 bar has high=100   -> asian_high = 100
  * 05:00 bar has low=90     -> asian_low  = 90
  So after 08:00 the levels are 100 (short side) and 90 (long side).
"""
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

import config as cfg
import signal_engine as se

UTC = timezone.utc
BAR = cfg.BAR_DURATION
BARS_PER_DAY = 24 * cfg.MINUTES_PER_HOUR // cfg.BAR_MINUTES            # 288
D1 = datetime(2024, 3, 4, tzinfo=UTC)

DEFAULT_BAR = (95.0, 96.0, 94.0, 95.0)       # open, high, low, close


def at(hh, mm=0, day=0):
    return D1 + timedelta(days=day, hours=hh, minutes=mm)


def make_days(n_days=1, overrides=None, drop=()):
    """Build n_days of complete 5m bars. overrides: {datetime: (o, h, l, c)}; drop: datetimes to omit."""
    overrides = dict(overrides or {})
    rows = []
    for day in range(n_days):
        overrides.setdefault(at(3, 0, day), (95.0, 100.0, 94.0, 95.0))    # Asian high = 100
        overrides.setdefault(at(5, 0, day), (95.0, 96.0, 90.0, 95.0))     # Asian low  = 90
    for i in range(n_days * BARS_PER_DAY):
        ts = D1 + i * BAR
        if ts in drop:
            continue
        o, h, l, c = overrides.get(ts, DEFAULT_BAR)
        rows.append((ts, o, h, l, c, 1.0))
    df = pl.DataFrame(rows, schema=["timestamp", "open", "high", "low", "close", "volume"], orient="row")
    return df.with_columns(pl.col("timestamp").cast(pl.Datetime(cfg.TIME_UNIT, cfg.TIMEZONE)))


def run(df, mode=None):
    return se.to_output(se.compute_signals(df, sweep_mode=mode))


def rows_where(out, col):
    return out.filter(pl.col(col))["timestamp"].to_list()


def status_at(out, ts):
    return out.filter(pl.col("timestamp") == ts)["signal_status"][0]


# --------------------------------------------------------------------------
# (a) one clean sweep -> exactly one confirmed signal
# --------------------------------------------------------------------------
def test_a_clean_sweep_produces_one_confirmed_signal():
    # 09:00: wick above 100, closes back at 99.5  -> short sweep.
    # 09:05: default bar closes at 95 (< 100)     -> confirmation.
    df = make_days(overrides={at(9, 0): (99.0, 101.0, 98.0, 99.5)})
    out = run(df)

    assert out.columns == cfg.SIGNAL_COLUMNS
    assert rows_where(out, "sweep_short") == [at(9, 0)]
    assert rows_where(out, "confirm_short") == [at(9, 5)]
    assert rows_where(out, "sweep_long") == [] and rows_where(out, "confirm_long") == []
    assert status_at(out, at(9, 0)) == cfg.STATUS_PENDING
    assert status_at(out, at(9, 5)) == cfg.STATUS_CONFIRMED
    assert out.filter(pl.col("signal_status") == cfg.STATUS_CONFIRMED).height == 1
    assert out.filter(pl.col("signal_status") == cfg.STATUS_EXPIRED).height == 0
    assert out.filter(pl.col("signal_status") == cfg.STATUS_REJECTED).height == 0


def test_a2_clean_long_sweep():
    # 10:00 wicks below 90 and closes back at 90.5 -> long sweep; 10:05 default (close 95 > 90) confirms.
    df = make_days(overrides={at(10, 0): (91.0, 92.0, 89.0, 90.5)})
    out = run(df)
    assert rows_where(out, "sweep_long") == [at(10, 0)]
    assert rows_where(out, "confirm_long") == [at(10, 5)]
    assert rows_where(out, "sweep_short") == []


# --------------------------------------------------------------------------
# (b) genuine breakout (close beyond) -> zero sweeps in wick_only
# --------------------------------------------------------------------------
def breakout_day():
    # From 09:00 price closes ABOVE 100 and stays there: a breakout, never closes back inside.
    ov = {at(9, 0) + i * BAR: (100.5, 103.0, 100.2, 102.0) for i in range(0, 12 * 12)}   # 09:00 .. 20:55
    return make_days(overrides=ov)


def test_b_breakout_is_not_a_sweep_in_wick_only():
    out = run(breakout_day(), mode=cfg.SWEEP_MODE_WICK)
    assert out.filter(pl.col("sweep_short") | pl.col("sweep_long")).height == 0
    assert out.filter(pl.col("confirm_short") | pl.col("confirm_long")).height == 0
    assert out["signal_status"].null_count() == out.height          # no signal activity at all


def test_b2_same_breakout_is_a_sweep_in_close_beyond_but_never_confirms():
    out = run(breakout_day(), mode=cfg.SWEEP_MODE_CLOSE)
    # first London sweep at 09:00; NY lifetime session starts 16:00 and gets its own first sweep
    assert rows_where(out, "sweep_short") == [at(9, 0), at(16, 0)]
    assert out.filter(pl.col("confirm_short")).height == 0
    assert rows_where(out.filter(pl.col("signal_status") == cfg.STATUS_EXPIRED), "sweep_short") == []
    expired = out.filter(pl.col("signal_status") == cfg.STATUS_EXPIRED)["timestamp"].to_list()
    assert expired == [at(15, 55), at(20, 55)]                      # last bar of London / NY lifetime sessions


# --------------------------------------------------------------------------
# (c) a signal that never confirms is marked expired
# --------------------------------------------------------------------------
def test_c_unconfirmed_signal_expires_at_session_end():
    # 09:00 wick sweep (closes back at 99.5), then price closes ABOVE 100 for the rest of the day:
    # no later close back inside -> never confirms.
    ov = {at(9, 0): (99.0, 101.0, 98.0, 99.5)}
    ov.update({at(9, 5) + i * BAR: (100.5, 103.0, 100.2, 102.0) for i in range(0, 130)})   # 09:05 .. 19:50
    out = run(make_days(overrides=ov))                              # default mode: wick_only

    assert rows_where(out, "sweep_short") == [at(9, 0)]
    assert out.filter(pl.col("confirm_short")).height == 0
    assert status_at(out, at(9, 0)) == cfg.STATUS_PENDING
    assert status_at(out, at(12, 0)) == cfg.STATUS_PENDING          # still pending mid-session
    assert status_at(out, at(15, 55)) == cfg.STATUS_EXPIRED         # last scheduled London bar
    assert out.filter(pl.col("signal_status") == cfg.STATUS_EXPIRED).height == 1


# --------------------------------------------------------------------------
# (d) ATR(14) vs hand-computed values on 20 synthetic bars
# --------------------------------------------------------------------------
# (high, low, close) per bar; open = close. True range by hand (|H-pc| = |high-prev_close|, |L-pc| = |low-prev_close|):
#  bar : H    L    C   prev_c  H-L  |H-pc| |L-pc|  TR
#   0  : 102  98   100   -      -     -      -      (none: no previous close)
#   1  : 103  99   102   100    4     3      1      4
#   2  : 105  101  104   102    4     3      1      4
#   3  : 108  103  107   104    5     4      1      5
#   4  : 110  106  109   107    4     3      1      4
#   5  : 115  111  114   109    4     6      2      6   <- gap up: |H-pc| dominates
#   6  : 114  110  111   114    4     0      4      4
#   7  : 112  104  105   111    8     1      7      8
#   8  : 100  96   97    105    4     5      9      9   <- gap down: |L-pc| dominates
#   9  : 99   95   98    97     4     2      2      4
#  10  : 101  97   100   98     4     3      1      4
#  11  : 104  99   103   100    5     4      1      5
#  12  : 106  102  105   103    4     3      1      4
#  13  : 107  103  104   105    4     2      2      4
#  14  : 108  104  107   104    4     4      0      4
#  15  : 111  106  110   107    5     4      1      5
#  16  : 112  108  109   110    4     2      2      4
#  17  : 110  105  106   109    5     1      4      5
#  18  : 109  104  108   106    5     3      2      5
#  19  : 112  107  111   108    5     4      1      5
# ATR(14) at bar 14 = mean(TR 1..14) = 69/14.  Rolling by one bar: add the new TR, drop the oldest:
#  bar15: (69 - 4 + 5)/14 = 70/14 = 5.0      bar16: (70 - 4 + 4)/14 = 5.0     bar17: (70 - 5 + 5)/14 = 5.0
#  bar18: (70 - 4 + 5)/14 = 71/14            bar19: (71 - 6 + 5)/14 = 70/14 = 5.0
ATR_BARS = [
    (102, 98, 100), (103, 99, 102), (105, 101, 104), (108, 103, 107), (110, 106, 109),
    (115, 111, 114), (114, 110, 111), (112, 104, 105), (100, 96, 97), (99, 95, 98),
    (101, 97, 100), (104, 99, 103), (106, 102, 105), (107, 103, 104), (108, 104, 107),
    (111, 106, 110), (112, 108, 109), (110, 105, 106), (109, 104, 108), (112, 107, 111),
]
ATR_EXPECTED = {14: 69 / 14, 15: 5.0, 16: 5.0, 17: 5.0, 18: 71 / 14, 19: 5.0}


def atr_frame(bars, start=D1, skip=()):
    rows = [(start + i * BAR, float(c), float(h), float(l), float(c), 1.0)
            for i, (h, l, c) in enumerate(bars) if i not in skip]
    df = pl.DataFrame(rows, schema=["timestamp", "open", "high", "low", "close", "volume"], orient="row")
    return df.with_columns(pl.col("timestamp").cast(pl.Datetime(cfg.TIME_UNIT, cfg.TIMEZONE)))


def test_d_atr14_matches_hand_computed_values():
    out = se.add_atr(atr_frame(ATR_BARS))
    atr = out[cfg.ATR_COLUMN].to_list()
    assert all(v is None for v in atr[:14])                         # needs 14 true ranges -> first value at bar 14
    for idx, expected in ATR_EXPECTED.items():
        assert atr[idx] == pytest.approx(expected, abs=1e-12), f"bar {idx}"
    # the two "dominant term" bars the hand table highlights
    tr = out["true_range"].to_list()
    assert tr[5] == 6 and tr[8] == 9 and tr[0] is None


def test_d2_atr_is_null_after_a_gap_instead_of_silently_bridging_it():
    out = se.add_atr(atr_frame(ATR_BARS, skip={10}))               # bar 10 missing -> 5m series has a hole
    atr = out[cfg.ATR_COLUMN].to_list()
    assert all(v is None for v in atr)                              # only 19 rows; bar 11 has no valid prev_close


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------
def test_session_tags_including_overlap():
    out = se.add_session_tags(make_days())
    tag = lambda hh, mm=0: out.filter(pl.col("timestamp") == at(hh, mm))["session_tag"][0]
    assert tag(0) == cfg.TAG_ASIAN and tag(7, 55) == cfg.TAG_ASIAN
    assert tag(8) == cfg.TAG_LONDON and tag(12, 55) == cfg.TAG_LONDON
    assert tag(13) == cfg.TAG_OVERLAP and tag(15, 55) == cfg.TAG_OVERLAP      # never plain London / NY
    assert tag(16) == cfg.TAG_NY and tag(20, 55) == cfg.TAG_NY
    assert tag(21) == cfg.TAG_OFF and tag(23, 55) == cfg.TAG_OFF


# --------------------------------------------------------------------------
# Look-ahead protection
# --------------------------------------------------------------------------
def test_asian_level_is_hidden_until_the_asian_session_closes():
    out = run(make_days())
    before = out.filter(pl.col("timestamp") < at(8))
    after = out.filter(pl.col("timestamp") >= at(8))
    assert before["asian_high"].null_count() == before.height       # Asian bars, incl. 07:55, see no level
    assert before["asian_low"].null_count() == before.height
    assert (after["asian_high"] == 100.0).all() and (after["asian_low"] == 90.0).all()


def test_asian_level_ignores_post_asian_bars():
    # a huge spike at 10:00 must NOT leak into the Asian high
    out = run(make_days(overrides={at(10, 0): (95.0, 500.0, 94.0, 95.0)}))
    assert (out.filter(pl.col("timestamp") >= at(8))["asian_high"] == 100.0).all()


def test_signals_are_causal_truncating_the_future_changes_nothing():
    """Output on bars before a cut must be identical whether or not the later bars exist."""
    ov = {
        at(9, 0, 1): (99.0, 101.0, 98.0, 99.5),            # day 2 short sweep, confirms at 09:05
        at(10, 0, 1): (91.0, 92.0, 89.0, 90.5),            # day 2 long sweep
        at(11, 0, 1): (99.0, 101.5, 98.0, 99.0),           # later duplicate short sweep -> rejected
        at(17, 0, 1): (99.0, 101.0, 98.0, 99.5),           # NY-session sweep
    }
    full_in = make_days(n_days=3, overrides=ov)
    full_out = run(full_in)
    for cut in (at(6, 0, 1), at(9, 2, 1), at(9, 30, 1), at(14, 0, 1), at(18, 0, 1), at(23, 0, 1)):
        trunc_out = run(full_in.filter(pl.col("timestamp") < cut))
        expected = full_out.filter(pl.col("timestamp") < cut)
        assert trunc_out.equals(expected), f"output changed when future bars after {cut} were removed"


# --------------------------------------------------------------------------
# Lifetime rules
# --------------------------------------------------------------------------
def test_only_first_sweep_per_session_and_level_later_ones_are_rejected():
    df = make_days(overrides={
        at(9, 0): (99.0, 101.0, 98.0, 99.5),      # first London sweep
        at(10, 0): (99.0, 101.5, 98.0, 99.0),     # second London sweep of the same level -> ignored
        at(17, 0): (99.0, 101.0, 98.0, 99.5),     # NY session: new "first sweep"
    })
    out = run(df)
    assert rows_where(out, "sweep_short") == [at(9, 0), at(17, 0)]
    assert status_at(out, at(10, 0)) == cfg.STATUS_REJECTED
    assert rows_where(out, "confirm_short") == [at(9, 5), at(17, 5)]


def test_overlap_sweep_is_tagged_overlap_but_lifetime_belongs_to_london():
    out = run(make_days(overrides={at(14, 0): (99.0, 101.0, 98.0, 99.5)}))
    row = out.filter(pl.col("sweep_short"))
    assert row["session_tag"].to_list() == [cfg.TAG_OVERLAP]
    assert rows_where(out, "confirm_short") == [at(14, 5)]


def test_confirm_on_sweep_bar_flag(monkeypatch):
    df = make_days(overrides={at(9, 0): (99.0, 101.0, 98.0, 99.5)})
    monkeypatch.setattr(cfg, "CONFIRM_ON_SWEEP_BAR", True)
    out = run(df)
    assert rows_where(out, "sweep_short") == [at(9, 0)]
    assert rows_where(out, "confirm_short") == [at(9, 0)]          # sweep candle itself closed back inside
    assert status_at(out, at(9, 0)) == cfg.STATUS_CONFIRMED


def test_unknown_sweep_mode_raises():
    with pytest.raises(ValueError):
        run(make_days(), mode="nonsense")


# --------------------------------------------------------------------------
# Gaps / resampling
# --------------------------------------------------------------------------
def test_resample_drops_partial_5m_bars_and_aggregates_correctly():
    mins = [m for m in range(0, 20) if m != 7]                       # 1m bar 00:07 missing -> 00:05 bucket has 4 bars
    rows = [(D1 + timedelta(minutes=m), float(m), float(m) + 10, float(m) - 1, float(m) + 0.5, 2.0) for m in mins]
    df_1m = pl.DataFrame(rows, schema=["timestamp", "open", "high", "low", "close", "volume"], orient="row")
    df_1m = df_1m.with_columns(pl.col("timestamp").cast(pl.Datetime(cfg.TIME_UNIT, cfg.TIMEZONE)))

    out = se.resample_5m(df_1m)
    assert out["timestamp"].to_list() == [at(0, 0), at(0, 10), at(0, 15)]   # 00:05 dropped, not filled
    first = out.row(0, named=True)                                   # minutes 0-4
    assert first == {"timestamp": at(0, 0), "open": 0.0, "high": 14.0, "low": -1.0, "close": 4.5, "volume": 10.0}


def test_bar_after_a_gap_cannot_carry_a_sweep():
    # drop the 08:55 bar -> the 09:00 bar follows a hole, so its sweep pattern must be ignored
    df = make_days(overrides={at(9, 0): (99.0, 101.0, 98.0, 99.5)}, drop={at(8, 55)})
    out = run(df)
    assert out.filter(pl.col("sweep_short")).height == 0


def test_incomplete_asian_session_disables_the_day():
    df = make_days(overrides={at(9, 0): (99.0, 101.0, 98.0, 99.5)}, drop={at(2, 0)})
    out = run(df)
    assert out["asian_high"].null_count() == out.height
    assert out.filter(pl.col("sweep_short")).height == 0


def test_gap_while_pending_rejects_the_signal():
    # sweep 09:00, hole at 09:05-09:10 -> 09:15 follows a gap -> pending signal invalidated, no confirmation after
    df = make_days(overrides={at(9, 0): (99.0, 101.0, 98.0, 99.5)}, drop={at(9, 5), at(9, 10)})
    out = run(df)
    assert rows_where(out, "sweep_short") == [at(9, 0)]
    assert out.filter(pl.col("confirm_short")).height == 0
    assert status_at(out, at(9, 15)) == cfg.STATUS_REJECTED
