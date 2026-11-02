"""Phase 1 tests: validator and gap detection (synthetic data only, no network)."""
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

import config as cfg
from utils import validation as v

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
MINUTE = timedelta(seconds=cfg.EXPECTED_BAR_SECONDS)


def make_frame(minute_offsets, **overrides):
    """Build a valid OHLCV frame at the given minute offsets from T0 (UTC)."""
    n = len(minute_offsets)
    data = {
        "timestamp": [T0 + k * MINUTE for k in minute_offsets],
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.5] * n, "volume": [10.0] * n,
    }
    data.update(overrides)
    df = pl.DataFrame(data)
    return df.with_columns(pl.col("timestamp").cast(pl.Datetime(cfg.TIME_UNIT, cfg.TIMEZONE)))


def test_clean_frame_passes():
    result = v.validate_ohlcv(make_frame(range(10)), label="clean")
    assert result.ok, result.describe()


def test_duplicate_nonmonotonic_and_ohlc_violation_are_all_caught():
    # offsets: 0,1,2,2 (duplicate of 2),  5,4 (4 goes backwards after 5),  6
    offsets = [0, 1, 2, 2, 5, 4, 6]
    highs = [101.0] * len(offsets)
    highs[6] = 99.5                    # last row: high (99.5) < close (100.5)  -> OHLC violation
    df = make_frame(offsets, high=highs)

    result = v.validate_ohlcv(df, label="synthetic")

    assert not result.ok
    assert v.CHECK_DUPLICATE in result.checks()
    assert v.CHECK_NON_MONOTONIC in result.checks()
    assert v.CHECK_OHLC in result.checks()
    # each check should point at exactly the planted rows
    counts = {x.check: x.count for x in result.violations}
    assert counts[v.CHECK_DUPLICATE] == 1
    assert counts[v.CHECK_NON_MONOTONIC] == 1
    assert counts[v.CHECK_OHLC] == 1


def test_raise_if_invalid_aborts():
    df = make_frame([0, 1, 1])
    with pytest.raises(v.DataValidationError):
        v.raise_if_invalid(v.validate_ohlcv(df, label="dup"))


def test_nonpositive_price_negative_volume_and_nulls():
    df = make_frame([0, 1, 2], low=[99.0, 0.0, 99.0], volume=[10.0, -1.0, None])
    checks = v.validate_ohlcv(df).checks()
    assert v.CHECK_PRICE_POSITIVE in checks
    assert v.CHECK_VOLUME in checks
    assert v.CHECK_NULLS in checks


def test_naive_timestamps_rejected():
    df = make_frame([0, 1, 2]).with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    assert v.CHECK_TZ in v.validate_ohlcv(df).checks()


def test_gap_report_lists_every_missing_bar_and_fills_nothing():
    # bars at minutes 0,1,2, then 6 (3 missing: 3,4,5), then 7, then 10 (2 missing: 8,9)
    df = make_frame([0, 1, 2, 6, 7, 10])
    assert v.validate_ohlcv(df).ok

    report = v.build_gap_report(df)
    missing = [t for t in report["missing_timestamp"].to_list()]
    assert missing == [T0 + k * MINUTE for k in (3, 4, 5, 8, 9)]
    assert report["gap_id"].to_list() == [0, 0, 0, 1, 1]
    assert report["prev_timestamp"].to_list()[0] == T0 + 2 * MINUTE
    assert report["next_timestamp"].to_list()[0] == T0 + 6 * MINUTE

    gaps = v.find_gaps(df)
    assert gaps["gap_missing_bars"].to_list() == [3, 2]

    flagged = v.flag_after_gap(df)
    assert flagged.height == df.height                       # no rows added
    assert flagged[cfg.AFTER_GAP_COLUMN].to_list() == [False, False, False, True, False, True]


def test_no_gaps_gives_empty_report():
    df = make_frame(range(5))
    assert v.build_gap_report(df).height == 0
    assert not v.flag_after_gap(df)[cfg.AFTER_GAP_COLUMN].any()
