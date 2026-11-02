"""Phase 1 tests: zip parsing and timestamp-unit handling (synthetic zips, no network)."""
import io
import zipfile
from datetime import date, datetime, timezone

import polars as pl

import config as cfg
import data_loader as dl

HEADER = ",".join(cfg.KLINE_COLUMNS)


def make_zip(tmp_path, rows, header=True, name="BTCUSDT-1m-2024-01.zip"):
    lines = ([HEADER] if header else []) + rows
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name.replace(".zip", ".csv"), "\n".join(lines) + "\n")
    return path


def row(open_time_ms, price=100.0):
    return f"{open_time_ms},{price},{price+1},{price-1},{price},5.0,{open_time_ms+59999},500.0,10,2.0,200.0,0"


MS_2024_01_01 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
STEP_MS = cfg.EXPECTED_BAR_SECONDS * cfg.MS_PER_SECOND


def test_parses_with_and_without_header(tmp_path):
    rows = [row(MS_2024_01_01), row(MS_2024_01_01 + STEP_MS)]
    for header in (True, False):
        df = dl.read_month_zip(make_zip(tmp_path, rows, header=header, name=f"h{header}-2024-01.zip"), "t")
        assert df.columns == cfg.OUTPUT_COLUMNS
        assert df.height == 2
        assert df.schema["timestamp"] == pl.Datetime(cfg.TIME_UNIT, cfg.TIMEZONE)
        assert df["timestamp"][0] == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_microsecond_timestamps_are_converted(tmp_path):
    rows = [row(MS_2024_01_01 * cfg.MICROS_PER_MILLI)]
    df = dl.read_month_zip(make_zip(tmp_path, rows), "t")
    assert df["timestamp"][0] == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_month_helpers():
    months = dl.month_range(cfg.DATA_START, cfg.DATA_END)
    assert len(months) == 36
    assert months[0] == date(2023, 11, 1) and months[-1] == date(2026, 10, 1)
    assert dl.month_range(date(2023, 12, 1), date(2024, 1, 1)) == [date(2023, 12, 1), date(2024, 1, 1)]
    assert dl.zip_url(date(2024, 1, 1)) == (
        "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip"
    )
    assert dl.expected_rows(date(2024, 2, 1)) == 29 * 1440      # leap year
