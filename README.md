# Session Liquidity Sweep - BTCUSDT perpetual research engine

Research and backtesting engine for a session liquidity-sweep strategy on Binance USD-M BTCUSDT
(1m data -> 5m signals -> event-driven backtest -> Streamlit report). All times are UTC.
Sessions are fixed UTC hours and are **not DST-aware** (see `config.py`).

## Setup (Windows 11, PowerShell or cmd, from the project folder)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run it end to end

| Phase | Command | Writes |
|---|---|---|
| 1 Data | `python data_loader.py` | `data/btc_1m.parquet`, `data/gaps_report.csv`, `data/load_summary.txt` |
| 2 Signals | `python signal_engine.py` | `data/signals.parquet`, `data/signals_meta.json` |
| 2 Chart check | `python scripts\plot_signals.py` | `data/signal_charts.html` (open in a browser and eyeball the sweeps) |
| 3 Backtest | `python backtester.py` | `data/trades.parquet`, `data/trades.csv` |
| 3 Hand check | `python scripts\verify_trade.py --random 5` | prints an independent audit of 5 trades (exit code 1 on any mismatch) |
| 4 Reconcile | `python scripts\reconcile.py` | prints headline KPIs recomputed from `trades.csv` |
| 4 Dashboard | `streamlit run app.py` | interactive report at http://localhost:8501 |

Re-run from the earliest phase whose settings you changed (for example, changing `SWEEP_MODE` means
re-running `signal_engine.py` and everything after it). Raw downloads are cached in `data/raw_zips/`.
Logs go to the console and `logs/pipeline.log`.

## Checking the dashboard against the trade log

1. `python scripts\reconcile.py`
2. In the dashboard, open **Key metrics -> Exact values for scripts/reconcile.py**.
3. The eight lines must be identical (Win Rate, Profit Factor, Net PnL, Max Drawdown, Total Trades,
   Expectancy, Average Win R, Average Loss R). `reconcile.py` also checks that `trades.csv` and
   `trades.parquet` describe the same trades and prints `MISMATCH` (exit code 1) if they do not.

## Tests

```
python -m pytest -v
```

Covers data validation and gap detection, 5m resampling, sessions, sweep/confirmation logic and look-ahead
(causality) checks, position sizing, costs, funding, SL/TP collisions, the backtester's visibility guard,
dashboard maths, reconcile agreement, and headless runs of the Streamlit app.

## Settings that change results (all in `config.py`)

`SWEEP_MODE`, `CONFIRM_ON_SWEEP_BAR`, `R_MULTIPLE`, `RISK_USD`, `MAX_LEVERAGE`, `STOP_BUFFER_*`,
`TAKER_FEE_RATE`, `SLIPPAGE_BPS`, `FORCE_EXIT_HOUR`, `DATA_START` / `DATA_END`.

Optional funding history: put `data/funding_rates.parquet` with columns `timestamp` (UTC settlement time)
and `funding_rate` (fraction per 8h, e.g. `0.0001`). Without it, funding is estimated at 0.01% per 8h and
every trade is flagged `funding_estimated = True`.
