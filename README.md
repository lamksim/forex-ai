# forex-ai: Forex AI research lab

A spec-driven lab that researches FX markets, tests trading hypotheses with honest
statistics, and only later paper-trades and trades. **Phase 1: research engine.**

Read first: [`constitution.md`](constitution.md) (non-negotiable rules) and
[`specs/requirements.md`](specs/requirements.md) (goals and roadmap).

> The 0.5%-per-day figure is a research aspiration, not a target the system is forced to hit.
> The primary objective is risk-adjusted return after costs, out of sample.
> Automated trading systems cannot consistently predict markets; nothing here is financial advice.

## Quick start

```bash
pip install -e ".[dev]"
pytest

# 1. Smoke test on no-edge synthetic data (everything should be REJECTED)
python -m forexlab synth --out data/EURUSD_synthetic_H1.csv.gz
python -m forexlab research --data data/EURUSD_synthetic_H1.csv.gz --log research_log/synthetic.jsonl

# 2. Real data: free Dukascopy M1 bid/ask candles (cached in data/raw, ~2 requests per day of data)
python -m forexlab download --symbol EURUSD --start 2019-01-01 --end 2025-10-31
python -m forexlab research --data data/EURUSD_M1_2019-01-01_2025-10-31.csv.gz

# 2b. Portfolio of all three pairs with realistic costs (MT5 data; swaps in pips per night)
python -m forexlab mt5 --symbol GBPUSD --start 2019-01-01 --end 2025-10-31
python -m forexlab portfolio --data data/EURUSD_*.csv.gz data/GBPUSD_*.csv.gz data/USDJPY_*.csv.gz \
    --min-spread 0.3 --swaps EURUSD=-0.7:0.2 GBPUSD=-0.4:0.0 USDJPY=1.2:-2.5

# 2c. Crosses built from the USD pairs (EURGBP, EURJPY, GBPJPY), then e.g. the gotobi family
python -m forexlab cross --symbol EURJPY --legs data/EURUSD_H1_MT5v2_*.csv.gz data/USDJPY_H1_MT5v2_*.csv.gz
python -m forexlab research --data data/EURJPY_H1_synthetic-cross.csv.gz --symbol EURJPY \
    --min-spread 0.7 --family gotobi

# 3. Only if a family passes every gate: one look at the sealed holdout
python -m forexlab holdout --data data/EURUSD_M1_...csv.gz --family donchian --version donchian-v1
```

## How it avoids fooling itself

| Risk | Guard |
| --- | --- |
| Costs ignored | Trades fill at bid/ask from the data (spread widening included), plus slippage and commission |
| Look-ahead | Signals at bar close execute next open; a test checks every family is causal |
| Overfitting | Walk-forward: parameters picked on 24 months, judged on the next unseen 6 |
| Data snooping | Every config tried is logged in `research_log/experiments.jsonl`; the deflated Sharpe ratio raises the bar as N grows |
| Peeking at the final test | Most recent 20% is a sealed holdout: one look per frozen version, logged |
| Risk creep | Independent Risk Manager: 0.25%/trade, 1% day, 2.5% week, 5% max drawdown (permanent halt) |

## Layout

```
constitution.md        non-negotiable rules
specs/                 requirements, strategy, risk, data, execution, metrics
skills/                procedures: research, strategy development, backtesting, learning
agents/                role definitions (Phase 1 has no autonomous agents by design)
src/forexlab/          data, strategies, risk, backtest, metrics, registry, splits, research, cli
tests/                 pipeline tests
research_log/          experiment log, holdout access log, reports (committed)
strategy_versions/     frozen strategy versions
trading_logs/          paper/live trade logs (Phase 4+)
data/                  downloaded data (git-ignored)
```
