# Requirements

## Goal
An autonomous research lab that researches FX markets, forms trading hypotheses,
tests them with honest statistics, learns from results, and only later trades.

**Primary objective:** risk-adjusted return after all costs, measured out of sample
(target: Sharpe > 1 after costs, with max drawdown under 5%).
**Tracked, not targeted:** average daily return (aspiration 0.5%/day, see constitution rule 1).

## Fixed decisions
| Item | Decision |
| --- | --- |
| Platform | MetaTrader 5 via the `MetaTrader5` Python package (Windows, or Wine) for demo/live; research runs anywhere |
| Capital | $10,000 simulated paper account; no real money before Phase 5 |
| Mode | Historical research and backtesting first, then demo trading |
| Pairs | EURUSD, GBPUSD, USDJPY |
| Research data | Dukascopy M1 bid/ask candles (free) |

## Roadmap
| Phase | Scope | Exit criteria |
| --- | --- | --- |
| **1 Research engine** (this repo now) | Data, cost-aware backtester, walk-forward research, experiment log, holdout | Reproducible backtests; tests pass; noise is rejected |
| 2 Strategy laboratory | More hypothesis families, regime/session filters, multi-pair, news-aware spreads | A strategy passes gates on 2+ pairs |
| 3 Learning engine | LLM-assisted hypothesis generation and trade diagnosis, feeding the same gates | Improvements validated out of sample |
| 4 Demo trading | MT5 demo execution engine, live-vs-backtest tracking | ≥3 months of paper results within backtest expectations |
| 5 Live | Small capital, strict controls | Human approval |

## Phase 1 non-goals
No live or demo order placement. No LLM in the loop yet. No database (JSONL + CSV are enough).
