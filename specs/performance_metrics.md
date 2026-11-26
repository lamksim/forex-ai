# Performance metrics and validation gates

## Reported for every run (`metrics.summarize`)
Net profit after costs, total return, Sharpe (daily, annualized ×√252), max drawdown,
average daily return, trades, win rate, profit factor, average win vs loss, max consecutive
losses, total costs, gross P&L before costs, P&L by session, exit reasons.
Later phases add live-vs-backtest deviation.

## How a strategy is judged
1. **Walk-forward:** rolling 24-month train / 6-month test windows over the research data.
   In each window the best config by train Sharpe is chosen and then traded on the
   unseen test window. Only stitched test windows count.
2. **Deflated Sharpe ratio (DSR):** the probability that the out-of-sample Sharpe beats
   the best Sharpe expected from *N* worthless strategies, where *N* is every config ever
   tried on that pair (all families, all runs, from the experiment log).
3. **Gates** (all required before a holdout look):

| Gate | Threshold |
| --- | --- |
| Out-of-sample Sharpe | ≥ 1.0 |
| Deflated Sharpe | ≥ 0.95 |
| Trades | ≥ 100 |
| Profit factor | ≥ 1.2 |
| Max drawdown | ≤ 5% |

4. **Holdout:** the most recent 20% of data. One look per strategy version. Passing means
   holdout Sharpe > 0.5 and drawdown ≤ 5%; it is a sanity check, not a second optimisation.
