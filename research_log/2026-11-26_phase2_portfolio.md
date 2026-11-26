# Phase 2 run: new families and the 3-pair portfolio (2026-11-26)

Run on Lam's laptop with commit 004d5ca, MT5 H1 data 2019-01-02..2025-10-31, 0.3-pip spread
floor, MetaQuotes-Demo swaps. Raw log (2,424 lines) and the 18 per-pair reports are on the
laptop. Holdout untouched.

## Portfolio results (sum of the three pairs, 0.25% risk per trade per pair)
| Family | Portfolio Sharpe | Return | Max DD | Trades | DSR (N) | EURUSD | GBPUSD | USDJPY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ts_momentum | +0.27 | +2.69% | -5.82% | 193 | 0.013 (44) | +0.28 | -0.61 | +0.66 |
| ma_cross_trending | +0.18 | +2.84% | -5.74% | 1336 | 0.012 (38) | -0.09 | -0.82 | +0.79 |
| ma_cross | +0.06 | +0.47% | -12.99% | 1483 | 0.007 (34) | -0.06 | -0.79 | +0.89 |
| donchian | -0.59 | -22.30% | -31.17% | 1164 | 0.000 | -0.89 | -0.96 | +0.65 |
| rsi_reversion_ex_overlap | -0.65 | -11.58% | -18.87% | 1994 | 0.000 | -0.11 | +0.73 | -1.45 |
| rsi_reversion | -0.94 | -23.72% | -27.45% | 1853 | 0.000 | -0.22 | +0.61 | -1.62 |

All rejected. Every trend family makes money only on USDJPY and loses on GBPUSD; the RSI
families do the reverse. Combining pairs therefore cancels out rather than diversifies.

## Notes on the new families (USDJPY, their best pair)
- **ma_cross_trending**: Sharpe 0.79, max DD -2.85%, 5 of 7 test windows positive. The trend
  filter halves plain ma_cross's drawdown but also cuts its Sharpe (0.90 -> 0.79). DSR 0.155.
- **ts_momentum**: Sharpe 0.66 on only 64 trades (2-15 per window): too few trades to judge.

## Caveats found in this run
- **Swaps are not realistic yet.** MetaQuotes-Demo charges tiny negative swaps on both sides
  (USDJPY long -0.01 pips/night), while a real broker pays a clear positive carry on USDJPY
  longs. A real broker's demo server is needed for meaningful swap costs.
- **Cost breakdown in reports was wrong** (fixed after this run): "gross before costs" still
  had commission deducted and "total costs" left out slippage, so gross - costs did not equal
  net. Trade P&L, equity, Sharpe and DSR were not affected; only the breakdown lines were.
- Portfolio correlations were computed but not saved; now written to a PORTFOLIO_*.md report
  and the log.
- Daily/weekly loss limits are still per pair, not on the combined account.

## Where this leaves the research
After 3 pairs, 6 families and 44 configurations per pair, nothing passes. The one recurring
pattern is trend following on USDJPY, which is one pair in one regime (2021-24 yen weakness).
