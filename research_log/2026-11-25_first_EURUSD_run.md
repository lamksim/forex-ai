# First real-data research run: EURUSD H1, 2019-01 to 2025-10

Run on Lam's laptop on 2026-11-25 with commit 67b82a3. The raw `experiments.jsonl`
(234 lines) and the three full reports are on that laptop in
`Documents\forex-ai\research_log\`. This file is a summary copied from them.
The holdout (from 2024-06-19) was **not** touched.

## Data
- Dukascopy hourly bid/ask candles, 164 monthly files, 4 h 17 min to download
  because of throttling (HTTP 503 and timeouts, all recovered).
- 42,638 H1 bars, 2019-01-01 22:00 to 2025-10-31 20:00 UTC; ask > bid on every bar.
- Spread: median 0.30 pips, p95 1.50, max 19.9. Median 0.3 pips from 00 to 19 UTC,
  widening to 1.1 (20h) and 1.7 (21h) at the daily rollover.
- Hourly files checked against the minute data: the 2024-01-02 00:00 bar opens at
  1.10366 in both.

## Results (walk-forward, out of sample, after costs)
| Family | OOS Sharpe | Return | Max DD | Trades | Profit factor | Gross before costs | Costs | DSR (N) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ma_cross | -0.68 | -12.97% | -17.11% | 423 | 0.80 | -$1,038 | $423 | 0.011 (16) | reject |
| donchian | -0.70 | -13.84% | -17.18% | 411 | 0.79 | -$1,170 | $388 | 0.007 (22) | reject |
| rsi_reversion | -0.08 | -1.37% | -10.29% | 547 | 0.99 | +$370 | $674 | 0.033 (30) | reject |

## What it says
- **The trend families have no edge, even before costs.** Their in-sample Sharpe fell from
  1.3-1.6 (2019-2021) to about 0 by 2023, and 5 of 7 test windows hit a risk halt.
- **RSI reversion has a small gross edge (+$370) that costs more than wipe out ($674).**
  Its losses concentrate in the London/New York overlap (-$536); other sessions were positive.
  That is a candidate Phase 2 hypothesis (session filter), and it will count as new trials.
- Max drawdown exceeds 5% on the stitched record because each 6-month test window starts
  with a fresh $10k account and its own 5% halt; the halts did fire inside the windows.
