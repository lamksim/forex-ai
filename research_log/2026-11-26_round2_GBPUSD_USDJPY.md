# Round 2: all pairs, MT5 data, and a cost stress test (2026-11-25/26)

Run on Lam's laptop. Raw `experiments.jsonl` (1,350 lines) and all reports are on the
laptop in `Documents\forex-ai\research_log\`; this is a summary. Holdout untouched.

## Data
- Dukascopy throttled the laptop to ~3 files in 9 hours, so GBPUSD/USDJPY came from
  MetaTrader 5 (MetaQuotes-Demo, demo account), H1, 2019-01-02 04:00 to 2025-10-31 19:00 UTC
  (42,506 bars per pair; the broker has no bars over New Year 2019).
- MT5 bar times are server time = New York + 7 h. A fixed offset put summer bars 1 h late,
  and the request window was shifted by the PC's UTC+8 zone. Both fixed in `load_mt5`
  (commit 96d65d1); the new export matches the hand-corrected data bar for bar, and EURUSD
  matches Dukascopy (median close difference 0.10-0.15 pips, hourly-return correlation 0.90).
- MetaQuotes-Demo spreads are unrealistically low (median 0.1-0.2 pips; zero on about half
  of EURUSD bars versus 0.3 at Dukascopy), so the final runs use `--min-spread 0.3`.

## New family on EURUSD (Dukascopy data)
`rsi_reversion_ex_overlap` (flat 12:00-16:00 UTC): Sharpe -0.61, return -5.56%, DSR 0.000 -> reject.
The idea came from EURUSD's own results and still failed there.

## GBPUSD and USDJPY, 0.3-pip spread floor (N = 34 configs per pair)
| Pair | Family | OOS Sharpe | Return | Max DD | Trades | PF | DSR | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| USDJPY | ma_cross | +0.90 | +19.83% | -5.34% | 482 | 1.32 | 0.247 | reject |
| USDJPY | donchian | +0.66 | +14.03% | -7.98% | 452 | 1.22 | 0.116 | reject |
| USDJPY | rsi_reversion | -1.63 | -26.30% | -26.87% | 775 | | 0.000 | reject |
| USDJPY | rsi_reversion_ex_overlap | -1.44 | -15.13% | -16.02% | 807 | | 0.000 | reject |
| GBPUSD | rsi_reversion | +0.63 | +7.36% | -4.17% | 472 | 1.13 | 0.422 | reject |
| GBPUSD | rsi_reversion_ex_overlap | +0.50 | +4.01% | -2.48% | 499 | 1.11 | 0.317 | reject |
| GBPUSD | ma_cross | -0.76 | -13.74% | -14.96% | 458 | | 0.001 | reject |
| GBPUSD | donchian | -0.94 | -16.72% | -17.19% | 330 | | 0.000 | reject |

Without the spread floor the near-misses were slightly better (USDJPY ma_cross 0.97,
GBPUSD ex_overlap 0.78), and DSRs looked higher only because the first families were
judged against fewer logged trials (N=16/22 instead of 34).

## What it says
- **USDJPY trend following is the only result that survives costs** (ma_cross Sharpe 0.90,
  5 of 7 test windows positive), but about 60% of its profit is the 2022 yen move, its
  latest window (2024H1) lost and halted, and its DSR (0.25) is far from 0.95.
- **GBPUSD RSI reversion is small and fragile** (Sharpe 0.63, PF 1.13).
- **The overlap filter does not generalise**: worse than plain RSI on GBPUSD at realistic
  costs, and both lose on USDJPY.
- Nothing is close to the gates. No strategy goes to the holdout.
