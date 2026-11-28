# Round 3: looking for a second strategy that diversifies USDJPY trend (2026-11-28)

Run in the cloud on copies of the laptop's MT5 H1 files (`*_H1_MT5v2_2019-01-01_2025-10-31`),
research period 2019-01-02 .. 2024-06-18, walk-forward 24 months train / 6 months test
(OOS 2021-01 .. 2024-06). Holdout (2024-06-19 onward) untouched. Trials were appended to a
copy of the laptop's `experiments.jsonl` (2,424 lines), so the deflated Sharpe counts every
config ever tried. The 1,418 new log lines are in the project files
(`forex-ai/round3_experiments_new_lines.jsonl`) for appending to the laptop log.

Costs: commission 3.5 USD per 100k per side, 0.2 pip slippage per fill, spread floor 0.3 pips
on the USD pairs, 0.5 EURGBP, 0.7 EURJPY, 1.0 GBPJPY (on top of the data's own spread).

## New: crosses built from the USD pairs
`forexlab cross` builds EURGBP, EURJPY and GBPJPY from the two legs (opens/closes exact, bid/ask
from the worse side of each leg, highs/lows as bounds). Both approximations are against the
strategy (wider spread, more stop-outs). Confirm anything on real broker cross data.

## Results (walk-forward OOS Sharpe after costs; DSR in brackets)
| Family | USDJPY | EURUSD | GBPUSD | EURGBP | EURJPY | GBPJPY |
| --- | --- | --- | --- | --- | --- | --- |
| **gotobi** | **+1.11** (0.27) | +0.08 | -0.13 | -2.48 | **+1.82** (0.93) | **+1.34** (0.84) |
| rsi_reversion_ranging | -1.11 | -0.07 | -0.09 | -1.65 | -0.99 | -0.54 |
| zscore_reversion | -1.03 | +0.06 | +0.63 (0.31) | -1.25 | -0.59 | -1.37 |
| london_breakout | -0.21 | -0.93 | -0.65 | -3.10 | -1.72 | -2.18 |

DSR on the crosses uses N=4 because they are new symbols; on the USD pairs N=48-68.
Nothing passes all gates. Range-only reversion, z-score reversion and the London breakout
are clear rejects. EURGBP mean reversion, often called range-bound, lost after costs.

## The candidate: gotobi (Tokyo fix calendar effect)
Rule: on gotobi days (5th, 10th, 15th, 20th, 25th, 30th, month end; weekend dates roll back to
Friday) go long the yen pair at 07:00 JST (22:00 UTC) and exit at 09:00 JST (00:00 UTC),
before the 9:55 fix. Stop 3 x H1 ATR (almost never hit). About 6 trades a month.
Source: Ito & Yamada (2017) and the gotobi literature: importers buy dollars into the fix.

USDJPY walk-forward: Sharpe 1.11, 266 trades, win rate 56%, profit factor 1.46, max DD -0.54%.
The same config (07->09 JST) was chosen in all 7 folds; 5 of 7 test windows positive.

Why it is believable:
- **It only works where the theory says.** Positive on the three yen pairs, zero on EURUSD /
  GBPUSD, negative (costs) on EURGBP.
- **Placebo:** the same 07-09 JST window on non-gotobi weekdays: USDJPY Sharpe -1.17,
  EURJPY +0.02, GBPJPY +0.05. The profit comes from the calendar, not the hour.
- **Diversifies:** daily-return correlation with the live USDJPY ma_cross_trending
  (2021-01..2024-06) is **0.01**. It is in the market ~2 hours on ~6 days a month.
- Positive every calendar year 2019-2024 on USDJPY and EURJPY (fixed config, no fitting).

Why it does not pass yet, and the risks:
- **Small edge per trade, cost sensitive.** USDJPY nets ~2 pips per trade after spread and
  slippage. Fixed-config Sharpe by spread floor: 0.3 -> 0.89, 0.6 -> 0.72, 1.0 -> 0.43,
  1.5 -> 0.03. EURJPY holds up better (1.76 -> 1.33), GBPJPY (1.25 -> 1.13), but on synthetic data.
- **Entry sits at the New York rollover in winter** (22:00 UTC = 17:00 New York), where real
  spreads spike for a few minutes. Live, the entry should wait until the spread is normal
  (e.g. 22:05-22:10 UTC). H1 bars cannot test that; M5/M1 data around 22:00 UTC can.
- Most of the profit is in the first hour (07-08 JST). Windows starting at 08:00 JST, which the
  literature emphasises, do not pay costs on their own.
- DSR 0.27 on USDJPY (N=48): the trial count from earlier families makes the bar high.
- Returns are small at 0.25% risk with 3-ATR stops (USDJPY +2.1% over 3.5 years; yen basket
  +5.1%, Sharpe 1.41 at a 0.5 pip floor). Sizing is a separate Risk Manager decision.
- Japanese public holidays are not modelled (a gotobi on a holiday should roll back a day).

## Next steps (in order)
1. Export real EURJPY and GBPJPY H1 data and M5 USDJPY/EURJPY around 21:00-01:00 UTC from MT5,
   or better a real broker's demo with realistic spreads; rerun gotobi on real crosses and with
   a 22:10 UTC entry.
2. If it still holds, take the one holdout look for the yen basket (pre-register the config:
   07->09 JST, stop 3 ATR, USDJPY+EURJPY+GBPJPY).
3. Only then consider a demo forward test next to the USDJPY trend engine (different magic number).

## Batch 2 (same day, same data and costs): four more ideas, all rejected
| Family | USDJPY | EURUSD | GBPUSD | EURGBP | EURJPY | GBPJPY |
| --- | --- | --- | --- | --- | --- | --- |
| home_hours_drift (Ranaldo: currencies weaken in home hours) | +0.08 | -0.31 | -0.60 | -2.98 | -0.90 | -0.84 |
| asia_reversion (z-score fade, 22-06 UTC only) | -2.28 | -0.56 | -1.11 | -1.85 | -2.04 | -1.96 |
| daily_reversal (fade > 1.5-2 daily ATR days) | -0.31 | -0.31 | -1.03 | no trades | -0.25 | +0.10 |
| gotobi_fix_reversal (short yen 10:00 JST to 12/15 JST) | -0.31 | -0.33 | -0.47 | -1.89 | -0.15 | -0.24 |

The home-hours drift is real in gross terms but about 1 pip a day, less than one round trip
of costs. Asia-session fading pays the wide overnight spread on every trade. Big-day fades
fire too rarely (14-47 trades) to judge and do not make money. The yen does not reliably
recover after the gotobi fix. Trials per USD pair are now 80 and per cross 36, which raises
the bar for every later idea; better data (real crosses, more pairs, realistic spreads) is
now worth more than more ideas on the same three pairs.
