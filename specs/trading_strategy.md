# Trading strategy specification

A strategy is a **family** (a hypothesis + a function + a parameter grid) in
`src/forexlab/strategies.py`. Each returns a position signal in {-1, 0, +1} and a stop distance.

| Family | Hypothesis | Grid size |
| --- | --- | --- |
| `ma_cross` | H1 MA crossovers capture persistent trends after costs | 16 |
| `donchian` | Breakouts from multi-day ranges run far enough to pay the spread | 6 |
| `rsi_reversion` | Short-term H1 overextension mean-reverts within a day | 8 |
| `rsi_reversion_ex_overlap` | RSI reversion works outside the London/NY overlap (round 1 idea) | 4 |
| `ma_cross_trending` | 20/50 crossover only while the efficiency ratio shows a trend (Phase 2) | 4 |
| `ts_momentum` | Weeks-to-months time-series momentum, decided daily, daily-ATR stops (Phase 2) | 6 |

The first three are baselines to prove the pipeline, not expected winners.

## Adding a family
1. Write the hypothesis as one falsifiable sentence.
2. Implement it causally (only past/current bars); add it to `FAMILIES`.
3. Keep the grid small. Every config counts as a trial and raises the bar for everything.
4. `tests/test_pipeline.py::test_no_lookahead` runs on every family automatically.

## Versions
A strategy that passes the gates is frozen as `strategy_versions/<family>-vN.json`
(family, params, data fingerprint, run id) before its holdout look.
