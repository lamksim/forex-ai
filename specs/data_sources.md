# Data sources

| Source | What | Pros | Limits |
| --- | --- | --- | --- |
| **Dukascopy** (default) | M1 bid **and** ask candles, from ~2003 | Free; real spreads including news widening and rollover | Unofficial endpoint, no SLA; ~2 requests per day of data (a 5-year download is ~3,000 requests and takes a while); one ECN's prices, not your broker's; spread is only observed at M1 resolution |
| **MT5 terminal** (`load_mt5`) | Broker bars with spread at bar open | Matches the broker you will trade with | Windows only; history depth depends on broker; only one spread per bar, so intrabar widening is understated |
| **Synthetic** (`synth`) | Random walk with session volatility and news-hour spread widening | Instant; no edge by construction | For tests and smoke runs only. A strategy that "works" on it is overfit |

## Bar format
UTC index, columns `bid_open..bid_close, ask_open..ask_close, volume`. Research resamples
M1 to H1 by default (`--timeframe`).

## Integrity rules
- Every experiment records the dataset fingerprint (`data.fingerprint`).
- Raw downloads are cached in `data/raw/` and never edited. `data/` is git-ignored.
- Zero-volume bars (holidays, feed gaps) are dropped at download.

## Known gaps (Phase 2)
- Economic calendar (to tag news bars and test news hypotheses).
- Swap is modelled with constant per-pair rates (Phase 2); historical swap rates are not available for free.
