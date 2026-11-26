# Risk management

Implemented in `src/forexlab/risk.py` (`RiskLimits`, `RiskManager`). These are initial
research parameters, not proven optima, and research does not tune them.

| Parameter | Value | Behaviour |
| --- | --- | --- |
| Risk per trade | 0.25% of equity | Size = equity × 0.25% ÷ stop distance (in USD) |
| Max daily loss | 1.0% | Flatten; no new trades until next UTC day |
| Max weekly loss | 2.5% | Flatten; no new trades until next ISO week |
| Max drawdown | 5% from peak | Flatten; **permanent halt** until a human reviews |
| Max leverage | 10× notional | Caps size when stops are very tight |
| Open positions | 1 per strategy (portfolio max 3 from Phase 2) | |
| Trade without validated signal | Prohibited | |
| Martingale / averaging down | Prohibited | Size never depends on recent losses |

**On the 0.5%/day aspiration:** with 0.25% risk per trade it needs, on average, two full
winning risk units every day. Hold it up against the 5% drawdown cap: these limits mean
the realistic outcome of a good strategy is closer to 5-20% per year. The research reports
both so the gap stays visible.
