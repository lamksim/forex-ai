# Execution rules (backtester and, later, live engine)

1. A signal decided at bar *t* close executes at bar *t+1* open. No same-bar execution.
2. Longs buy at ask, sell at bid. Shorts sell at bid, buy at ask.
3. Slippage: `slippage_pips` (default 0.2) against us on every fill.
4. Commission: `commission_per_100k_side` (default $3.5, i.e. $7 round turn) on notional.
5. Every position has a stop at entry ± `stop_atr × ATR(14)`. Stops are checked intrabar on the
   side we would exit; if both the stop and an exit signal apply on one bar, the stop wins.
6. A gap through the stop fills at the open (loss can exceed planned risk).
7. Risk limits are checked on bar-close equity. A breach flattens at the close.
8. One position per strategy. Reversal = exit then entry on the same open.
9. Open positions at the end of data are closed at the last close and marked `end_of_data`.
10. The execution engine is deterministic. No LLM call is ever on the order path.
