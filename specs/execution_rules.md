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
10. Overnight swap: positions open through the 21:00 UTC bar pay or earn `swap_long_pips` /
    `swap_short_pips` per night (triple on Wednesday). Values come from the broker's current
    MT5 swap table, so history is approximated with today's rates.
11. The execution engine is deterministic. No LLM call is ever on the order path.

## Live demo engine (`forexlab live`, Phase 4)

12. Demo accounts only. `require_demo` checks MT5's `trade_mode` on every cycle and the engine
    refuses to start (and stops) on any other account type.
13. One cycle per closed H1 bar, about 20 s after each UTC hour. The signal uses only closed
    bars (`copy_rates_from_pos` from position 1), so live decisions match rule 1.
14. Orders are market orders (deviation 20 points) and every entry carries a server-side
    stop-loss at entry ± the strategy's stop distance, so the position stays protected if the
    laptop sleeps or the engine crashes.
15. The Risk Manager sizes each order from real account equity (rounded down to the broker's
    volume step; skipped if below the minimum lot) and halts on the daily, weekly and max
    drawdown limits. Its state is saved to `trading_logs/<symbol>_<magic>_state.json`, so a
    restart does not reset a halt.
16. The engine only touches positions with its magic number (default 20261126).
17. Kill switch: create a file named `STOP` in the log folder. The next cycle closes our
    position and the engine exits.
18. Every cycle is appended to `trading_logs/<symbol>_<YYYY-MM>.jsonl` (signal, equity, risk
    state, every order request result).
