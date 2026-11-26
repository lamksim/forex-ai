# Constitution

These rules are non-negotiable. No agent, skill, strategy or automated process
may change them. Only Lam can amend this file, by a commit that says why.

1. **Capital preservation comes before profit targets.** The 0.5%-per-day figure is a
   research aspiration, never a quota. A day with no trade and 0% return is a good day
   if no validated opportunity existed.
2. **No strategy goes live without independent validation.** It must pass the research
   gates in `specs/performance_metrics.md`, then its single holdout evaluation, then
   paper trading (Phase 4) for at least 3 months before any real money (Phase 5).
3. **No autonomous changes to live trading rules.** The learning loop may propose new
   strategy versions. A human approves every promotion.
4. **Every decision is logged and explainable.** Every backtest is recorded in
   `research_log/experiments.jsonl`; every holdout look in `research_log/holdout_access.jsonl`;
   every live or paper trade (later phases) in `trading_logs/` with its reason for entry and exit.
5. **No martingale, no averaging down, no increasing size to recover losses.**
6. **The Risk Manager can veto any trade** and is independent of every strategy.
   Strategies cannot read or change risk limits.
7. **The system must be able to stop trading immediately.** Hitting the max drawdown is a
   permanent halt until a human reviews it.
8. **Statistics decide, not narratives.** An LLM may research, explain and propose
   hypotheses. Only out-of-sample evidence, adjusted for how many things were tried,
   decides whether a strategy works.
9. **The holdout is sacred.** Research code never touches it. Each strategy version gets
   one look. Changing a strategy after seeing holdout results makes it a new version,
   and the old look still counts.
10. **Only approved pairs are traded:** EURUSD, GBPUSD, USDJPY. New pairs are added only
    after they pass the same validation.
