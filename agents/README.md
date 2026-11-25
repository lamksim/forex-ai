# Agents

Phase 1 deliberately has **no autonomous agents**: one deterministic Python pipeline plus
the Risk Manager. Roles are documented here so later phases can split them out only when a
real need appears.

| Role | Phase 1 implementation | Becomes an agent in |
| --- | --- | --- |
| Orchestrator | `forexlab` CLI, run by a human | Phase 3 |
| Researcher / Strategist | Human + `skills/forex_research.md`, `skills/strategy_development.md` | Phase 3 (LLM proposes hypotheses) |
| Quant analyst | `research.py`, `metrics.py` (deterministic) | Stays deterministic |
| Risk manager | `risk.py` (deterministic, independent) | Stays deterministic, see `risk_manager.md` |
| Execution | `backtest.py` | Phase 4 MT5 demo engine, deterministic |
