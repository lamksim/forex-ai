# Orchestrator

Phase 1: a human runs the CLI (`download` → `research` → review report → `holdout`).

Phase 3 plan: an LLM orchestrator may schedule research runs and draft hypotheses, but it
calls the same deterministic CLI, cannot open the holdout without a human, and cannot
change `constitution.md`, `specs/risk_management.md` or `risk.py`.
