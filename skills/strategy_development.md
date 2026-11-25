# Skill: Strategy development

**Purpose:** encode a hypothesis as a strategy family.

1. Implement `fn(df, **params) -> (signal, stop)` in `src/forexlab/strategies.py`, using only
   past and current bars.
2. Register it in `FAMILIES` with its hypothesis text and a grid of **at most ~20 configs**.
3. Run `pytest` (the no-lookahead test covers new families automatically).
4. Hand off to `backtesting.md`.
