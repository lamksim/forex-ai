# Skill: Backtesting and validation

**Purpose:** decide, with statistics, whether a strategy family has an edge.

```bash
python -m forexlab research --data data/EURUSD_M1_....csv.gz --family donchian
```

1. Read the report in `research_log/reports/`. Look at the verdict, the stitched out-of-sample
   metrics, the deflated Sharpe and N, and whether chosen params are stable across folds
   (they jump around = likely noise).
2. Compare gross P&L before costs with net. If costs eat most of it, the edge is not tradable.
3. If every gate passed: freeze the version and run the holdout once:
   `python -m forexlab holdout --data ... --family donchian --version donchian-v1`.
4. Commit `research_log/` so the record of every trial is permanent.

**Must not:** edit or delete log lines, re-run the holdout, or tweak a strategy after seeing
holdout results and call it the same version.
