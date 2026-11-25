# Risk manager

- Independent of every strategy; strategies cannot read or modify its limits.
- Sizes every trade (0.25% risk) and can veto any trade (returns size 0).
- Enforces daily, weekly and max-drawdown halts (`specs/risk_management.md`).
- Will never be an LLM. Its code changes need human review and a test.
