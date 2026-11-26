"""Performance metrics (specs/performance_metrics.md) and the deflated Sharpe ratio."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649
_N = NormalDist()


def daily_returns(equity: pd.Series) -> pd.Series:
    """Daily close-to-close returns on days the market was open."""
    daily = equity.resample("1D").last().dropna()
    return daily.pct_change().dropna()


def sharpe(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())


def max_consecutive_losses(pnl: pd.Series) -> int:
    best = run = 0
    for x in pnl:
        run = run + 1 if x < 0 else 0
        best = max(best, run)
    return best


def summarize(equity: pd.Series, trades: pd.DataFrame, initial_equity: float) -> dict:
    r = daily_returns(equity)
    out = {
        "start": str(equity.index[0]) if len(equity) else None,
        "end": str(equity.index[-1]) if len(equity) else None,
        "net_profit_usd": float(equity.iloc[-1] - initial_equity) if len(equity) else 0.0,
        "total_return": float(equity.iloc[-1] / initial_equity - 1) if len(equity) else 0.0,
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(equity),
        "avg_daily_return": float(r.mean()) if len(r) else 0.0,
        "days": int(len(r)),
        "trades": int(len(trades)),
    }
    if len(trades):
        pnl = trades["pnl_usd"]
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        out.update({
            "win_rate": float((pnl > 0).mean()),
            "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else float("inf"),
            "avg_win_usd": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss_usd": float(losses.mean()) if len(losses) else 0.0,
            "max_consecutive_losses": max_consecutive_losses(pnl),
            "total_costs_usd": float(trades["commission_usd"].sum() + trades["spread_cost_usd"].sum()
                                     + trades.get("slippage_usd", pd.Series(dtype=float)).sum()),
            "swap_usd": float(trades.get("swap_usd", pd.Series(dtype=float)).sum()),
            "gross_before_costs_usd": float(trades["gross_mid_pnl_usd"].sum()),
            # Sum of trade P&L. net_profit_usd above comes from the equity curve, which for
            # walk-forward results compounds separate $10k test windows, so the two differ slightly.
            "net_trades_usd": float(pnl.sum()),
            "by_session": {k: {"trades": int(len(g)), "pnl_usd": float(g["pnl_usd"].sum())}
                           for k, g in trades.groupby("session")},
            "exit_reasons": {k: int(v) for k, v in trades["exit_reason"].value_counts().items()},
        })
    else:
        out.update({"win_rate": None, "profit_factor": None, "avg_win_usd": None,
                    "avg_loss_usd": None, "max_consecutive_losses": 0, "total_costs_usd": 0.0})
    return out


def probabilistic_sharpe(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Probability that the true per-period Sharpe exceeds `benchmark_sr` (Bailey & Lopez de Prado 2012).
    Both Sharpe ratios are per period (not annualized)."""
    r = returns.dropna().to_numpy()
    t = len(r)
    if t < 3 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std(ddof=0)
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return 0.0
    return float(_N.cdf((sr - benchmark_sr) * math.sqrt(t - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, trial_sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among n_trials skill-less strategies
    (the False Strategy Theorem, Bailey & Lopez de Prado 2014)."""
    if n_trials <= 1 or trial_sr_variance <= 0:
        return 0.0
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(trial_sr_variance) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sharpes_annual: list[float]) -> dict:
    """Deflated Sharpe ratio: the probabilistic Sharpe measured against the Sharpe
    you would expect from the best of `n_trials` worthless strategies.

    trial_sharpes_annual are the annualized Sharpes of every configuration tried,
    taken from the experiment log, so every test ever run counts against the result.
    """
    per_period = [s / math.sqrt(TRADING_DAYS) for s in trial_sharpes_annual]
    var = float(np.var(per_period, ddof=1)) if len(per_period) > 1 else 0.0
    sr0 = expected_max_sharpe(n_trials, var)
    return {
        "n_trials": n_trials,
        "sr0_annual": sr0 * math.sqrt(TRADING_DAYS),
        "dsr": probabilistic_sharpe(returns, sr0),
    }
