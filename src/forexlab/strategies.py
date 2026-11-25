"""Strategy families.

A strategy turns bars into two series aligned to the bar index:
    signal  in {-1, 0, +1}: the position wanted, decided at the bar's close
    stop    stop distance in price units (> 0), used for stop-loss and sizing
The backtester acts on a signal at the NEXT bar's open, so a strategy can use
the current bar's close without looking ahead. Every indicator here is causal
(uses only past and current bars); tests/test_no_lookahead.py checks this.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data import mid


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = mid(df, "high"), mid(df, "low"), mid(df, "close")
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def ma_cross(df: pd.DataFrame, fast: int, slow: int, stop_atr: float) -> tuple[pd.Series, pd.Series]:
    """Trend following: long when the fast moving average is above the slow one."""
    c = mid(df)
    f = c.rolling(fast, min_periods=fast).mean()
    s = c.rolling(slow, min_periods=slow).mean()
    signal = np.sign(f - s).fillna(0)
    return signal.astype(int), stop_atr * atr(df)


def donchian_breakout(df: pd.DataFrame, lookback: int, stop_atr: float) -> tuple[pd.Series, pd.Series]:
    """Trend following: go long on a close above the prior N-bar high, short below the low.
    Holds the position until the opposite breakout (or the stop)."""
    c = mid(df)
    hi = mid(df, "high").rolling(lookback, min_periods=lookback).max().shift(1)
    lo = mid(df, "low").rolling(lookback, min_periods=lookback).min().shift(1)
    raw = pd.Series(np.where(c > hi, 1, np.where(c < lo, -1, np.nan)), index=df.index)
    signal = raw.ffill().fillna(0)
    return signal.astype(int), stop_atr * atr(df)


def rsi_reversion(df: pd.DataFrame, period: int, band: int, stop_atr: float) -> tuple[pd.Series, pd.Series]:
    """Mean reversion: long below (50 - band), short above (50 + band), flat once RSI crosses 50."""
    r = rsi(mid(df), period)
    lo, hi = 50 - band, 50 + band
    raw = pd.Series(np.nan, index=df.index)
    raw[r < lo] = 1
    raw[r > hi] = -1
    # Exit to flat when RSI crosses back through 50.
    cross = (np.sign(r - 50) != np.sign(r.shift(1) - 50)) & r.notna() & r.shift(1).notna()
    raw[cross & raw.isna()] = 0
    signal = raw.ffill().fillna(0)
    return signal.astype(int), stop_atr * atr(df)


@dataclass(frozen=True)
class Family:
    name: str
    fn: Callable[..., tuple[pd.Series, pd.Series]]
    grid: dict[str, list]
    hypothesis: str

    def configs(self) -> list[dict]:
        keys = list(self.grid)
        out = []
        for values in itertools.product(*(self.grid[k] for k in keys)):
            params = dict(zip(keys, values))
            if self.name == "ma_cross" and params["fast"] >= params["slow"]:
                continue
            out.append(params)
        return out


FAMILIES: dict[str, Family] = {
    "ma_cross": Family(
        "ma_cross", ma_cross,
        {"fast": [10, 20, 50], "slow": [50, 100, 200], "stop_atr": [2.0, 3.0]},
        "H1 moving-average crossovers capture persistent trends in major pairs after costs.",
    ),
    "donchian": Family(
        "donchian", donchian_breakout,
        {"lookback": [24, 72, 120], "stop_atr": [2.0, 3.0]},
        "Breakouts from multi-day H1 ranges continue far enough to pay the spread.",
    ),
    "rsi_reversion": Family(
        "rsi_reversion", rsi_reversion,
        {"period": [7, 14], "band": [20, 25], "stop_atr": [1.5, 2.5]},
        "Short-term H1 overextension in majors mean-reverts within a day.",
    ),
}


def generate(family: str, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    return FAMILIES[family].fn(df, **params)
