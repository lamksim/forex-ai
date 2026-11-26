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


def rsi_reversion_ex_overlap(df: pd.DataFrame, period: int, band: int, stop_atr: float) -> tuple[pd.Series, pd.Series]:
    """RSI reversion, but flat during the London/New York overlap (12:00-16:00 UTC).

    A signal at bar t trades at bar t+1's open, so forcing the signal to 0 on bars
    starting 11:00-14:00 keeps the position flat from the 12:00 open to the 16:00 open.
    """
    signal, stop = rsi_reversion(df, period, band, stop_atr)
    hours = df.index.hour
    signal = signal.where(~((hours >= 11) & (hours <= 14)), 0)
    return signal.astype(int), stop


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Kaufman efficiency ratio: net move / total path over `window` bars (0 = noise, 1 = straight line)."""
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return net / path.replace(0, np.nan)


def ma_cross_trending(df: pd.DataFrame, er_window: int, er_min: float, stop_atr: float,
                      fast: int = 20, slow: int = 50) -> tuple[pd.Series, pd.Series]:
    """MA crossover that only holds a position while the market is trending
    (efficiency ratio above er_min); flat in choppy markets."""
    signal, stop = ma_cross(df, fast, slow, stop_atr)
    er = efficiency_ratio(mid(df), er_window)
    signal = signal.where(er >= er_min, 0)
    return signal.astype(int), stop


def daily_atr(df: pd.DataFrame, days: int = 14) -> pd.Series:
    """ATR of daily bars, known only from the day after (no look-ahead), on the H1 index."""
    d = pd.DataFrame({"h": mid(df, "high"), "l": mid(df, "low"), "c": mid(df, "close")})
    daily = d.resample("1D").agg({"h": "max", "l": "min", "c": "last"}).dropna()
    prev = daily["c"].shift(1)
    tr = pd.concat([daily["h"] - daily["l"], (daily["h"] - prev).abs(), (daily["l"] - prev).abs()], axis=1).max(axis=1)
    a = tr.rolling(days, min_periods=days).mean().shift(1)  # yesterday's value
    return a.reindex(df.index.normalize()).set_axis(df.index)


def ts_momentum(df: pd.DataFrame, lookback_days: int, stop_atr: float) -> tuple[pd.Series, pd.Series]:
    """Time-series momentum: hold the direction of the last `lookback_days` return,
    re-decided once a day at the 20:00 UTC bar close (before rollover). Stop on daily ATR."""
    c = mid(df)
    past = c.shift(lookback_days * 24)
    raw = pd.Series(np.nan, index=df.index)
    at_decision = df.index.hour == 20
    raw[at_decision] = np.sign(c - past)[at_decision]
    signal = raw.ffill().fillna(0)
    return signal.astype(int), stop_atr * daily_atr(df)


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
    "rsi_reversion_ex_overlap": Family(
        "rsi_reversion_ex_overlap", rsi_reversion_ex_overlap,
        {"period": [7, 14], "band": [20, 25], "stop_atr": [2.5]},
        "RSI mean reversion works outside the London/New York overlap, when news-driven flow "
        "is weaker. (Suggested by the first EURUSD run's session breakdown, so EURUSD results "
        "are partly in-sample; GBPUSD and USDJPY are the cleaner test.)",
    ),
    "ma_cross_trending": Family(
        "ma_cross_trending", ma_cross_trending,
        {"er_window": [48, 120], "er_min": [0.15, 0.3], "stop_atr": [3.0]},
        "The H1 20/50 crossover loses in choppy markets; holding it only when the efficiency "
        "ratio shows a trend removes most losing trades. (20/50 was the most-chosen setting in "
        "round 2, so this is partly informed by earlier results.)",
    ),
    "ts_momentum": Family(
        "ts_momentum", ts_momentum,
        {"lookback_days": [20, 60, 120], "stop_atr": [2.0, 4.0]},
        "Currencies trend over weeks to months (time-series momentum, documented in academic "
        "FX research); a daily decision with wide stops keeps trading costs low.",
    ),
}


def generate(family: str, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    return FAMILIES[family].fn(df, **params)
