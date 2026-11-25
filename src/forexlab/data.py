"""Market data: Dukascopy download, CSV/MT5 loading, synthetic data, resampling.

All bar data in the lab uses one shape: a UTC DatetimeIndex and the columns
    bid_open bid_high bid_low bid_close ask_open ask_high ask_low ask_close volume
Keeping both bid and ask means the spread actually paid (including widening
around news) comes from the data rather than from an assumption.
"""

from __future__ import annotations

import hashlib
import lzma
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import instruments

BAR_COLUMNS = [
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close", "volume",
]

DUKASCOPY_URL = (
    "https://datafeed.dukascopy.com/datafeed/{symbol}/{y:04d}/{m:02d}/{d:02d}/"
    "{side}_candles_min_1.bi5"
)
# Each record in a Dukascopy M1 candle file is 24 bytes, big-endian:
# seconds-from-midnight, open, close, low, high (int32 scaled by point) and volume (float32).
_CANDLE_DTYPE = np.dtype([
    ("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4"),
])


# --------------------------------------------------------------------------- Dukascopy

def parse_candles_bi5(raw: bytes, day: date, point_scale: float) -> pd.DataFrame:
    """Decode one day's Dukascopy M1 candle file (LZMA-compressed)."""
    cols = ["open", "high", "low", "close", "volume"]
    if not raw:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC"))
    data = lzma.decompress(raw)
    n = len(data) // _CANDLE_DTYPE.itemsize
    arr = np.frombuffer(data[: n * _CANDLE_DTYPE.itemsize], dtype=_CANDLE_DTYPE)
    idx = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(arr["t"].astype("int64"), unit="s")
    return pd.DataFrame(
        {
            "open": arr["o"] / point_scale,
            "high": arr["h"] / point_scale,
            "low": arr["l"] / point_scale,
            "close": arr["c"] / point_scale,
            "volume": arr["v"].astype("float64"),
        },
        index=pd.DatetimeIndex(idx, name="time"),
    )


def _fetch(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "forexlab-research"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return b""
            if attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    return b""


def download_dukascopy(symbol: str, start: date, end: date, cache_dir: Path,
                       pause: float = 0.2, verbose: bool = True) -> pd.DataFrame:
    """Download M1 bid and ask candles for [start, end] and merge them.

    Raw files are cached under cache_dir, so a re-run only downloads missing days.
    Note: Dukascopy's month in the URL is zero-based (January = 00).
    """
    inst = instruments.get(symbol)
    cache_dir = Path(cache_dir) / inst.symbol
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    day = start
    while day <= end:
        if day.weekday() != 5:  # Saturday has no FX trading
            sides = {}
            for side in ("BID", "ASK"):
                f = cache_dir / f"{day.isoformat()}_{side}.bi5"
                if f.exists():
                    raw = f.read_bytes()
                else:
                    url = DUKASCOPY_URL.format(symbol=inst.symbol, y=day.year,
                                               m=day.month - 1, d=day.day, side=side)
                    raw = _fetch(url)
                    f.write_bytes(raw)
                    time.sleep(pause)
                sides[side] = parse_candles_bi5(raw, day, inst.point_scale)
            if len(sides["BID"]) and len(sides["ASK"]):
                b = sides["BID"].add_prefix("bid_")
                a = sides["ASK"].add_prefix("ask_")
                merged = b.join(a, how="inner")
                merged["volume"] = merged.pop("bid_volume")
                merged = merged.drop(columns=["ask_volume"])
                frames.append(merged[merged["volume"] > 0])
            if verbose and day.day == 1:
                print(f"  {inst.symbol} {day:%Y-%m} ...", flush=True)
        day += timedelta(days=1)
    if not frames:
        raise RuntimeError(f"No data downloaded for {symbol} {start}..{end}")
    return pd.concat(frames)[BAR_COLUMNS].sort_index()


# --------------------------------------------------------------------------- MT5 (optional)

def load_mt5(symbol: str, start, end, timeframe: str = "M1") -> pd.DataFrame:
    """Load bars from a running MetaTrader 5 terminal (Windows only).

    MT5 gives bid OHLC plus the spread in points at bar open; the ask side is
    reconstructed as bid + spread. That under-states intrabar spread widening,
    so Dukascopy bid/ask candles are preferred for research.
    """
    import MetaTrader5 as mt5  # type: ignore  # optional dependency

    inst = instruments.get(symbol)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tf = getattr(mt5, f"TIMEFRAME_{timeframe}")
        rates = mt5.copy_rates_range(inst.symbol, tf, pd.Timestamp(start).to_pydatetime(),
                                     pd.Timestamp(end).to_pydatetime())
    finally:
        mt5.shutdown()
    df = pd.DataFrame(rates)
    df.index = pd.to_datetime(df["time"], unit="s", utc=True).rename("time")
    spread = df["spread"] / inst.point_scale
    out = pd.DataFrame(index=df.index)
    for k in ("open", "high", "low", "close"):
        out[f"bid_{k}"] = df[k]
        out[f"ask_{k}"] = df[k] + spread
    out["volume"] = df["tick_volume"].astype(float)
    return out[BAR_COLUMNS]


# --------------------------------------------------------------------------- storage

def save_bars(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, float_format="%.6f", compression="infer")


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, compression="infer")
    df.index = pd.to_datetime(df.index, utc=True).rename("time")
    missing = set(BAR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return df[BAR_COLUMNS].sort_index()


def fingerprint(df: pd.DataFrame) -> str:
    """Short, stable hash of a dataset, recorded with every experiment."""
    h = hashlib.sha256()
    h.update(str(df.index[0]).encode() + str(df.index[-1]).encode() + str(len(df)).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- transforms

def resample(df: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    agg = {}
    for side in ("bid", "ask"):
        agg[f"{side}_open"] = "first"
        agg[f"{side}_high"] = "max"
        agg[f"{side}_low"] = "min"
        agg[f"{side}_close"] = "last"
    agg["volume"] = "sum"
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["bid_open"])[BAR_COLUMNS]


def mid(df: pd.DataFrame, field: str = "close") -> pd.Series:
    return (df[f"bid_{field}"] + df[f"ask_{field}"]) / 2


# --------------------------------------------------------------------------- synthetic

def synthetic_bars(symbol: str = "EURUSD", start: str = "2019-01-01", end: str = "2024-12-31",
                   freq: str = "1h", seed: int = 7, base_spread_pips: float = 0.8,
                   news_spread_mult: float = 4.0, drift: float = 0.0) -> pd.DataFrame:
    """Random-walk bid/ask bars with realistic spread behaviour.

    There is no edge in this data by construction, so a pipeline that "finds"
    a profitable strategy here is overfitting. That makes it the standard
    smoke test for the research loop.
    """
    inst = instruments.get(symbol)
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    # FX trades Sunday 22:00 to Friday 22:00 UTC.
    wd, hr = idx.weekday, idx.hour
    open_mask = ~((wd == 5) | ((wd == 4) & (hr >= 22)) | ((wd == 6) & (hr < 22)))
    idx = idx[open_mask]
    n = len(idx)
    hours = idx.hour.values
    # Volatility is higher in the London and New York sessions.
    vol = np.where((hours >= 7) & (hours < 17), 1.4, 0.7) * 0.0009 * (inst.pip / 0.0001)
    start_price = {"EURUSD": 1.12, "GBPUSD": 1.28, "USDJPY": 130.0}[inst.symbol]
    rets = rng.standard_t(df=4, size=n) * vol / np.sqrt(2) + drift
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, vol * start_price * 0.6, size=(2, n)))
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    # Spread widens at the usual US data release hours (12:30, 14:00 UTC) and the daily rollover.
    news = np.isin(hours, [12, 14])
    rollover = np.isin(hours, [21, 22])
    spread = inst.pip * base_spread_pips * np.where(news, news_spread_mult, 1.0) \
        * np.where(rollover, 3.0, 1.0) * rng.uniform(0.8, 1.3, size=n)
    half = spread / 2
    df = pd.DataFrame(
        {
            "bid_open": open_ - half, "bid_high": high - half, "bid_low": low - half,
            "bid_close": close - half,
            "ask_open": open_ + half, "ask_high": high + half, "ask_low": low + half,
            "ask_close": close + half,
            "volume": rng.integers(500, 5000, size=n).astype(float),
        },
        index=pd.DatetimeIndex(idx, name="time"),
    )
    return df[BAR_COLUMNS]
