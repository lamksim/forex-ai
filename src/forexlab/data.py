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
# Hourly candles come one file per month (far fewer requests; research runs on H1 anyway).
DUKASCOPY_H1_URL = (
    "https://datafeed.dukascopy.com/datafeed/{symbol}/{y:04d}/{m:02d}/{side}_candles_hour_1.bi5"
)
# Each record in a Dukascopy candle file is 24 bytes, big-endian:
# seconds from the start of the file's period (day for M1, month for H1), open, close, low, high (int32 scaled by point) and volume (float32).
_CANDLE_DTYPE = np.dtype([
    ("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4"),
])


# --------------------------------------------------------------------------- Dukascopy

def parse_candles_bi5(raw: bytes, day: date, point_scale: float) -> pd.DataFrame:
    """Decode a Dukascopy candle file (LZMA-compressed). `day` is the start of the
    file's period: the day for M1 files, the first of the month for H1 files."""
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


class FetchError(RuntimeError):
    pass


def _fetch(url: str, retries: int = 7) -> bytes:
    """GET with exponential backoff. Dukascopy answers 503 when it is throttling."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (forexlab research)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return b""
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = str(exc)
        time.sleep(min(2 ** attempt, 60))
    raise FetchError(f"{url}: gave up after {retries} tries ({last})")


def download_dukascopy(symbol: str, start: date, end: date, cache_dir: Path,
                       batch_size: int = 8, pause: float = 1.0, verbose: bool = True) -> pd.DataFrame:
    """Download M1 bid and ask candles for [start, end] and merge them.

    Files are fetched in small concurrent batches with a pause between batches
    (the pacing the community Dukascopy clients use) and cached under cache_dir,
    so a re-run, or a run after a failure, only downloads missing days.
    Note: Dukascopy's month in the URL is zero-based (January = 00).
    """
    from concurrent.futures import ThreadPoolExecutor

    inst = instruments.get(symbol)
    cache_dir = Path(cache_dir) / inst.symbol
    cache_dir.mkdir(parents=True, exist_ok=True)
    days = []
    d = start
    while d <= end:
        if d.weekday() != 5:  # Saturday has no FX trading
            days.append(d)
        d += timedelta(days=1)

    def path(day, side):
        return cache_dir / f"{day.isoformat()}_{side}.bi5"

    todo = [(day, side) for day in days for side in ("BID", "ASK") if not path(day, side).exists()]
    if verbose:
        print(f"{inst.symbol}: {len(days)} days, {len(todo)} files to download "
              f"({2 * len(days) - len(todo)} cached)", flush=True)

    def get(job):
        day, side = job
        url = DUKASCOPY_URL.format(symbol=inst.symbol, y=day.year, m=day.month - 1, d=day.day, side=side)
        raw = _fetch(url)
        tmp = path(day, side).with_suffix(".part")
        tmp.write_bytes(raw)
        tmp.rename(path(day, side))  # only complete files land in the cache
        return len(raw)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        for i in range(0, len(todo), batch_size):
            list(pool.map(get, todo[i:i + batch_size]))
            done = min(i + batch_size, len(todo))
            if verbose and ((done // batch_size) % 25 == 0 or done == len(todo)):
                print(f"  {done}/{len(todo)} files, {time.time() - t0:.0f}s, at {todo[done - 1][0]}", flush=True)
            time.sleep(pause)

    frames = []
    for day in days:
        b = parse_candles_bi5(path(day, "BID").read_bytes(), day, inst.point_scale)
        a = parse_candles_bi5(path(day, "ASK").read_bytes(), day, inst.point_scale)
        if len(b) and len(a):
            merged = b.add_prefix("bid_").join(a.add_prefix("ask_"), how="inner")
            merged["volume"] = merged.pop("bid_volume")
            merged = merged.drop(columns=["ask_volume"])
            frames.append(merged[merged["volume"] > 0])
    if not frames:
        raise RuntimeError(f"No data downloaded for {symbol} {start}..{end}")
    return pd.concat(frames)[BAR_COLUMNS].sort_index()


def download_dukascopy_h1(symbol: str, start: date, end: date, cache_dir: Path,
                          pause: float = 3.0, verbose: bool = True) -> pd.DataFrame:
    """Download H1 bid and ask candles, one file per side per month.

    About 25x fewer requests than M1, which matters because Dukascopy throttles
    hard. The trade-off: spread is only seen at hourly resolution.
    """
    inst = instruments.get(symbol)
    cache_dir = Path(cache_dir) / inst.symbol / "h1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    months = []
    m = date(start.year, start.month, 1)
    while m <= end:
        months.append(m)
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    frames = []
    t0 = time.time()
    for k, month in enumerate(months, 1):
        sides = {}
        for side in ("BID", "ASK"):
            f = cache_dir / f"{month:%Y-%m}_{side}.bi5"
            if not f.exists():
                url = DUKASCOPY_H1_URL.format(symbol=inst.symbol, y=month.year, m=month.month - 1, side=side)
                raw = _fetch(url)
                tmp = f.with_suffix(".part")
                tmp.write_bytes(raw)
                tmp.rename(f)
                time.sleep(pause)
            sides[side] = parse_candles_bi5(f.read_bytes(), month, inst.point_scale)
        if len(sides["BID"]) and len(sides["ASK"]):
            merged = sides["BID"].add_prefix("bid_").join(sides["ASK"].add_prefix("ask_"), how="inner")
            merged["volume"] = merged.pop("bid_volume")
            merged = merged.drop(columns=["ask_volume"])
            frames.append(merged[merged["volume"] > 0])
        if verbose:
            print(f"  {month:%Y-%m} done ({k}/{len(months)}, {time.time() - t0:.0f}s)", flush=True)
    if not frames:
        raise RuntimeError(f"No data downloaded for {symbol} {start}..{end}")
    df = pd.concat(frames)[BAR_COLUMNS].sort_index()
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= lo) & (df.index < hi)]


# --------------------------------------------------------------------------- MT5 (optional)

def server_time_to_utc(server: pd.DatetimeIndex | pd.Series, utc_offset_hours: float | None = None) -> pd.DatetimeIndex:
    """Convert naive broker server times to UTC.

    Default rule: server time = New York local time + 7 h (UTC+2 winter, UTC+3 summer).
    The US clock change happens on a Sunday at 02:00 New York time, while the FX
    market is closed, so ambiguous or missing hours do not occur in real bars.
    """
    server = pd.DatetimeIndex(server)
    if utc_offset_hours is not None:
        return (server - pd.Timedelta(hours=utc_offset_hours)).tz_localize("UTC")
    ny = (server - pd.Timedelta(hours=7)).tz_localize("America/New_York", ambiguous=False,
                                                       nonexistent="shift_forward")
    return ny.tz_convert("UTC")


def load_mt5(symbol: str, start, end, timeframe: str = "H1", broker_symbol: str | None = None,
             utc_offset_hours: float | None = None, terminal_path: str | None = None) -> pd.DataFrame:
    """Load bars from a running MetaTrader 5 terminal (Windows only).

    MT5 gives bid OHLC plus the broker's spread in points at each bar; the ask
    side is reconstructed as bid + spread. That understates intrabar spread
    widening a little, but the spread is your broker's own.
    `broker_symbol` covers brokers that add a suffix (e.g. "EURUSD.m").
    MT5 bar times are in the broker's server time, not UTC. Most FX brokers,
    MetaQuotes-Demo included, run "New York time + 7 h" (UTC+2 in winter, UTC+3
    in summer), so by default bars are converted with that rule, which tracks the
    US daylight-saving switch. Pass `utc_offset_hours` only for a broker that
    uses a fixed offset all year.
    """
    import MetaTrader5 as mt5  # type: ignore  # optional dependency

    instruments.get(symbol)  # only approved pairs
    name = broker_symbol or symbol.upper()
    ok = mt5.initialize(path=terminal_path, timeout=120_000) if terminal_path else mt5.initialize(timeout=120_000)
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}. Is the terminal open and logged in?")
    try:
        if not mt5.symbol_select(name, True):
            raise RuntimeError(f"Symbol {name} not available in this MT5 account. "
                               "Check Market Watch for the exact name and pass --mt5-symbol.")
        info = mt5.symbol_info(name)
        point = info.point
        swap = {"swap_long": getattr(info, "swap_long", None), "swap_short": getattr(info, "swap_short", None),
                "swap_mode": getattr(info, "swap_mode", None)}
        tf = getattr(mt5, f"TIMEFRAME_{timeframe}")
        # Request a padded window as explicit UTC datetimes (naive ones are read
        # in the PC's local time zone), then clip exactly after converting to UTC.
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        rates = mt5.copy_rates_range(name, tf, (lo - pd.Timedelta(days=1)).to_pydatetime(),
                                     (hi + pd.Timedelta(days=1)).to_pydatetime())
    finally:
        mt5.shutdown()
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no {timeframe} bars for {name}. Scroll the chart back "
                           "or raise Tools > Options > Charts > Max bars, then retry.")
    df = pd.DataFrame(rates)
    df.index = server_time_to_utc(pd.to_datetime(df["time"], unit="s"), utc_offset_hours).rename("time")
    df = df[(df.index >= lo) & (df.index < hi)]
    spread = df["spread"] * point
    out = pd.DataFrame(index=df.index)
    for k in ("open", "high", "low", "close"):
        out[f"bid_{k}"] = df[k]
        out[f"ask_{k}"] = df[k] + spread
    out["volume"] = df["tick_volume"].astype(float)
    out = out[BAR_COLUMNS]
    out.attrs["swap"] = swap
    return out


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
    vol = np.where((hours >= 7) & (hours < 17), 1.4, 0.7) * 0.0009  # log-return vol, same for every pair
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
