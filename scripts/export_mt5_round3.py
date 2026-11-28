"""Read-only MT5 data export for research round 3 (gotobi re-test).

Attaches to the ALREADY RUNNING, logged-in MT5 terminal and only reads price history.
It places no orders, changes no settings, and does not pass a path or login (so it
never restarts the terminal or switches account). mt5.shutdown() closes only this
script's own connection; the live demo engine keeps its own.

Output: data/round3/<SYMBOL>_<TF>_MT5_<start>_<end>.csv.gz in the lab's bar format
(UTC index, bid/ask OHLC, volume; ask = bid + the bar's spread).
Usage (from Documents/forex-ai):  .venv/Scripts/python.exe round3_export/export_mt5.py
"""
from pathlib import Path
import sys, time
import pandas as pd
import MetaTrader5 as mt5

H1_SYMBOLS = ["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY", "EURGBP",
              "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
M5_SYMBOLS = ["USDJPY", "EURJPY", "GBPJPY"]
START, END = "2019-01-01", "2025-10-31"
OUT = Path("data/round3")
COLS = ["bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close", "volume"]


def server_to_utc(t):
    ny = (pd.DatetimeIndex(t) - pd.Timedelta(hours=7)).tz_localize(
        "America/New_York", ambiguous=False, nonexistent="shift_forward")
    return ny.tz_convert("UTC")


def fetch(name, tf_name, start, end):
    if not mt5.symbol_select(name, True):
        return None, "not in this account"
    point = mt5.symbol_info(name).point
    tf = getattr(mt5, f"TIMEFRAME_{tf_name}")
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    frames, cur = [], lo - pd.Timedelta(days=1)
    step = pd.Timedelta(days=365 if tf_name == "H1" else 30)
    while cur < hi + pd.Timedelta(days=1):
        nxt = min(cur + step, hi + pd.Timedelta(days=1))
        r = mt5.copy_rates_range(name, tf, cur.to_pydatetime(), nxt.to_pydatetime())
        if r is not None and len(r):
            frames.append(pd.DataFrame(r))
        cur = nxt
        time.sleep(0.2)
    if not frames:
        return None, f"no bars ({mt5.last_error()})"
    df = pd.concat(frames).drop_duplicates("time")
    df.index = server_to_utc(pd.to_datetime(df["time"], unit="s")).rename("time")
    df = df[(df.index >= lo) & (df.index < hi)].sort_index()
    sp = df["spread"] * point
    out = pd.DataFrame(index=df.index)
    for k in ("open", "high", "low", "close"):
        out[f"bid_{k}"] = df[k]
        out[f"ask_{k}"] = df[k] + sp
    out["volume"] = df["tick_volume"].astype(float)
    return out[COLS], f"{len(out):,} bars {out.index[0]} .. {out.index[-1]}, median spread {df['spread'].median():.0f} pts"


def main():
    if not mt5.initialize(timeout=120_000):
        sys.exit(f"MT5 initialize failed: {mt5.last_error()} (is the terminal open?)")
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        for tf, syms in (("H1", H1_SYMBOLS), ("M5", M5_SYMBOLS)):
            for s in syms:
                df, msg = fetch(s, tf, START, END)
                if df is not None:
                    p = OUT / f"{s}_{tf}_MT5_{START}_{END}.csv.gz"
                    df.to_csv(p, float_format="%.6f")
                    msg += f" -> {p}"
                print(f"{s} {tf}: {msg}", flush=True)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
