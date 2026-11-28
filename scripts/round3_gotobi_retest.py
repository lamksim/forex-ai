"""Round 3: re-test gotobi on REAL broker cross data (MT5 export in data/round3).

Research only: reads bar files, writes reports and a separate experiment log.
Run from the lab root:  python scripts/round3_gotobi_retest.py
Holdout rule: H1 research stops at the usual 80% cut. The M5 files (Jul-Oct 2025) lie
inside the holdout, so they are used ONLY to measure spreads around the entry time,
never strategy returns.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forexlab import data as D, metrics as M
from forexlab.backtest import CostModel, run_backtest
from forexlab.instruments import get
from forexlab.registry import ExperimentLog
from forexlab.research import research
from forexlab.splits import split_holdout
from forexlab.strategies import generate, gotobi_days

R3 = Path("data/round3")
TAG = "MT5_2019-01-01_2025-10-31.csv.gz"
FLOORS = {"USDJPY": 0.3, "EURJPY": 0.7, "GBPJPY": 1.0, "AUDJPY": 0.8, "NZDJPY": 1.2, "CADJPY": 1.2, "CHFJPY": 1.2}
LOG = Path("research_log/round3_real.jsonl")
REPORTS = Path("research_log/reports/round3_real")
GOTOBI = {"entry_jst": 7, "exit_jst": 9, "stop_atr": 3.0}


def load(sym: str) -> pd.DataFrame:
    b = D.load_bars(R3 / f"{sym}_H1_{TAG}")
    if sym != "USDJPY":
        uj = D.load_bars(R3 / f"USDJPY_H1_{TAG}")
        b = b.join((1.0 / D.mid(uj, "open")).rename("quote_usd"), how="inner")
    return D.resample(b, "1h")


def fixed(sym, rb, sig, stp, floor):
    res = run_backtest(rb, sig, stp, sym, costs=CostModel(3.5, 0.2, floor))
    t = res.trades
    pips = ((t.exit - t.entry) * t.direction / get(sym).pip).mean() if len(t) else float("nan")
    return M.daily_returns(res.equity), res, pips


def main():
    if not LOG.exists():
        shutil.copy("research_log/experiments.jsonl", LOG)  # keep every earlier trial in the count
    log = ExperimentLog(LOG)
    out = {}
    live = None
    print("== 1. Walk-forward on real H1 data (24m train / 6m test), research period only")
    for sym, floor in FLOORS.items():
        bars = load(sym)
        rb, cut = split_holdout(bars, 0.2)
        r = research(rb, sym, "gotobi", D.fingerprint(bars), log, CostModel(3.5, 0.2, floor), report_dir=REPORTS)
        m = r.oos_metrics
        sp = ((rb.ask_open - rb.bid_open) / get(sym).pip)
        print(f"{sym} floor {floor}: WF sharpe {m['sharpe']:+.2f} ret {m['total_return']:+.2%} maxDD {m['max_drawdown']:.2%} "
              f"trades {m['trades']} PF {m.get('profit_factor') or 0:.2f} DSR {r.deflated['dsr']:.3f} (N={r.deflated['n_trials']}) "
              f"{'PASS' if r.passed else 'reject'} | params per fold {[f['params']['entry_jst'] * 100 + f['params']['exit_jst'] for f in r.folds]} "
              f"| data spread at 22 UTC median {sp[rb.index.hour == 22].median():.2f} pips")
        out[sym] = r.oos_equity
    print("\n== 2. Fixed 07->09 JST, by spread floor (pips), net pips/trade, by year, placebo")
    for sym in FLOORS:
        rb, _ = split_holdout(load(sym), 0.2)
        sig, stp = generate("gotobi", rb, GOTOBI)
        row = []
        for f in (0.3, 0.6, 1.0, 1.5, 2.0):
            r, res, p = fixed(sym, rb, sig, stp, f)
            row.append(f"{f}:{M.sharpe(r):+.2f}")
        r, res, p = fixed(sym, rb, sig, stp, FLOORS[sym])
        yr = r.groupby(r.index.year).apply(M.sharpe)
        nxt = (rb.index + pd.Timedelta(hours=1)).tz_convert("Asia/Tokyo")
        days = gotobi_days(rb.index)
        nong = np.array([(d not in days) and d.weekday() < 5 for d in nxt.date])
        psig = pd.Series(np.where(nong & (nxt.hour >= 7) & (nxt.hour < 9), 1, 0), index=rb.index)
        pr, pres, pp = fixed(sym, rb, psig, stp, FLOORS[sym])
        print(f"{sym}: trades {len(res.trades)} net pips/trade {p:+.2f} | SR by floor {' '.join(row)} | "
              f"by year {' '.join(f'{y}:{v:+.1f}' for y, v in yr.items())} | placebo SR {M.sharpe(pr):+.2f} ({pp:+.2f} pips)")
        if sym == "USDJPY":
            lsig, lstp = generate("ma_cross_trending", rb, {"er_window": 120, "er_min": 0.15, "stop_atr": 3.0})
            # From 2021 (the walk-forward OOS span); over 2019-20 it hits the 5% halt.
            live, _, _ = fixed(sym, rb[rb.index >= "2021-01-01"], lsig, lstp, 0.3)
    print("\n== 3. Basket of 7 yen pairs (walk-forward OOS, equal weight daily returns)")
    rets = pd.concat({s: e.pct_change() for s, e in out.items()}, axis=1).fillna(0)
    basket = rets.mean(axis=1)
    eq = (1 + basket).cumprod()
    print(f"basket sharpe {M.sharpe(basket):+.2f}, maxDD {M.max_drawdown(eq):.2%}, avg pair corr {rets.corr().values[np.triu_indices(len(out), 1)].mean():.2f}")
    lv = live.copy(); lv.index = lv.index.tz_convert("UTC").normalize().tz_localize(None)
    bk = basket.copy(); bk.index = pd.DatetimeIndex(bk.index).tz_convert("UTC").normalize().tz_localize(None)
    j = pd.DataFrame({"live": lv, "basket": bk}).dropna()
    print(f"corr basket vs live USDJPY ma_cross_trending: {j['live'].corr(j['basket']):.3f} over {len(j)} days")
    print("\n== 4. Real M5 spreads around the 22:00 UTC entry (Jul-Oct 2025, spreads only), pips")
    for sym in ("USDJPY", "EURJPY", "GBPJPY"):
        p = R3 / f"{sym}_M5_{TAG}"
        if not p.exists():
            continue
        m5 = D.load_bars(p)
        sp = (m5.ask_open - m5.bid_open) / get(sym).pip
        hm = m5.index.strftime("%H:%M")
        winter = m5.index.tz_convert("America/New_York").map(lambda x: x.utcoffset().total_seconds() == -5 * 3600)
        rows = []
        for t in ("20:55", "21:00", "21:05", "21:10", "21:30", "22:00", "22:05", "22:10", "22:15", "22:30", "23:00", "00:00"):
            s = sp[hm == t]
            rows.append(f"{t} {s.median():.1f}/{s.quantile(0.9):.1f}")
        print(f"{sym} (median/p90; summer rollover 21:00 UTC; {int(np.asarray(winter).sum())} winter bars): " + "  ".join(rows))
    print("\nReports in", REPORTS, "log", LOG)


if __name__ == "__main__":
    sys.exit(main())
