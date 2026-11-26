import lzma
from datetime import date

import numpy as np
import pandas as pd
import pytest

from forexlab import data as D
from forexlab import metrics as M
from forexlab.backtest import CostModel, run_backtest
from forexlab.registry import ExperimentLog
from forexlab.research import research
from forexlab.risk import RiskLimits
from forexlab.splits import HoldoutVault, HoldoutViolation, split_holdout, walk_forward_folds
from forexlab.strategies import FAMILIES, generate


@pytest.fixture(scope="module")
def bars():
    return D.synthetic_bars("EURUSD", "2020-01-01", "2021-01-01", seed=1)


def flat_bars(n=50, price=1.1000, spread=0.0001, start="2024-01-02"):
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    half = spread / 2
    return pd.DataFrame({
        "bid_open": price - half, "bid_high": price - half, "bid_low": price - half, "bid_close": price - half,
        "ask_open": price + half, "ask_high": price + half, "ask_low": price + half, "ask_close": price + half,
        "volume": 1.0,
    }, index=idx)


# ---------------------------------------------------------------- data

def test_dukascopy_parser_roundtrip():
    rec = np.array([(0, 110000, 110010, 109990, 110020, 12.5), (60, 110010, 110005, 110000, 110015, 3.0)],
                   dtype=D._CANDLE_DTYPE)
    raw = lzma.compress(rec.tobytes(), format=lzma.FORMAT_ALONE)
    df = D.parse_candles_bi5(raw, date(2024, 3, 4), 1e5)
    assert list(df.index.strftime("%H:%M")) == ["00:00", "00:01"]
    assert df.iloc[0]["open"] == pytest.approx(1.1)
    assert df.iloc[0]["close"] == pytest.approx(1.1001)
    assert df.iloc[0]["high"] == pytest.approx(1.1002)
    assert df.iloc[0]["low"] == pytest.approx(1.0999)


def test_resample_keeps_bid_ask(bars):
    h4 = D.resample(bars, "4h")
    assert (h4["ask_close"] > h4["bid_close"]).all()
    assert list(h4.columns) == D.BAR_COLUMNS


# ---------------------------------------------------------------- strategies

@pytest.mark.parametrize("family", list(FAMILIES))
def test_no_lookahead(bars, family):
    """Signals up to bar t must not change when future bars are removed."""
    params = FAMILIES[family].configs()[0]
    full_sig, full_stop = generate(family, bars, params)
    cut = len(bars) * 2 // 3
    part_sig, part_stop = generate(family, bars.iloc[:cut], params)
    pd.testing.assert_series_equal(full_sig.iloc[:cut], part_sig, check_names=False)
    pd.testing.assert_series_equal(full_stop.iloc[:cut], part_stop, check_names=False)


# ---------------------------------------------------------------- backtest

def test_costs_make_a_flat_market_lose_money():
    b = flat_bars()
    sig = pd.Series([1, 0] * 25, index=b.index)
    stop = pd.Series(0.0020, index=b.index)
    res = run_backtest(b, sig, stop, "EURUSD")
    assert len(res.trades) > 0
    assert res.equity.iloc[-1] < 10_000
    assert (res.trades["pnl_usd"] < 0).all()
    assert res.trades["commission_usd"].gt(0).all()


def test_trade_sized_to_quarter_percent_risk():
    b = flat_bars(10)
    # Price collapses on bar 3 so the long stop is hit.
    b.loc[b.index[3], ["bid_low", "ask_low"]] = [1.0900, 1.0901]
    sig = pd.Series(1, index=b.index)
    stop = pd.Series(0.0020, index=b.index)
    res = run_backtest(b, sig, stop, "EURUSD", costs=CostModel(0, 0, 0))
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["pnl_usd"] == pytest.approx(-25.0, rel=1e-6)  # 0.25% of 10,000


def test_gap_through_stop_fills_at_open():
    b = flat_bars(10)
    i = b.index[3]
    b.loc[i, ["bid_open", "bid_high", "bid_low", "bid_close"]] = 1.0950
    b.loc[i, ["ask_open", "ask_high", "ask_low", "ask_close"]] = 1.0951
    res = run_backtest(b, pd.Series(1, index=b.index), pd.Series(0.0020, index=b.index), "EURUSD",
                       costs=CostModel(0, 0, 0))
    t = res.trades.iloc[0]
    assert t["exit"] == pytest.approx(1.0950)
    assert t["pnl_usd"] < -25.0  # gap loss exceeds planned risk


def test_signal_acts_on_next_bar_open():
    b = flat_bars(6)
    sig = pd.Series([0, 1, 1, 1, 1, 1], index=b.index)
    res = run_backtest(b, sig, pd.Series(0.002, index=b.index), "EURUSD")
    assert res.trades.iloc[0]["entry_time"] == b.index[2]


def test_daily_loss_limit_halts_trading():
    b = flat_bars(48, start="2024-01-02")
    # Every 2 bars the price drops 30 pips: each long loses its full risk.
    price = 1.1000 - 0.0030 * (np.arange(48) // 2)
    for side, off in (("bid", -0.00005), ("ask", 0.00005)):
        for k in ("open", "high", "low", "close"):
            b[f"{side}_{k}"] = price + off
    sig = pd.Series(1, index=b.index)
    res = run_backtest(b, sig, pd.Series(0.0020, index=b.index), "EURUSD",
                       limits=RiskLimits(max_drawdown=0.5, max_weekly_loss=0.5))
    first_day = res.trades[res.trades["entry_time"].dt.date == b.index[0].date()]
    # 1% daily loss / 0.25% per trade: trading stops after about 4 losers.
    assert 3 <= len(first_day) <= 5
    assert any(e["event"] == "daily_loss" for e in res.risk_events)


def test_max_drawdown_is_a_permanent_halt():
    b = flat_bars(24 * 10, start="2024-01-01")
    price = 1.1000 - 0.0030 * (np.arange(len(b)) // 2)
    for side, off in (("bid", -0.00005), ("ask", 0.00005)):
        for k in ("open", "high", "low", "close"):
            b[f"{side}_{k}"] = price + off
    res = run_backtest(b, pd.Series(1, index=b.index), pd.Series(0.0020, index=b.index), "EURUSD")
    assert res.halted
    assert res.equity.iloc[-1] > 10_000 * (1 - 0.05) - 100  # stopped near the 5% line


def test_usdjpy_pnl_converted_to_usd():
    b = flat_bars(10, price=150.00, spread=0.01)
    b.loc[b.index[3], ["bid_low", "ask_low"]] = [149.0, 149.01]
    res = run_backtest(b, pd.Series(1, index=b.index), pd.Series(0.30, index=b.index), "USDJPY",
                       costs=CostModel(0, 0, 0))
    assert res.trades.iloc[0]["pnl_usd"] == pytest.approx(-25.0, rel=0.01)


# ---------------------------------------------------------------- metrics

def test_deflated_sharpe_penalizes_many_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.005, 500))
    one = M.deflated_sharpe(r, 1, [1.5])["dsr"]
    many = M.deflated_sharpe(r, 500, list(rng.normal(0, 1, 500)))["dsr"]
    assert one > many


def test_max_drawdown():
    eq = pd.Series([100, 110, 99, 120])
    assert M.max_drawdown(eq) == pytest.approx(-0.1)


# ---------------------------------------------------------------- splits & log

def test_holdout_split_and_folds(bars):
    rb, cut = split_holdout(bars, 0.2)
    assert rb.index[-1] < cut <= bars.index[-1]
    folds = walk_forward_folds(rb.index, 3, 1)
    assert folds and all(f.train_end < f.test_end for f in folds)
    assert all(a.train_end <= b.train_end for a, b in zip(folds, folds[1:]))


def test_holdout_vault_allows_one_look(tmp_path):
    v = HoldoutVault(tmp_path / "vault.jsonl")
    v.open(symbol="EURUSD", version="v1", config_id="abc", data_fingerprint="x", reason="t")
    with pytest.raises(HoldoutViolation):
        v.open(symbol="EURUSD", version="v1", config_id="other", data_fingerprint="x", reason="t")
    with pytest.raises(HoldoutViolation):
        v.open(symbol="EURUSD", version="v2", config_id="abc", data_fingerprint="x", reason="t")


def test_research_logs_every_trial_and_rejects_noise(tmp_path):
    b = D.synthetic_bars("EURUSD", "2019-01-01", "2021-07-01", seed=3)
    log = ExperimentLog(tmp_path / "exp.jsonl")
    r = research(b, "EURUSD", "donchian", D.fingerprint(b), log, train_months=12, test_months=6,
                 report_dir=tmp_path / "reports")
    n_cfg = len(FAMILIES["donchian"].configs())
    train = [e for e in log.entries() if e["kind"] == "train"]
    assert len(train) == n_cfg * len(r.folds)
    assert r.deflated["n_trials"] == n_cfg
    assert not r.passed  # random walk: there is no edge to find
    assert r.report_path.exists()


def test_download_uses_cache_and_merges_bid_ask(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        base = 110000 if "BID" in url else 110002
        rec = np.array([(m * 60, base, base + 1, base - 1, base + 2, 1.0) for m in range(3)],
                       dtype=D._CANDLE_DTYPE)
        return lzma.compress(rec.tobytes(), format=lzma.FORMAT_ALONE)

    monkeypatch.setattr(D, "_fetch", fake_fetch)
    df = D.download_dukascopy("EURUSD", date(2024, 1, 1), date(2024, 1, 7), tmp_path, pause=0, verbose=False)
    assert len(calls) == 12  # 6 trading days (Saturday skipped) x bid/ask
    assert "/2024/00/01/" in calls[0]  # zero-based month
    assert (df["ask_close"] - df["bid_close"]).round(6).eq(0.00002).all()
    D.download_dukascopy("EURUSD", date(2024, 1, 1), date(2024, 1, 7), tmp_path, pause=0, verbose=False)
    assert len(calls) == 12  # second run served from cache


def test_h1_download_monthly_files(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        base = 110000 if "BID" in url else 110003
        rec = np.array([(h * 3600, base, base + 1, base - 1, base + 2, 5.0) for h in range(24 * 3)],
                       dtype=D._CANDLE_DTYPE)
        return lzma.compress(rec.tobytes(), format=lzma.FORMAT_ALONE)

    monkeypatch.setattr(D, "_fetch", fake_fetch)
    df = D.download_dukascopy_h1("EURUSD", date(2024, 1, 2), date(2024, 2, 2), tmp_path, pause=0, verbose=False)
    assert len(calls) == 4  # Jan + Feb, bid + ask
    assert calls[0].endswith("/EURUSD/2024/00/BID_candles_hour_1.bi5")
    assert df.index[0] == pd.Timestamp("2024-01-02", tz="UTC")  # clipped to start
    assert pd.Timestamp("2024-02-01 05:00", tz="UTC") in df.index  # offsets from month start


def test_ex_overlap_is_flat_during_overlap(bars):
    sig, _ = generate("rsi_reversion_ex_overlap", bars, {"period": 7, "band": 20, "stop_atr": 2.5})
    res = run_backtest(bars, sig, generate("rsi_reversion", bars, {"period": 7, "band": 20, "stop_atr": 2.5})[1],
                       "EURUSD", costs=CostModel(0, 0, 0))
    assert len(res.trades)
    held = []
    for _, t in res.trades.iterrows():
        in_trade = bars.index[(bars.index >= t["entry_time"]) & (bars.index < t["exit_time"])]
        held += list(in_trade.hour)
    assert not any(12 <= h < 16 for h in held)


def test_load_mt5_builds_ask_from_spread(monkeypatch):
    import sys, types
    rates = np.array([(1704153600 + 3600 * i, 1.1, 1.1005, 1.0995, 1.1001, 100, 12, 0) for i in range(3)],
                     dtype=[("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
                            ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")])
    fake = types.SimpleNamespace(
        initialize=lambda **kw: True, shutdown=lambda: None, last_error=lambda: None,
        symbol_select=lambda name, on: name == "EURUSD.m",
        symbol_info=lambda name: types.SimpleNamespace(point=0.00001),
        TIMEFRAME_H1=16385, copy_rates_range=lambda *a: rates)
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    df = D.load_mt5("EURUSD", "2024-01-02", "2024-01-03", "H1", broker_symbol="EURUSD.m", utc_offset_hours=0)
    assert list(df.columns) == D.BAR_COLUMNS
    assert (df["ask_close"] - df["bid_close"]).round(6).eq(0.00012).all()  # 12 points = 1.2 pips
    assert str(df.index[0]) == "2024-01-02 00:00:00+00:00"
    shifted = D.load_mt5("EURUSD", "2024-01-02", "2024-01-03", "H1", broker_symbol="EURUSD.m", utc_offset_hours=2)
    # server 00:00-02:00 at UTC+2 is 22:00-00:00 UTC; bars before the requested start are clipped
    assert list(shifted.index.strftime("%Y-%m-%d %H:%M")) == ["2024-01-02 00:00"]


def test_server_time_follows_us_daylight_saving():
    winter = D.server_time_to_utc(pd.DatetimeIndex(["2024-01-10 10:00"]))
    summer = D.server_time_to_utc(pd.DatetimeIndex(["2024-07-10 10:00"]))
    assert str(winter[0]) == "2024-01-10 08:00:00+00:00"  # UTC+2
    assert str(summer[0]) == "2024-07-10 07:00:00+00:00"  # UTC+3
    fixed = D.server_time_to_utc(pd.DatetimeIndex(["2024-07-10 10:00"]), 2)
    assert str(fixed[0]) == "2024-07-10 08:00:00+00:00"
