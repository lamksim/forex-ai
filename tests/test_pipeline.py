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
