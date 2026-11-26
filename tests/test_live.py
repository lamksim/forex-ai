"""Demo forward-tester against a fake MetaTrader5 module (no terminal needed)."""

import types

import numpy as np
import pandas as pd
import pytest

from forexlab import live as L
from forexlab.risk import RiskLimits


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_REAL = 0, 2
    TIMEFRAME_H1 = 16385
    POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    TRADE_ACTION_DEAL = 1
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    ORDER_TIME_GTC = 0

    def __init__(self, demo=True, equity=10_000.0, fillings_ok=(1,)):
        self.trade_mode = 0 if demo else 2
        self.equity = equity
        self.positions = []
        self.orders = []
        self.fillings_ok = fillings_ok
        n = 600
        t0 = int(pd.Timestamp("2026-06-01").timestamp())
        close = 150 + np.cumsum(np.full(n, 0.01))  # steady uptrend -> long signal
        self.rates = np.zeros(n, dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                                        ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4")])
        self.rates["time"] = t0 + 3600 * np.arange(n)
        self.rates["open"] = close - 0.01
        self.rates["high"] = close + 0.02
        self.rates["low"] = close - 0.03
        self.rates["close"] = close
        self.rates["tick_volume"] = 100
        self.rates["spread"] = 10

    def account_info(self):
        return types.SimpleNamespace(login=1, server="Fake-Demo", trade_mode=self.trade_mode,
                                     equity=self.equity, balance=self.equity)

    def last_error(self):
        return (0, "ok")

    def symbol_info(self, name):
        return types.SimpleNamespace(point=0.001, digits=3, trade_contract_size=100_000,
                                     volume_step=0.01, volume_min=0.01, volume_max=100.0)

    def symbol_info_tick(self, name):
        c = float(self.rates["close"][-1])
        return types.SimpleNamespace(bid=c, ask=c + 0.01)

    def copy_rates_from_pos(self, name, tf, start, count):
        return self.rates[-count:]

    def positions_get(self, symbol=None):
        return tuple(self.positions)

    def order_send(self, req):
        self.orders.append(req)
        if req["type_filling"] not in self.fillings_ok:
            return types.SimpleNamespace(retcode=10030)
        if "position" in req:
            self.positions = [p for p in self.positions if p.ticket != req["position"]]
        else:
            self.positions.append(types.SimpleNamespace(
                ticket=len(self.orders), magic=req["magic"], volume=req["volume"],
                type=self.POSITION_TYPE_BUY if req["type"] == self.ORDER_TYPE_BUY else self.POSITION_TYPE_SELL))
        return types.SimpleNamespace(retcode=10009, order=len(self.orders), price=req["price"],
                                     volume=req["volume"], comment="done")


def sig_fn(value, stop=0.5):
    def fn(family, bars, params):
        return (pd.Series(value, index=bars.index), pd.Series(stop, index=bars.index))
    return fn


NOW = pd.Timestamp("2026-07-01 10:00", tz="UTC")


@pytest.fixture
def cfg(tmp_path):
    return L.LiveConfig(log_dir=tmp_path, utc_offset_hours=0)


def test_refuses_real_account(cfg):
    with pytest.raises(L.NotDemoAccount):
        L.run_once(FakeMT5(demo=False), cfg, NOW, sig_fn(1))


def test_opens_with_risk_sized_lots_and_server_stop(cfg):
    mt5 = FakeMT5(fillings_ok=(2,))  # broker only accepts RETURN filling
    ev = L.run_once(mt5, cfg, NOW, sig_fn(1, stop=0.5))
    op = [a for a in ev["actions"] if a["action"] == "open"][0]
    assert op["ok"] and op["direction"] == 1
    # 0.25% of 10k = $25 risk; 0.5 JPY stop at ~156 USDJPY -> ~7,800 units -> 0.07 lots
    price = mt5.symbol_info_tick("USDJPY").ask
    expected_units = 25 / (0.5 / price)
    assert op["lots"] == pytest.approx(np.floor(expected_units / 100_000 / 0.01) * 0.01)
    assert op["sl"] == pytest.approx(round(price - 0.5, 3))
    assert mt5.orders[-1]["sl"] == op["sl"] and mt5.orders[-1]["magic"] == cfg.magic
    assert (cfg.log_dir / f"USDJPY_{pd.Timestamp.now():%Y-%m}.jsonl").exists()


def test_closes_on_signal_flip_and_reverses(cfg):
    mt5 = FakeMT5()
    L.run_once(mt5, cfg, NOW, sig_fn(1))
    ev = L.run_once(mt5, cfg, NOW + pd.Timedelta(hours=1), sig_fn(-1))
    kinds = [(a["action"], a.get("reason"), a.get("direction")) for a in ev["actions"]]
    assert kinds == [("close", "signal", None), ("open", None, -1)]
    assert len(mt5.positions) == 1 and mt5.positions[0].type == mt5.POSITION_TYPE_SELL


def test_holds_when_signal_unchanged(cfg):
    mt5 = FakeMT5()
    L.run_once(mt5, cfg, NOW, sig_fn(1))
    ev = L.run_once(mt5, cfg, NOW + pd.Timedelta(hours=1), sig_fn(1))
    assert ev["actions"] == [] and ev["position_before"] == 1


def test_ignores_other_magic_numbers(cfg):
    mt5 = FakeMT5()
    mt5.positions.append(types.SimpleNamespace(ticket=999, magic=1, volume=1.0, type=1))
    ev = L.run_once(mt5, cfg, NOW, sig_fn(0))
    assert ev["actions"] == [] and any(p.ticket == 999 for p in mt5.positions)


def test_stop_file_flattens(cfg):
    mt5 = FakeMT5()
    L.run_once(mt5, cfg, NOW, sig_fn(1))
    (cfg.log_dir / "STOP").touch()
    ev = L.run_once(mt5, cfg, NOW + pd.Timedelta(hours=1), sig_fn(1))
    assert ev["event"] == "stopped" and ev["actions"][0]["reason"] == "kill_switch"
    assert mt5.positions == []


def test_skips_below_broker_minimum(cfg):
    mt5 = FakeMT5(equity=100.0)
    ev = L.run_once(mt5, cfg, NOW, sig_fn(1, stop=5.0))
    assert ev["actions"][0]["action"] == "skip" and mt5.positions == []


def test_daily_loss_halts_and_survives_restart(cfg):
    mt5 = FakeMT5()
    L.run_once(mt5, cfg, NOW, sig_fn(1))
    mt5.equity = 9_880.0  # -1.2% on the day
    ev = L.run_once(mt5, cfg, NOW + pd.Timedelta(hours=1), sig_fn(1))
    assert ev["risk"]["breach"] == "daily_loss" and ev["actions"][0]["action"] == "close"
    assert mt5.positions == []
    # New process, same day: still halted, no new trade.
    ev = L.run_once(mt5, cfg, NOW + pd.Timedelta(hours=2), sig_fn(1))
    assert ev["risk"]["halted_until"] == "day" and ev["actions"] == []
    # Next day the halt lifts.
    ev = L.run_once(mt5, cfg, NOW + pd.Timedelta(days=1), sig_fn(1))
    assert ev["risk"]["halted_until"] is None and ev["actions"][0]["action"] == "open"


def test_max_drawdown_is_permanent(cfg):
    mt5 = FakeMT5()
    L.run_once(mt5, cfg, NOW, sig_fn(1))
    mt5.equity = 9_400.0
    L.run_once(mt5, cfg, NOW + pd.Timedelta(hours=1), sig_fn(1))
    ev = L.run_once(mt5, cfg, NOW + pd.Timedelta(days=10), sig_fn(1))
    assert ev["risk"]["halted_until"] == "forever" and mt5.positions == []


def test_real_strategy_runs_on_fake_bars(cfg):
    mt5 = FakeMT5()
    ev = L.run_once(mt5, cfg, NOW)  # real ma_cross_trending on a steady uptrend
    assert ev["signal"] == 1 and ev["actions"][0]["action"] == "open"
