"""Demo forward-testing engine for MetaTrader 5 (Phase 4).

Runs one strategy on one pair, once per closed H1 bar:
    fetch closed bars -> strategy signal -> Risk Manager -> place/close orders -> log

Hard rules (constitution.md):
- Refuses to run on anything but a DEMO account.
- The Risk Manager sizes every order (0.25% risk) and halts on daily/weekly loss and
  max drawdown, using the account's real equity. Halt state survives restarts.
- Every order carries a server-side stop-loss, so a crash or sleep cannot leave an
  unprotected position.
- Only positions with this engine's magic number are ever touched.
- Creating a file named STOP in the log folder flattens our positions and stops the engine.
- Every decision is appended to trading_logs/<symbol>_<YYYY-MM>.jsonl.
No LLM is involved anywhere on this path.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from . import data as D
from . import instruments
from .risk import RiskLimits, RiskManager
from .strategies import generate


@dataclass
class LiveConfig:
    symbol: str = "USDJPY"
    broker_symbol: str | None = None
    family: str = "ma_cross_trending"
    params: dict = field(default_factory=lambda: {"er_window": 120, "er_min": 0.15, "stop_atr": 3.0})
    magic: int = 20261126
    history_bars: int = 600
    log_dir: Path = Path("trading_logs")
    limits: RiskLimits = field(default_factory=RiskLimits)
    deviation_points: int = 20
    utc_offset_hours: float | None = None  # None = New York + 7 h rule

    @property
    def name(self) -> str:
        return self.broker_symbol or self.symbol

    @property
    def tag(self) -> str:
        return f"forexlab {self.family}"[:31]


class NotDemoAccount(RuntimeError):
    pass


# --------------------------------------------------------------------------- state & log

def _state_path(cfg: LiveConfig) -> Path:
    return Path(cfg.log_dir) / f"{cfg.symbol}_{cfg.magic}_state.json"


def load_risk(cfg: LiveConfig) -> RiskManager:
    rm = RiskManager(cfg.limits)
    p = _state_path(cfg)
    if p.exists():
        s = json.loads(p.read_text())
        rm.peak, rm.day_start, rm.week_start = s["peak"], s["day_start"], s["week_start"]
        rm.day = datetime.fromisoformat(s["day"]).date() if s["day"] else None
        rm.week = tuple(s["week"]) if s["week"] else None
        rm.halted_until = s["halted_until"]
    return rm


def save_risk(cfg: LiveConfig, rm: RiskManager) -> None:
    p = _state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"peak": rm.peak, "day_start": rm.day_start, "week_start": rm.week_start,
                             "day": rm.day.isoformat() if rm.day else None,
                             "week": list(rm.week) if rm.week else None,
                             "halted_until": rm.halted_until}, indent=2))


def log_event(cfg: LiveConfig, event: dict) -> dict:
    now = datetime.now(timezone.utc)
    event = {"ts": now.isoformat(timespec="seconds"), "symbol": cfg.symbol, "family": cfg.family,
             "params": cfg.params, "magic": cfg.magic, **event}
    p = Path(cfg.log_dir) / f"{cfg.symbol}_{now:%Y-%m}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    return event


# --------------------------------------------------------------------------- MT5 helpers

def require_demo(mt5) -> object:
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError(f"No MT5 account info: {mt5.last_error()}")
    if acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        raise NotDemoAccount(f"Account {acc.login} on {acc.server} is not a DEMO account. "
                             "The forward-tester only runs on demo accounts (constitution rule 2).")
    return acc


def fetch_closed_bars(mt5, cfg: LiveConfig) -> pd.DataFrame:
    """The last `history_bars` CLOSED H1 bars (position 1 skips the forming bar), in UTC."""
    info = mt5.symbol_info(cfg.name)
    rates = mt5.copy_rates_from_pos(cfg.name, mt5.TIMEFRAME_H1, 1, cfg.history_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No bars for {cfg.name}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    idx = D.server_time_to_utc(pd.to_datetime(df["time"], unit="s"), cfg.utc_offset_hours)
    spread = df["spread"].to_numpy() * info.point
    out = pd.DataFrame(index=idx.rename("time"))
    for k in ("open", "high", "low", "close"):
        out[f"bid_{k}"] = df[k].to_numpy()
        out[f"ask_{k}"] = df[k].to_numpy() + spread
    out["volume"] = df["tick_volume"].astype(float).to_numpy()
    return out[D.BAR_COLUMNS]


def our_position(mt5, cfg: LiveConfig):
    for p in mt5.positions_get(symbol=cfg.name) or ():
        if p.magic == cfg.magic:
            return p
    return None


def _send(mt5, request: dict):
    """order_send, falling back through filling modes the broker may require."""
    result = None
    for filling in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
        result = mt5.order_send({**request, "type_filling": filling})
        if result is not None and result.retcode != 10030:  # 10030 = unsupported filling mode
            break
    return result


def _result(r) -> dict:
    if r is None:
        return {"retcode": None, "ok": False}
    return {"retcode": r.retcode, "ok": r.retcode == 10009, "order": getattr(r, "order", None),
            "price": getattr(r, "price", None), "volume": getattr(r, "volume", None),
            "comment": getattr(r, "comment", None)}


def close_position(mt5, cfg: LiveConfig, pos, reason: str) -> dict:
    tick = mt5.symbol_info_tick(cfg.name)
    is_long = pos.type == mt5.POSITION_TYPE_BUY
    r = _send(mt5, {"action": mt5.TRADE_ACTION_DEAL, "symbol": cfg.name, "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket, "price": tick.bid if is_long else tick.ask,
                    "deviation": cfg.deviation_points, "magic": cfg.magic, "comment": cfg.tag,
                    "type_time": mt5.ORDER_TIME_GTC})
    return {"action": "close", "reason": reason, "ticket": pos.ticket, **_result(r)}


def lots_for(rm: RiskManager, equity: float, stop_distance: float, price: float, info, inst) -> float:
    units = rm.size(equity, stop_distance, price, inst)
    lots = units / info.trade_contract_size
    step = info.volume_step
    lots = math.floor(lots / step + 1e-9) * step
    return round(min(lots, info.volume_max), 8)


# --------------------------------------------------------------------------- one cycle

def run_once(mt5, cfg: LiveConfig, now: pd.Timestamp | None = None,
             signal_fn: Callable = generate) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    inst = instruments.get(cfg.symbol)
    acc = require_demo(mt5)
    rm = load_risk(cfg)
    if rm.day is None:
        rm.start(acc.equity, now)

    pos = our_position(mt5, cfg)
    if (Path(cfg.log_dir) / "STOP").exists():
        actions = [close_position(mt5, cfg, pos, "kill_switch")] if pos else []
        done = all(a["ok"] for a in actions)  # keep retrying next hour if the close failed
        return log_event(cfg, {"event": "stopped" if done else "stop_failed", "equity": acc.equity,
                               "actions": actions})

    bars = fetch_closed_bars(mt5, cfg)
    signal, stop = signal_fn(cfg.family, bars, cfg.params)
    want, stop_d = int(signal.iloc[-1]), float(stop.iloc[-1])
    have = 0 if pos is None else (1 if pos.type == mt5.POSITION_TYPE_BUY else -1)
    position_before = have

    rm.on_new_bar(acc.equity, now)
    breach = rm.check(acc.equity, now)
    actions = []
    if pos is not None and (breach or rm.halted_until or want != have):
        reason = breach or (f"halted:{rm.halted_until}" if rm.halted_until else "signal")
        actions.append(close_position(mt5, cfg, pos, reason))
        if actions[-1]["ok"]:
            pos, have = None, 0
    if pos is None and want != 0 and rm.can_trade():
        info = mt5.symbol_info(cfg.name)
        tick = mt5.symbol_info_tick(cfg.name)
        price = tick.ask if want > 0 else tick.bid
        lots = lots_for(rm, mt5.account_info().equity, stop_d, price, info, inst)
        if lots < info.volume_min:
            actions.append({"action": "skip", "reason": f"size {lots} below broker minimum {info.volume_min}"})
        else:
            sl = round(price - want * stop_d, info.digits)
            r = _send(mt5, {"action": mt5.TRADE_ACTION_DEAL, "symbol": cfg.name, "volume": lots,
                            "type": mt5.ORDER_TYPE_BUY if want > 0 else mt5.ORDER_TYPE_SELL,
                            "price": price, "sl": sl, "deviation": cfg.deviation_points,
                            "magic": cfg.magic, "comment": cfg.tag, "type_time": mt5.ORDER_TIME_GTC})
            actions.append({"action": "open", "direction": want, "lots": lots, "sl": sl,
                            "stop_distance": stop_d, **_result(r)})
    save_risk(cfg, rm)
    return log_event(cfg, {
        "event": "bar", "bar_time": str(bars.index[-1]), "bar_close_bid": float(bars["bid_close"].iloc[-1]),
        "signal": want, "position_before": position_before, "equity": acc.equity,
        "balance": acc.balance, "risk": {"halted_until": rm.halted_until, "breach": breach,
                                         "peak": rm.peak, "day_start": rm.day_start},
        "actions": actions,
    })


# --------------------------------------------------------------------------- loop

def run_forever(cfg: LiveConfig, terminal_path: str | None = None, delay_s: int = 20) -> None:
    """Wake shortly after every UTC hour, run one cycle, repeat. Errors are logged, not fatal,
    except a non-demo account, which stops the engine."""
    import MetaTrader5 as mt5  # type: ignore

    while True:
        ok = mt5.initialize(path=terminal_path, timeout=120_000) if terminal_path else mt5.initialize(timeout=120_000)
        try:
            if not ok:
                log_event(cfg, {"event": "error", "error": f"initialize failed: {mt5.last_error()}"})
            else:
                ev = run_once(mt5, cfg)
                print(json.dumps(ev, default=str), flush=True)
                if ev.get("event") == "stopped":
                    return
        except NotDemoAccount as exc:
            log_event(cfg, {"event": "refused", "error": str(exc)})
            raise
        except Exception as exc:  # keep running; positions are protected by server-side stops
            log_event(cfg, {"event": "error", "error": repr(exc)})
        finally:
            mt5.shutdown()
        now = time.time()
        time.sleep(3600 - now % 3600 + delay_s)
