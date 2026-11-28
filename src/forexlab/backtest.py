"""Deterministic, cost-aware bar backtester.

Execution model (see specs/execution_rules.md):
- A signal decided at bar t's close is acted on at bar t+1's open.
- Longs buy at the ASK and sell at the BID; shorts the reverse. The spread
  paid is therefore whatever the data says it was, including news widening.
- Slippage is added against us on every market fill and stop fill.
- Commission is charged per side on notional.
- Stops are checked intrabar against the bar's high/low on the side we would
  exit on. A gap through the stop fills at the open, not the stop price.
- Risk limits are checked on bar-close equity; a breach flattens the position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import instruments
from .risk import RiskLimits, RiskManager


@dataclass(frozen=True)
class CostModel:
    commission_per_100k_side: float = 3.5   # USD, typical ECN account (7 USD round turn)
    slippage_pips: float = 0.2              # per fill, against us
    min_spread_pips: float = 0.0            # floor applied to the data's spread
    # Overnight financing per night held, in pips (MT5 convention: negative = you pay).
    # Charged when a position is open through the 21:00 UTC rollover, triple on Wednesdays.
    swap_long_pips: float = 0.0
    swap_short_pips: float = 0.0

    def describe(self) -> dict:
        return {"commission_per_100k_side": self.commission_per_100k_side,
                "slippage_pips": self.slippage_pips, "min_spread_pips": self.min_spread_pips,
                "swap_long_pips": self.swap_long_pips, "swap_short_pips": self.swap_short_pips}


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    risk_events: list[dict]
    halted: bool
    initial_equity: float
    costs: dict = field(default_factory=dict)


def session_of(ts: pd.Timestamp) -> str:
    h = ts.hour
    if h < 7:
        return "asia"
    if h < 12:
        return "london"
    if h < 16:
        return "london_ny_overlap"
    if h < 21:
        return "new_york"
    return "rollover"


def run_backtest(bars: pd.DataFrame, signal: pd.Series, stop: pd.Series, symbol: str,
                 initial_equity: float = 10_000.0, costs: CostModel | None = None,
                 limits: RiskLimits | None = None) -> BacktestResult:
    costs = costs or CostModel()
    inst = instruments.get(symbol)
    risk = RiskManager(limits or RiskLimits())
    slip = costs.slippage_pips * inst.pip
    min_sp = costs.min_spread_pips * inst.pip

    idx = bars.index
    bo, bh, bl, bc = (bars[c].to_numpy() for c in ("bid_open", "bid_high", "bid_low", "bid_close"))
    ao, ah, al, ac = (bars[c].to_numpy() for c in ("ask_open", "ask_high", "ask_low", "ask_close"))
    # Enforce the spread floor by widening the ask side.
    if min_sp > 0:
        ao = np.maximum(ao, bo + min_sp)
        ah = np.maximum(ah, bh + min_sp)
        al = np.maximum(al, bl + min_sp)
        ac = np.maximum(ac, bc + min_sp)
    # USD per unit of quote currency, per bar (only crosses need it from the data).
    qusd = bars["quote_usd"].to_numpy(dtype=float) if "quote_usd" in bars else np.full(len(idx), np.nan)
    if inst.is_cross and not np.isfinite(qusd).all():
        raise ValueError(f"{symbol} bars need a complete quote_usd column (see data.synthetic_cross)")
    sig = signal.reindex(idx).fillna(0).astype(int).to_numpy()
    stp = stop.reindex(idx).to_numpy(dtype=float)

    cash = initial_equity
    pos = None  # dict(dir, units, entry, stop, time, spread_cost, commission, i)
    trades: list[dict] = []
    eq = np.empty(len(idx))

    def usd(amount: float, price: float, i: int) -> float:
        return inst.to_usd(amount, price, qusd[i])

    def commission(units: float, price: float, i: int) -> float:
        return inst.notional_usd(units, price, qusd[i]) / 100_000 * costs.commission_per_100k_side

    def close(i: int, price: float, reason: str, spread_now: float) -> None:
        nonlocal cash, pos
        pnl_quote = (price - pos["entry"]) * pos["units"] * pos["dir"]
        pnl = usd(pnl_quote, price, i)
        comm = commission(pos["units"], price, i)
        cash += pnl - comm
        swap = pos["swap"]  # already in cash
        spread_cost = pos["spread_cost"] + usd(spread_now / 2 * pos["units"], price, i)
        slip_cost = usd(2 * slip * pos["units"], price, i)
        trades.append({
            "entry_time": idx[pos["i"]], "exit_time": idx[i], "direction": pos["dir"],
            "entry": pos["entry"], "exit": price, "units": pos["units"],
            "pnl_usd": pnl - comm - pos["commission"] + swap,
            "swap_usd": swap,
            # Price move at mid, before commission, spread, slippage and swap:
            # pnl_usd = gross_mid_pnl_usd - commission - spread - slippage + swap
            "gross_mid_pnl_usd": pnl + spread_cost + slip_cost,
            "commission_usd": comm + pos["commission"], "spread_cost_usd": spread_cost,
            "slippage_usd": slip_cost,
            "exit_reason": reason, "session": session_of(idx[pos["i"]]),
            "bars_held": i - pos["i"],
        })
        pos = None

    for i in range(len(idx)):
        ts = idx[i]
        if i == 0:
            risk.start(cash, ts)
        risk.on_new_bar(eq[i - 1] if i else cash, ts)
        want = sig[i - 1] if i else 0

        # 1. Signal exit at the open.
        if pos is not None and want != pos["dir"]:
            px = bo[i] - slip if pos["dir"] > 0 else ao[i] + slip
            close(i, px, "signal", ao[i] - bo[i])

        # 2. Entry at the open, sized and approved by the Risk Manager.
        if pos is None and want != 0 and i < len(idx) - 1:
            entry = ao[i] + slip if want > 0 else bo[i] - slip
            units = risk.size(cash, stp[i - 1], entry, inst, qusd[i])
            if units > 0:
                c = commission(units, entry, i)
                cash -= c
                pos = {"dir": want, "units": units, "entry": entry,
                       "stop": entry - want * stp[i - 1], "i": i, "commission": c, "swap": 0.0,
                       "spread_cost": usd((ao[i] - bo[i]) / 2 * units, entry, i)}

        # 2b. Overnight financing for positions carried through the rollover.
        if pos is not None and i > pos["i"] and ts.hour == 21 and (costs.swap_long_pips or costs.swap_short_pips):
            pips = costs.swap_long_pips if pos["dir"] > 0 else costs.swap_short_pips
            nights = 3 if ts.weekday() == 2 else 1
            amount = usd(pips * inst.pip * pos["units"] * nights, bo[i], i)
            cash += amount
            pos["swap"] += amount

        # 3. Intrabar stop.
        if pos is not None:
            if pos["dir"] > 0 and bl[i] <= pos["stop"]:
                close(i, min(pos["stop"], bo[i]) - slip, "stop", ao[i] - bo[i])
            elif pos["dir"] < 0 and ah[i] >= pos["stop"]:
                close(i, max(pos["stop"], ao[i]) + slip, "stop", ao[i] - bo[i])

        # 4. Mark to market at the close and apply risk limits.
        def mark() -> float:
            if pos is None:
                return cash
            px = bc[i] if pos["dir"] > 0 else ac[i]
            return cash + usd((px - pos["entry"]) * pos["units"] * pos["dir"], px, i)

        equity_now = mark()
        breach = risk.check(equity_now, ts)
        if breach and pos is not None:
            px = bc[i] - slip if pos["dir"] > 0 else ac[i] + slip
            close(i, px, breach, ac[i] - bc[i])
            equity_now = cash
        eq[i] = equity_now

    if pos is not None:
        last = len(idx) - 1
        px = bc[last] if pos["dir"] > 0 else ac[last]
        close(last, px, "end_of_data", ac[last] - bc[last])
        eq[last] = cash

    trades_df = pd.DataFrame(trades)
    return BacktestResult(
        equity=pd.Series(eq, index=idx, name="equity"),
        trades=trades_df,
        risk_events=risk.events,
        halted=risk.halted_until == "forever",
        initial_equity=initial_equity,
        costs=costs.describe(),
    )
