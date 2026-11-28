"""The Risk Manager.

Independent of every strategy: it sizes positions and can veto any trade.
Its limits come from specs/risk_management.md and are not tuned by research.
Strategies never see or change this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .instruments import Instrument


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade: float = 0.0025     # 0.25% of equity lost if the stop is hit
    max_daily_loss: float = 0.01       # 1.0% from the day's starting equity
    max_weekly_loss: float = 0.025     # 2.5% from the week's starting equity
    max_drawdown: float = 0.05         # 5% from peak equity: permanent halt (kill switch)
    max_leverage: float = 10.0         # notional / equity cap
    max_open_positions: int = 1        # per strategy; the portfolio cap of 3 is enforced later


@dataclass
class RiskManager:
    limits: RiskLimits = field(default_factory=RiskLimits)
    peak: float = 0.0
    day_start: float = 0.0
    week_start: float = 0.0
    day: object = None
    week: object = None
    halted_until: str | None = None    # "day", "week", or "forever"
    events: list[dict] = field(default_factory=list)

    def start(self, equity: float, ts: pd.Timestamp) -> None:
        self.peak = self.day_start = self.week_start = equity
        self.day, self.week = ts.date(), ts.isocalendar()[:2]

    def on_new_bar(self, equity: float, ts: pd.Timestamp) -> None:
        """Roll the day and week anchors; lift day/week halts when they expire."""
        if ts.date() != self.day:
            self.day = ts.date()
            self.day_start = equity
            if self.halted_until == "day":
                self.halted_until = None
        wk = ts.isocalendar()[:2]
        if wk != self.week:
            self.week = wk
            self.week_start = equity
            if self.halted_until in ("day", "week"):
                self.halted_until = None

    def check(self, equity: float, ts: pd.Timestamp) -> str | None:
        """Return a breach reason if equity has crossed a limit (caller must flatten)."""
        self.peak = max(self.peak, equity)
        reason = None
        if equity <= self.peak * (1 - self.limits.max_drawdown):
            reason, self.halted_until = "max_drawdown", "forever"
        elif self.halted_until != "forever" and equity <= self.week_start * (1 - self.limits.max_weekly_loss):
            reason, self.halted_until = "weekly_loss", "week"
        elif self.halted_until not in ("forever", "week") and equity <= self.day_start * (1 - self.limits.max_daily_loss):
            reason, self.halted_until = "daily_loss", "day"
        if reason:
            self.events.append({"time": str(ts), "event": reason, "equity": round(equity, 2)})
        return reason

    def can_trade(self) -> bool:
        return self.halted_until is None

    def size(self, equity: float, stop_distance: float, price: float, inst: Instrument,
             rate: float | None = None) -> float:
        """Units of base currency so that hitting the stop loses risk_per_trade of equity.
        Returns 0 (veto) when the trade is not allowed or the stop is invalid."""
        if not self.can_trade() or not (stop_distance > 0):
            return 0.0
        loss_per_unit_usd = inst.to_usd(stop_distance, price, rate)
        units = equity * self.limits.risk_per_trade / loss_per_unit_usd
        max_units = equity * self.limits.max_leverage / inst.notional_usd(1.0, price, rate)
        return float(min(units, max_units))
