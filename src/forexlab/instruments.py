"""Static facts about the currency pairs the lab trades."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    pip: float            # price move of one pip
    point_scale: float    # Dukascopy stores prices as integers: price = int / point_scale
    quote: str            # quote currency (P&L is earned in this currency)

    def to_usd(self, amount_in_quote: float, price: float) -> float:
        """Convert an amount in the quote currency to USD at `price`."""
        if self.quote == "USD":
            return amount_in_quote
        # USD is the base currency (e.g. USDJPY): divide by the rate.
        return amount_in_quote / price

    def notional_usd(self, units: float, price: float) -> float:
        """USD value of `units` of the base currency."""
        if self.quote == "USD":
            return units * price
        return units  # base currency is USD


INSTRUMENTS: dict[str, Instrument] = {
    "EURUSD": Instrument("EURUSD", pip=0.0001, point_scale=1e5, quote="USD"),
    "GBPUSD": Instrument("GBPUSD", pip=0.0001, point_scale=1e5, quote="USD"),
    "USDJPY": Instrument("USDJPY", pip=0.01, point_scale=1e3, quote="JPY"),
}


def get(symbol: str) -> Instrument:
    try:
        return INSTRUMENTS[symbol.upper()]
    except KeyError as exc:
        raise ValueError(
            f"{symbol} is not an approved pair. Approved: {sorted(INSTRUMENTS)}. "
            "New pairs are added only after they pass validation (constitution rule 2)."
        ) from exc
