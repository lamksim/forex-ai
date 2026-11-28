"""Static facts about the currency pairs the lab trades."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    pip: float            # price move of one pip
    point_scale: float    # Dukascopy stores prices as integers: price = int / point_scale
    quote: str            # quote currency (P&L is earned in this currency)

    @property
    def is_cross(self) -> bool:
        """Neither side is USD (e.g. EURGBP): converting P&L to USD needs a third rate."""
        return self.quote != "USD" and not self.symbol.startswith("USD")

    def quote_usd(self, price: float, rate: float | None = None) -> float:
        """USD value of one unit of the quote currency. For a cross, `rate` must be given
        (the bar data carries it as the `quote_usd` column)."""
        if self.quote == "USD":
            return 1.0
        if not self.is_cross:
            return 1.0 / price  # USD is the base currency (e.g. USDJPY)
        if rate is None or not rate > 0:
            raise ValueError(f"{self.symbol} is a cross: a quote_usd rate is required")
        return rate

    def to_usd(self, amount_in_quote: float, price: float, rate: float | None = None) -> float:
        """Convert an amount in the quote currency to USD at `price`."""
        return amount_in_quote * self.quote_usd(price, rate)

    def notional_usd(self, units: float, price: float, rate: float | None = None) -> float:
        """USD value of `units` of the base currency."""
        return units * price * self.quote_usd(price, rate)


INSTRUMENTS: dict[str, Instrument] = {
    "EURUSD": Instrument("EURUSD", pip=0.0001, point_scale=1e5, quote="USD"),
    "GBPUSD": Instrument("GBPUSD", pip=0.0001, point_scale=1e5, quote="USD"),
    "USDJPY": Instrument("USDJPY", pip=0.01, point_scale=1e3, quote="JPY"),
    # Crosses. Research builds them from the USD pairs (data.synthetic_cross) until a
    # broker feed is downloaded; their bars carry a `quote_usd` conversion column.
    "EURGBP": Instrument("EURGBP", pip=0.0001, point_scale=1e5, quote="GBP"),
    "EURJPY": Instrument("EURJPY", pip=0.01, point_scale=1e3, quote="JPY"),
    "GBPJPY": Instrument("GBPJPY", pip=0.01, point_scale=1e3, quote="JPY"),
}


def get(symbol: str) -> Instrument:
    try:
        return INSTRUMENTS[symbol.upper()]
    except KeyError as exc:
        raise ValueError(
            f"{symbol} is not an approved pair. Approved: {sorted(INSTRUMENTS)}. "
            "New pairs are added only after they pass validation (constitution rule 2)."
        ) from exc
