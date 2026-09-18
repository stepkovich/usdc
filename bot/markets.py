"""Фильтры биржи и Decimal-хелперы. Биржа — единственный источник правды
по шагам цены/количества (exchangeInfo)."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal


@dataclass(frozen=True)
class SymbolFilters:
    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal

    def round_price(self, price: Decimal) -> Decimal:
        """Кратность тику (не просто знаков!): 90.018 при тике 0.01 -> 90.01."""
        return (price / self.tick_size).to_integral_value(rounding=ROUND_DOWN) \
            * self.tick_size

    def round_qty(self, qty: Decimal) -> Decimal:
        return (qty / self.step_size).to_integral_value(rounding=ROUND_DOWN) \
            * self.step_size

    def qty_for_notional(self, notional: Decimal, price: Decimal) -> Decimal | None:
        """Максимальное qty под нотионал или None, если пара не торгуема."""
        raw = self.round_qty(notional / price)
        if raw < self.min_qty:
            return None
        if raw * price < self.min_notional:
            # пробуем следующий шаг вверх — не превышая разумный допуск к нотионалу
            bumped = (raw + self.step_size).quantize(self.step_size, rounding=ROUND_DOWN)
            if bumped * price <= notional * Decimal("1.5"):
                raw = bumped
            else:
                return None
        return raw


def parse_filters(symbol: str, filters: list[dict]) -> SymbolFilters:
    by_type = {f["filterType"]: f for f in filters}
    return SymbolFilters(
        symbol=symbol,
        tick_size=Decimal(by_type["PRICE_FILTER"]["tickSize"]),
        step_size=Decimal(by_type["LOT_SIZE"]["stepSize"]),
        min_qty=Decimal(by_type["LOT_SIZE"]["minQty"]),
        min_notional=Decimal(by_type["MIN_NOTIONAL"]["notional"]),
    )
