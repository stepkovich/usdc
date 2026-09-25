"""Честная модель исполнения лимиток для бэктестов (ПАКЕТ B).

Урок проекта: бэктесты с "коснулся = исполнилось" завышают результаты
(гэп-кредит сквозь лимитку дал +21.7% на одном баре ZEC). Здесь — модель
touch-through: лимитка исполняется, только если цена ПРОШЛА сквозь её
уровень на touch_through тиков, а не просто коснулась его.

Правила (консервативные, в духе MFI/2tf-симуляторов):
- вход SELL-лимитка (шорт) на уровне L исполняется, если low бара <= L - thr;
  BUY-лимитка (лонг) — если high бара >= L + thr; thr = touch_through * tick;
- маркитабельная заявка (уровень уже за рынком на открытии бара) исполняется
  по max/min(open, level) — честная тейкерская цена, без гэп-кредита;
- same-bar TP на входном баре не засчитывается (строгая конвенция):
  вход и выход в одном баре неразличимы по порядку;
- стоп-приоритет внутри бара: если и стоп, и TP достижимы в одном баре,
  считаем исполненным СТОП (худший случай).

Только Decimal, без внешних зависимостей — модуль импортируется и ботом
(для справки), и исследовательскими движками (scalp_research).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class FillDecision:
    filled: bool
    price: Decimal          # честная цена исполнения
    kind: str               # "maker" | "taker" | "none"
    note: str = ""


def limit_fill_touch_through(bar_open: Decimal, bar_high: Decimal,
                             bar_low: Decimal, level: Decimal,
                             side: str, tick: Decimal,
                             touch_through: int = 1) -> FillDecision:
    """Исполнение ПАССИВНОЙ лимитки внутри бара по модели touch-through.

    side: "BUY" (лонг, лимитка НИЖЕ рынка — цену сбивают к нам) или
    "SELL" (шорт, лимитка ВЫШЕ рынка — цену поднимают к нам).
    thr = touch_through * tick — цена должна ПРОЙТИ сквозь уровень:
    BUY-лимитка исполняется, если min бара <= level - thr (прошли вниз),
    SELL-лимитка — если max бара >= level + thr (прошли вверх).
    Касание без проторговки (low == level у BUY) НЕ исполняется.
    Если уровень уже за рынком на открытии (маркитабельная заявка) —
    исполнение по open (тейкер), не по уровню (без гэп-кредита).
    """
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side={side}: ждём BUY/SELL")
    thr = Decimal(touch_through) * tick
    if side == "BUY":
        if bar_open <= level:
            # уже за рынком: тейкер по open — исполнение по рынку, не по уровню
            return FillDecision(True, bar_open, "taker",
                                "marketable at open")
        if bar_low <= level - thr:
            return FillDecision(True, level, "maker", "touch-through ok")
        return FillDecision(False, level, "none",
                            f"low {bar_low} > level-thr {level - thr}")
    # SELL
    if bar_open >= level:
        return FillDecision(True, bar_open, "taker", "marketable at open")
    if bar_high >= level + thr:
        return FillDecision(True, level, "maker", "touch-through ok")
    return FillDecision(False, level, "none",
                        f"high {bar_high} < level+thr {level + thr}")


def same_bar_exit_allowed(fill_index: str = "strict") -> bool:
    """Конвенция same-bar выхода: strict — входной бар не даёт выхода
    (порядок неизвестен); standard — TP на входном баре возможен.
    По нашему опыту разница конвенций качает итог в разы — фиксируем явно."""
    return fill_index == "standard"


def stop_priority() -> str:
    """Если в баре достижимы и стоп, и TP — верим СТОПУ (худший случай).
    Возвращает метку конвенции для отчётов."""
    return "stop"
