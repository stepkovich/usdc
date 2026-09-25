"""VPVR-фильтр по объёмному профилю (ПАКЕТ C, кандидат — по умолчанию ВЫКЛ).

Идея из проекта VPVR (папка bybit): пробой через «вакуум» (зона тонкого
объёма) продолжается охотно; пробой В плотный объём (HVN) чаще отбивается —
объём работает магнитом и стеной.

Здесь профиль строится из минутных баров истории (объём бара равномерно
размазывается по его диапазону [low, high]) — это АППРОКСИМАЦИЯ тикового
профиля, зато без дополнительных стримов и памяти. Классификация зоны:
- HVN (high volume node): объём бина >= 2x медианы — плотная зона;
- LVN (low volume node): объём <= 0.3x медианы — вакуум.

Решение по пробою (закрытие бара за Дончианом):
- цена ушла В LVN -> вакуум, пробой разрешаем;
- цена упёрлась В HVN -> стена/магнит, вход пропускаем;
- нейтральная зона -> разрешаем (фильтр только отсекает, не добавляет).

Честность: индикатор не доказан на наших данных — включать только после
бэктеста на готовой истории (edge_lab corpus) и параллельного прогона на демо
(ledger event vpvr_skip).
"""
from __future__ import annotations

from statistics import median
from typing import Iterable, Sequence


def profile_from_bars(bars: Sequence, bins: int = 48) -> dict | None:
    """Профиль объёма по барам (нужны .high, .low, .volume)."""
    usable = [b for b in bars if b.volume and b.volume > 0]
    if len(usable) < 10:
        return None
    lo = min(float(b.low) for b in usable)
    hi = max(float(b.high) for b in usable)
    if hi <= lo:
        return None
    width = (hi - lo) / bins
    hist = [0.0] * bins
    for b in usable:
        blo, bhi = float(b.low), float(b.high)
        vol = float(b.volume)
        if bhi <= blo:
            idx = min(bins - 1, max(0, int((blo - lo) / width)))
            hist[idx] += vol
            continue
        # равномерно размазываем объём бара по покрытым бинам
        i0 = max(0, int((blo - lo) / width))
        i1 = min(bins - 1, int((bhi - lo) / width))
        per = vol / (i1 - i0 + 1)
        for i in range(i0, i1 + 1):
            hist[i] += per
    return {"lo": lo, "hi": hi, "width": width, "hist": hist,
            "total": sum(hist)}


def classify_zone(prof: dict, price: float) -> str:
    """Зона цены в профиле: "hvn" | "lvn" | "mid"."""
    idx = int((price - prof["lo"]) / prof["width"])
    idx = min(len(prof["hist"]) - 1, max(0, idx))
    h = prof["hist"]
    med = median([v for v in h if v > 0]) or 0.0
    if med <= 0:
        return "mid"
    if h[idx] >= 2.0 * med:
        return "hvn"
    if h[idx] <= 0.3 * med:
        return "lvn"
    return "mid"


def breakout_ok(bars: Sequence, price, side: str, bins: int = 48) \
        -> tuple[bool, dict]:
    """Разрешён ли пробой ценой price (закрытие бара за уровнем).
    Возвращает (решение, диагностика для журнала)."""
    prof = profile_from_bars(bars, bins=bins)
    if prof is None:
        return True, {"zone": "no_profile"}
    zone = classify_zone(prof, float(price))
    ok = zone != "hvn"          # HVN = стена: не входим; вакуум/нейтрально — ок
    return ok, {"zone": zone, "bars": len(bars),
                "profile_total": round(prof["total"], 2)}
