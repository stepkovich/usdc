"""Фильтр направления по структуре SMC (15-минутки).

Считает структуру ВЕНДОРЕННОЙ библиотекой (bot/smc_lib.py — копия
smartmoneyconcepts без изменений): swing_highs_lows + bos_choch на
скользящем окне 15-минуток, агрегируемых из наших 1м баров.
bias = направление последнего ИЗВЕСТНОГО слома (BOS/CHoCH); события,
чей свинг ещё не подтверждён (меньше SWING баров назад), не учитываются —
заглядывания в будущее нет. До прогрева (~60 15-минуток) bias=None,
фильтр не режет.

ВАЖНО: numpy/pandas/smc_lib импортируются ЛЕНИВО (внутри _recompute) —
модуль не должен ронять импорт bot.py там, где фильтр выключен и этих
пакетов нет (контейнер реалнета; инцидент 23.09).
"""
from __future__ import annotations

from collections import deque

SWING = 20                       # баров 15м для подтверждения свинга (5 ч)
WINDOW = 600                     # скользящее окно 15м баров (~6 суток)


class Structure15m:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bias: int | None = None
        self._rows: deque = deque(maxlen=WINDOW)     # (ts, o, h, l, c, v)
        self._buf: list[tuple[float, float]] = []    # (high, low) текущей 15-минутки
        self._last_open_ms: int | None = None

    def push_1m(self, open_time_ms: int, high: float, low: float,
                close: float, volume: float, recompute: bool = True) -> None:
        """Одна ЗАКРЫТАЯ минутка; на границе 15 минут — пересчёт структуры.
        recompute=False при заливке истории: пересчёт один раз в конце."""
        self._buf.append((high, low))
        if (open_time_ms // 60000) % 15 != 14:
            return
        h = max(x[0] for x in self._buf)
        l = min(x[1] for x in self._buf)
        self._buf = []
        self._rows.append((open_time_ms - 14 * 60000, float(close),
                           h, l, float(close), volume))
        if recompute:
            self._recompute()

    def _recompute(self) -> None:
        import numpy as np                      # лениво: см. шапку модуля
        import pandas as pd
        from bot.smc_lib import smc as _smc

        n = len(self._rows)
        if n < SWING * 3:
            self.bias = None
            return
        idx = pd.DatetimeIndex(
            pd.to_datetime([r[0] for r in self._rows], unit="ms", utc=True))
        ohlc = pd.DataFrame({"open": [r[1] for r in self._rows],
                             "high": [r[2] for r in self._rows],
                             "low": [r[3] for r in self._rows],
                             "close": [r[4] for r in self._rows],
                             "volume": [r[5] for r in self._rows]}, index=idx)
        swings = _smc.swing_highs_lows(ohlc, swing_length=SWING)
        bos = _smc.bos_choch(ohlc, swings, close_break=True)

        def to_pos(v) -> int | None:
            if isinstance(v, (int, np.integer)):
                return int(v) if 0 <= v < n else None
            try:
                return idx.get_loc(v)
            except KeyError:
                return None

        events: list[tuple[int, int]] = []
        for col in ("BOS", "CHOCH"):
            sub = bos[bos[col].notna()]
            for ts, row in sub.iterrows():
                i = to_pos(ts)
                if i is None:
                    continue
                events.append((i + SWING, 1 if row[col] == 1 else -1))
        if not events:
            self.bias = None
            return
        events.sort()
        cur = None
        for known, d in events:
            if known <= n - 1:
                cur = d
        self.bias = cur
