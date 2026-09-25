"""Рыночный фид: минутные свечи с БОЕВЫХ стримов (публичные, ключи не нужны)
+ тёплый старт историей с боевого REST. Отдаёт события закрытия бара."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from binance_common.configuration import ConfigurationRestAPI, ConfigurationWebSocketStreams
from binance_common.constants import (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
    WebsocketMode,
)
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
    KlineCandlestickDataIntervalEnum,
)
from binance_sdk_derivatives_trading_usds_futures.websocket_streams.models import (
    IndividualSymbolBookTickerStreamsResponse,
    KlineCandlestickStreamsResponse,
    KlineCandlestickStreamsIntervalEnum,
)

from bot.config import BotConfig

log = logging.getLogger("feed")


@dataclass
class Bar:
    open_time: int          # ms
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    closed: bool = False
    volume: Decimal = Decimal(0)   # базовый объём бара (для VPVR-профиля)


class SymbolHistory:
    """Состояние одной пары: окно баров (Дончиан), ATR-фракция, ADX (Wilder).

    Окна Дончиана и ATR раздельные: Дончиан может смотреть на 12ч (720
    минутных баров), пока ATR остаётся 8ч (480 баров) — как в бэктесте
    развёртки окон 25.09.

    ADX считается инкрементально на барах ЭТОЙ истории (в TF-режиме —
    на TF-барах, т.е. adx_window=14 при TF=15м это ADX(14) на 15-минутках).
    ADX — индикатор, не деньги: внутри float, наружу None до прогрева.
    """

    def __init__(self, symbol: str, window: int, atr_window: int | None = None,
                 adx_window: int = 0):
        self.symbol = symbol
        self.window = window
        self.atr_len = atr_window or window
        self.bars: deque[Bar] = deque(maxlen=window)
        self.trs: deque[Decimal] = deque(maxlen=self.atr_len)
        self.tr_sum = Decimal(0)          # скользящая сумма True Range за ATR-окно
        self.last_price = Decimal(0)
        self.updated_at = 0.0
        # ---- ADX (Wilder), инкрементально ----
        self.adx_len = adx_window
        self._adx_str = 0.0               # сглаженный TR (Wilder-сумма)
        self._adx_spdm = 0.0              # сглаженный +DM
        self._adx_sndm = 0.0              # сглаженный -DM
        self._adx_bars = 0
        self._adx_seed: list[float] = []  # первые adx_len значений DX — сидирование
        self._adx_val: float | None = None
        self._ph = 0.0                    # prev high/low/close (float)
        self._pl = 0.0
        self._pc = 0.0

    def _tr(self, prev_close: Decimal, bar: Bar) -> Decimal:
        if prev_close == 0:
            return bar.high - bar.low
        return max(bar.high - bar.low,
                   abs(bar.high - prev_close), abs(bar.low - prev_close))

    def _push_adx(self, bar: Bar) -> None:
        if not self.adx_len:
            return
        h, l, c = float(bar.high), float(bar.low), float(bar.close)
        if self._pc > 0:
            up, dn = h - self._ph, self._pl - l
            pdm = up if (up > dn and up > 0) else 0.0
            ndm = dn if (dn > up and dn > 0) else 0.0
            tr = max(h - l, abs(h - self._pc), abs(l - self._pc))
            if self._adx_bars < self.adx_len:      # первичное накопление
                self._adx_str += tr
                self._adx_spdm += pdm
                self._adx_sndm += ndm
            else:                                   # сглаживание Уайлдера
                self._adx_str += tr - self._adx_str / self.adx_len
                self._adx_spdm += pdm - self._adx_spdm / self.adx_len
                self._adx_sndm += ndm - self._adx_sndm / self.adx_len
            self._adx_bars += 1
            if self._adx_bars >= self.adx_len and self._adx_str > 0:
                pdi = 100.0 * self._adx_spdm / self._adx_str
                ndi = 100.0 * self._adx_sndm / self._adx_str
                den = pdi + ndi
                dx = 100.0 * abs(pdi - ndi) / den if den > 0 else 0.0
                if len(self._adx_seed) < self.adx_len:
                    self._adx_seed.append(dx)
                    if len(self._adx_seed) == self.adx_len:
                        self._adx_val = sum(self._adx_seed) / self.adx_len
                elif self._adx_val is not None:
                    self._adx_val = (self._adx_val * (self.adx_len - 1)
                                     + dx) / self.adx_len
        self._ph, self._pl, self._pc = h, l, c

    def push_closed(self, bar: Bar) -> None:
        prev_close = self.bars[-1].close if self.bars else Decimal(0)
        tr = self._tr(prev_close, bar)
        if len(self.trs) == self.atr_len:
            self.tr_sum -= self.trs[0]
        self.trs.append(tr)
        self.tr_sum += tr
        self.bars.append(bar)
        self.last_price = bar.close
        self.updated_at = time.time()
        self._push_adx(bar)

    @property
    def ready(self) -> bool:
        return len(self.bars) >= self.window

    @property
    def adx(self) -> float | None:
        """ADX(Wilder) по барам этой истории; None пока не прогрето."""
        return self._adx_val

    @property
    def atr_frac(self) -> Decimal:
        """ATR(окно) как доля цены; вырожденные данные -> ValueError."""
        if not self.ready:
            raise ValueError(f"{self.symbol}: история не прогрета")
        if self.tr_sum == 0 or self.bars[-1].close == 0:
            # замороженный/неликвидный контракт: нулевой диапазон всех баров
            raise ValueError(f"{self.symbol}: нулевой ATR (контракт заморожен?)")
        return self.tr_sum / Decimal(len(self.trs)) / self.bars[-1].close

    def donchian(self) -> tuple[Decimal, Decimal]:
        """(max high, min low) за окно, исключая последний закрытый бар
        (как в бэктесте: shift(1))."""
        if not self.ready:
            raise ValueError(f"{self.symbol}: история не прогрета")
        b = list(self.bars)[:-1]
        return (max(x.high for x in b), min(x.low for x in b))


class MarketFeed:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.hist: dict[str, SymbolHistory] = {}
        self.market_rest = DerivativesTradingUsdsFutures(
            config_rest_api=ConfigurationRestAPI(
                api_key=cfg.api_key, api_secret=cfg.api_secret,
                base_path="https://fapi.binance.com"))
        self._streams_client: DerivativesTradingUsdsFutures | None = None
        self._conn = None
        self.last_message_ts = 0.0
        self.book: dict[str, tuple[Decimal, Decimal]] = {}   # sym -> (bid, ask)
        self.forming: dict[str, tuple[Decimal, Decimal, Decimal, Decimal]] = {}  # o,h,l,c
        self._queue: asyncio.Queue[tuple[str, Bar]] = asyncio.Queue(maxsize=10000)

    # ---------- тёплый старт ----------
    async def warmup(self, symbols: list[str]) -> None:
        window = self.cfg.atr_window
        need = max(self.cfg.warmup_bars, window + 10)
        for chunk_start in range(0, len(symbols), 8):
            chunk = symbols[chunk_start:chunk_start + 8]
            results = await asyncio.gather(
                *[asyncio.to_thread(self._fetch_klines, s, need) for s in chunk],
                return_exceptions=True)
            for s, res in zip(chunk, results):
                if isinstance(res, Exception):
                    log.error("warmup %s: %s", s, res)
                    continue
                self.hist[s] = res
            await asyncio.sleep(0.5)
        log.info("тёплый старт: %d/%d пар прогрето", len(self.hist), len(symbols))

    def _fetch_klines(self, symbol: str, limit: int) -> SymbolHistory:
        h = SymbolHistory(symbol, self.cfg.donchian_bars, self.cfg.atr_window,
                          adx_window=self.cfg.adx_window)
        resp = self.market_rest.rest_api.kline_candlestick_data(
            symbol=symbol,
            interval=KlineCandlestickDataIntervalEnum["INTERVAL_1m"].value,
            limit=limit)
        rows = getattr(resp.data(), "root", None) or resp.data()
        rows = list(rows)[:-1]                # последний бар ещё не закрыт
        for r in rows:                        # [open_time, o, h, l, c, v, ...]
            h.push_closed(Bar(int(r[0]), Decimal(str(r[1])), Decimal(str(r[2])),
                              Decimal(str(r[3])), Decimal(str(r[4])), True,
                              volume=Decimal(str(r[5] or 0))))
        return h

    # ---------- живые стримы ----------
    async def start_streams(self, symbols: list[str]) -> None:
        # перезапуск: сначала честно закрыть старое соединение
        if self._conn is not None:
            try:
                await self._conn.close_connection(close_session=True)
            except Exception:                    # noqa: BLE001
                pass
            self._conn = None
        self.last_message_ts = time.time()   # иначе watchdog ругается до первого бара
        self._streams_client = DerivativesTradingUsdsFutures(
            config_ws_streams=ConfigurationWebSocketStreams(
                stream_url=self.cfg.market_streams_url,
                mode=WebsocketMode.POOL, pool_size=4,
                reconnect_attempts=10, reconnect_delay=5000))
        self._conn = await self._streams_client.websocket_streams.create_connection()
        # батч-подписка: один SUBSCRIBE с массивом имён (лимит биржи: 200 стримов
        # на соединение); базовый subscribe группирует их по соединениям пула
        names = [f"{s.lower()}@kline_1m" for s in symbols]
        await self._conn.subscribe(
            streams=names,
            response_model=KlineCandlestickStreamsResponse,
            stream_url="market")
        for name in names:
            self._conn.on("message", self._on_kline, name)
        # bookTicker: лучший bid/ask каждой пары — спред в момент сигнала
        bnames = [f"{s.lower()}@bookTicker" for s in symbols]
        await self._conn.subscribe(
            streams=bnames,
            response_model=IndividualSymbolBookTickerStreamsResponse,
            stream_url="market")
        for name in bnames:
            self._conn.on("message", self._on_book, name)
        log.info("стримы подписаны одним запросом: %d пар (+bookTicker)", len(names))

    def _on_kline(self, model) -> None:
        try:
            d = model.model_dump(by_alias=True) if hasattr(model, "model_dump") else dict(model)
            k = d.get("k") or {}
            self.last_message_ts = time.time()   # живость — по любому сообщению
            if not k:
                return
            sym = d["s"]
            # формирующийся бар: экстремумы текущей минуты (для протрузии)
            self.forming[sym] = (Decimal(k["o"]), Decimal(k["h"]),
                                 Decimal(k["l"]), Decimal(k["c"]))
            if not k.get("x"):
                return                       # бар ещё формируется
            bar = Bar(int(k["t"]), Decimal(k["o"]), Decimal(k["h"]),
                      Decimal(k["l"]), Decimal(k["c"]), True,
                      volume=Decimal(str(k.get("v") or 0)))
            if sym in self.hist:
                self.hist[sym].push_closed(bar)
                self._queue.put_nowait((sym, bar))
        except Exception:                    # noqa: BLE001 — фид не должен падать
            log.exception("kline parse error")

    def _on_book(self, model) -> None:
        try:
            d = model.model_dump(by_alias=True) if hasattr(model, "model_dump") else dict(model)
            inst = d.get("actual_instance")
            if isinstance(inst, dict):
                d = inst
            sym, bid, ask = d.get("s"), d.get("b"), d.get("a")
            if sym and bid and ask:
                self.book[sym] = (Decimal(bid), Decimal(ask))
                self.last_message_ts = time.time()
        except Exception:                    # noqa: BLE001
            pass                              # bookTicker высокочастотный — не логируем

    async def closed_bars(self) -> tuple[str, Bar]:
        return await self._queue.get()

    def stale(self, limit_s: float = 180) -> bool:
        return time.time() - self.last_message_ts > limit_s
