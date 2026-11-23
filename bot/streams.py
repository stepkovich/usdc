"""Потоковый слой бота (ветка ws, 23.11): все рыночные данные и события
пользователя приходят по вебсокетам через официальный SDK
binance-sdk-derivatives-trading-usds-futures (транспорт — binance-connector).
REST остаётся источником правды для заявок, сверки и дневного учёта;
поток — для данных в реальном времени.

Три канала (через клиент SDK, режим POOL — 3 соединения):
 1) свечи 30м всех монет — пачечная подписка одним SUBSCRIBE
    на соединение (лимит биржи 200 потоков на соединение);
 2) лучшие цены всего рынка — один поток !bookTicker;
 3) пользовательский поток по listenKey — исполнение заявок
    (ORDER_TRADE_UPDATE) и изменение баланса (ACCOUNT_UPDATE);
    ключ продлевается REST-запросом каждые 30 минут (SDK сам не продлевает).

Сторожа: тишина рынка дольше 3 минут -> переподписка
(встроенное переподключение коннектора — первая линия обороны).
"""
from __future__ import annotations

import asyncio
import logging
import time

import aiohttp

from binance_common.configuration import ConfigurationWebSocketStreams, WebsocketMode
from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)
from binance_sdk_derivatives_trading_usds_futures.websocket_streams.models import (
    KlineCandlestickStreamsIntervalEnum,
)

log = logging.getLogger("streams")

BAR_MS = 30 * 60_000
SILENCE_RECONNECT_SEC = 180       # нет ни одного события 3 мин — пересборка
BOOK_STALE_SEC = 120              # лучшие цены старше 2 мин — не верим
LISTENKEY_KEEPALIVE_SEC = 1800    # продление ключа раз в 30 минут
POOL_SIZE = 3                     # 515 потоков / 200 на соединение


class MarketStreams:
    """Свечи 30м и лучшие цены по всем монетам через вебсокеты."""

    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.bars: dict[str, dict] = {}       # sym -> последний закрытый бар
        self.live_bar: dict[str, dict] = {}   # sym -> текущая незакрытая свеча
        self.book: dict[str, tuple[float, float]] = {}  # sym -> (bid, ask)
        self.book_ts = 0.0
        self.last_event = 0.0
        self.events_total = 0
        self.closed_seen = 0
        self._handles: list = []
        self._client = None
        self._healthy = False

    # ---------- обработчики событий ----------
    def _kline_handler(self, sym: str):
        """Замыкание: события потока одной монеты."""
        def handler(data) -> None:
            try:
                if hasattr(data, "model_dump"):
                    data = data.model_dump(by_alias=True)
                k = data.get("k") if isinstance(data, dict) else None
                if not isinstance(k, dict):
                    return
                self.last_event = time.time()
                self.events_total += 1
                bar = {
                    "open_time": int(k.get("t") or 0),
                    "open": float(k.get("o") or 0),
                    "high": float(k.get("h") or 0),
                    "low": float(k.get("l") or 0),
                    "close": float(k.get("c") or 0),
                    "volume": float(k.get("v") or 0),
                    "taker_buy_base": float(k.get("V") or 0),
                }
                if bool(k.get("x")):
                    # свеча закрылась: храним как последний закрытый бар
                    if self.bars.get(sym, {}).get("open_time") != bar["open_time"]:
                        self.closed_seen += 1
                    self.bars[sym] = bar
                else:
                    self.live_bar[sym] = bar
            except Exception:
                log.exception("kline %s: событие не обработано", sym)
        return handler

    def _on_book_event(self, data) -> None:
        try:
            if hasattr(data, "model_dump"):
                data = data.model_dump(by_alias=True)
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sym = row.get("s")
                bid = float(row.get("b") or 0)
                ask = float(row.get("a") or 0)
                if sym and bid > 0 and ask > 0:
                    self.book[sym] = (bid, ask)
            self.last_event = time.time()
            self.book_ts = time.time()
        except Exception:
            log.exception("book: событие не обработано")

    # ---------- события соединений ----------
    def _conn_event(self, event: str):
        def handler(*args) -> None:
            log.warning("соединение потоков: %s %s", event, args[:1])
            if event in ("reconnect", "close"):
                self._healthy = False
                asyncio.ensure_future(self._resubscribe_if_silent())
        return handler

    async def _resubscribe_if_silent(self) -> None:
        """Дать встроенному переподключению до 3 минут; если данных нет —
        переподписаться целиком."""
        await asyncio.sleep(SILENCE_RECONNECT_SEC)
        if self.last_event and time.time() - self.last_event < SILENCE_RECONNECT_SEC:
            return
        log.warning("потоки: тишина %d сек — переподписка заново",
                    SILENCE_RECONNECT_SEC)
        try:
            ws = self._client.websocket_streams
            streams = [f"{s.lower()}@kline_30m" for s in self.symbols]
            await ws.subscribe(streams=streams, stream_url="market")
            self._healthy = True
            log.info("потоки: переподписаны все %d", len(streams))
        except Exception:
            log.exception("переподписка не удалась")

    # ---------- жизнь ----------
    async def start(self) -> None:
        config = ConfigurationWebSocketStreams(
            stream_url=DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
            mode=WebsocketMode.POOL,
            pool_size=POOL_SIZE,
        )
        self._client = DerivativesTradingUsdsFutures(config_ws_streams=config)
        ws = self._client.websocket_streams
        await ws.create_connection()          # пул: market/public/private
        for event in ("reconnect", "close", "error"):
            ws.on_connection(event, self._conn_event(event))

        streams = [f"{s.lower()}@kline_30m" for s in self.symbols]
        await ws.subscribe(streams=streams, stream_url="market")
        for name in streams:
            sym = name.split("@")[0].upper()
            ws.on("message", self._kline_handler(sym), stream=name)
        book = await ws.all_book_tickers_stream()
        book.on("message", self._on_book_event)
        self._handles.append(book)

        self.last_event = time.time()
        self._healthy = True
        log.info("потоки подписаны: %d свечей + лучшие цены (пул %d)",
                 len(streams), POOL_SIZE)
        asyncio.ensure_future(self._silence_watchdog())

    async def _silence_watchdog(self) -> None:
        while True:
            await asyncio.sleep(60)
            if self.last_event and \
                    time.time() - self.last_event > SILENCE_RECONNECT_SEC:
                log.warning("потоки: тишина %d сек — переподписка",
                            SILENCE_RECONNECT_SEC)
                self._healthy = False
                await self._resubscribe_if_silent()
                self.last_event = time.time()

    # ---------- запросы к слою ----------
    def healthy(self) -> bool:
        return self._healthy and (not self.last_event
                                  or time.time() - self.last_event < 300)

    def closed_bar(self, sym: str, open_ms: int):
        """Последний закрытый бар монеты, ЕСЛИ он открылся ровно в
        open_ms (ожидание скана: бар, закрывшийся на последней границе).
        None — бар не пришёл (данные не дошли — скан дозапросит REST)."""
        bar = self.bars.get(sym)
        if bar and bar["open_time"] == open_ms:
            return bar
        return None

    def bid_ask(self, sym: str):
        """Лучшие цены из потока; устаревшие — None (пусть зовут REST)."""
        ba = self.book.get(sym)
        if ba and time.time() - self.book_ts < BOOK_STALE_SEC:
            return ba
        return None

    def status(self) -> str:
        return (f"потоки: {'здоров' if self.healthy() else 'ПРОБЛЕМА'} | "
                f"событий {self.events_total:,} | закрытых баров "
                f"{self.closed_seen:,} | монет с барами {len(self.bars)}")


class UserStreams:
    """Пользовательский поток: исполнение заявок и баланс по listenKey.
    SDK ключ не продлевает — продление REST-запросом каждые 30 минут."""

    def __init__(self, api_key: str, rest_base: str):
        self.api_key = api_key
        self.rest_base = rest_base
        self.listen_key: str | None = None
        self.last_event = 0.0
        self.on_order_update = None    # колбэк(словарь события заявки)
        self.on_balance_update = None  # колбэк(баланс USDT float)
        self._client = None

    async def _new_listen_key(self) -> str | None:
        url = f"{self.rest_base}/fapi/v1/listenKey"
        try:
            async with aiohttp.ClientSession() as s:
                async with s.put(url, headers={
                    "X-MBX-APIKEY": self.api_key}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("listenKey")
                    log.warning("listenKey: биржа ответила %d", resp.status)
        except Exception:
            log.exception("listenKey: не получен")
        return None

    async def start(self) -> None:
        self.listen_key = await self._new_listen_key()
        if not self.listen_key:
            log.warning("пользовательский поток: ключа нет — сделки будут "
                        "подтверждаться REST-опросом, как раньше")
            return
        config = ConfigurationWebSocketStreams(
            stream_url=DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
        )
        self._client = DerivativesTradingUsdsFutures(config_ws_streams=config)
        ws = self._client.websocket_streams
        await ws.create_connection()
        handle = await ws.user_data(self.listen_key)
        handle.on("message", self._on_user_event)
        asyncio.ensure_future(self._keepalive_loop())
        log.info("пользовательский поток подключён")

    async def _keepalive_loop(self) -> None:
        url = f"{self.rest_base}/fapi/v1/listenKey"
        while True:
            await asyncio.sleep(LISTENKEY_KEEPALIVE_SEC)
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.put(url, headers={
                        "X-MBX-APIKEY": self.api_key}) as resp:
                        log.info("listenKey: продлён (%d)", resp.status)
            except Exception:
                log.exception("listenKey: продление не удалось")

    def _on_user_event(self, data) -> None:
        try:
            self.last_event = time.time()
            if hasattr(data, "model_dump"):
                data = data.model_dump(by_alias=True)
            if not isinstance(data, dict):
                return
            etype = data.get("e")
            if etype == "ORDER_TRADE_UPDATE" and self.on_order_update:
                o = data.get("o", {})
                self.on_order_update({
                    "order_id": o.get("i"),
                    "client_id": o.get("c"),
                    "symbol": o.get("s"),
                    "side": o.get("S"),
                    "status": o.get("X"),
                    "price": o.get("L"),          # цена последнего исполнения
                    "filled_qty": o.get("z"),     # исполнено всего
                    "time": data.get("T"),
                })
            elif etype == "ACCOUNT_UPDATE" and self.on_balance_update:
                for a in data.get("a", {}).get("B", []):
                    if a.get("a") == "USDT":
                        self.on_balance_update(float(a.get("wb", 0) or 0))
        except Exception:
            log.exception("user: событие не обработано")

    def status(self) -> str:
        if not self.listen_key:
            return "пользовательский поток: НЕТ (REST-опрос)"
        age = time.time() - self.last_event if self.last_event else None
        return ("пользовательский поток: жив" +
                (f", событие {age:.0f}с назад" if age is not None else ""))
