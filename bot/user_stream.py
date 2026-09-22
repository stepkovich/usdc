"""User-data стрим демо/тестнет/mainnet.

Формы URL у Binance РАЗНЫЕ на разных окружениях (проверено raw-тестом
с контрольной заявкой 22.09: один listenKey, два сокета, события на одном):
- демо/тестнет (fstream.binancefuture.com): доставляет '/ws/<listenKey>',
  SDK-форма '/private/stream?streams=' молчит;
- mainnet (fstream.binance.com): доставляет '/private/stream?streams=<listenKey>'
  (combined-конверт {"stream":..., "data":{...}} — разворачиваем), форма '/ws/'
  молчит (соединение живо, событий ноль).
История: изначально SDK-форма была заменена на '/ws/' ради демо и захардкожена —
на mainnet стрим умер тихо (баг #1, 22.09); '/ws/' на mainnet-хосте — баг #2.
Реконсиляция по REST остаётся источником правды и страховкой."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

import aiohttp

log = logging.getLogger("userstream")

WS_BASE_DEMO = "wss://fstream.binancefuture.com/ws/"
WS_BASE_MAINNET = "wss://fstream.binance.com/private/stream?streams="


class RawUserStream:
    def __init__(self, new_listen_key, keepalive_key, on_event: Callable[[dict], None],
                 ws_base: str):
        """
        new_listen_key: () -> str  (POST listenKey — создание)
        keepalive_key:  () -> None (PUT listenKey  — продление текущего)
        on_event: callable(dict)   — словарь события (ORDER_TRADE_UPDATE, ...)
        ws_base:                   — адрес стрима по режиму (WS_BASE_*)
        """
        if not ws_base:
            raise ValueError("ws_base обязателен: у демо и mainnet разные хосты")
        self._ws_base = ws_base
        self._new_key = new_listen_key
        self._keepalive = keepalive_key
        self._on_event = on_event
        self._stop = asyncio.Event()

    async def run(self) -> None:
        backoff = 5
        while not self._stop.is_set():
            session = aiohttp.ClientSession()
            try:
                key = await asyncio.to_thread(self._new_key)
                async with session.ws_connect(
                        self._ws_base + key, timeout=aiohttp.ClientWSTimeout(ws_close=10),
                        autoping=True) as ws:
                    log.info("user-data WS подключён: %s (ключ хвост %s)",
                             self._ws_base, key[-6:])
                    backoff = 5
                    key_refresh = asyncio.create_task(self._refresh_key_loop())
                    try:
                        while not self._stop.is_set():
                            msg = await ws.receive()
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    d = json.loads(msg.data)
                                    if isinstance(d, dict) and "data" in d:
                                        d = d["data"]   # combined-форма mainnet: конверт
                                    self._on_event(d)
                                except Exception:           # noqa: BLE001
                                    log.exception("user event error")
                            elif msg.type in (aiohttp.WSMsgType.CLOSED,
                                              aiohttp.WSMsgType.ERROR):
                                break
                    finally:
                        key_refresh.cancel()
            except Exception as e:                          # noqa: BLE001
                if self._stop.is_set():
                    break
                log.warning("user-data WS отвалился (%s), переподключение через %ds",
                            e, backoff)
            finally:
                await session.close()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 120)

    async def _refresh_key_loop(self) -> None:
        """listenKey живёт 60 минут — продлеваем (PUT) каждые 30."""
        while True:
            await asyncio.sleep(1800)
            try:
                await asyncio.to_thread(self._keepalive)
                log.info("listenKey продлён")
            except Exception as e:                          # noqa: BLE001
                log.warning("продление listenKey не удалось: %s", e)

    def stop(self) -> None:
        self._stop.set()
