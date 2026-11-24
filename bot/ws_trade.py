"""Торговое вебсокет-соединение (ветка ws): размещение заявок, стопов,
снятие и запросы статусов — через подписанный WS-API канал.
REST остаётся запасным транспортом и источником правды для дневного
учёта; повторная отправка исключена: при сбое канала вызывающий код
падает обратно на REST-метод (б logistics в вызывающей стороне).

Соединение: подписанное (API-ключи), переподключение — встроенное
(до 10 попыток) + свой контроль: после N неудач подряд канал помечается
нездоровым и вызывающие коды уходят на REST.
"""
from __future__ import annotations

import logging
import time

from binance_common.configuration import ConfigurationWebSocketAPI, WebsocketMode
from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_WS_API_PROD_URL
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)
from binance_sdk_derivatives_trading_usds_futures.websocket_api.models import (
    NewAlgoOrderAlgoTypeEnum,
    NewAlgoOrderClosePositionEnum,
    NewAlgoOrderSideEnum,
    NewAlgoOrderTypeEnum,
    NewAlgoOrderWorkingTypeEnum,
    NewOrderSideEnum,
    NewOrderTimeInForceEnum,
    NewOrderTypeEnum,
)

log = logging.getLogger("ws_trade")

UNHEALTHY_COOLDOWN_SEC = 60        # после сбоя канал пробуют через минуту


def _enum(enum_cls, name: str):
    """Перечисление по строке; нет такого — исключение вверх."""
    return enum_cls[name]


class WsTradeApi:
    """Подписанный канал размещения/снятия/запроса ордеров."""

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self._client = None
        self._connection = None
        self.healthy = False
        self._sick_since = 0.0
        self.orders_sent = 0

    async def start(self) -> None:
        config = ConfigurationWebSocketAPI(
            api_key=self.api_key,
            api_secret=self.api_secret,
            stream_url=DERIVATIVES_TRADING_USDS_FUTURES_WS_API_PROD_URL,
        )
        self._client = DerivativesTradingUsdsFutures(config_ws_api=config)
        self._connection = await self._client.websocket_api.create_connection()
        self.healthy = True
        log.info("торговый вебсокет подключён")

    def mark_unhealthy(self) -> None:
        self.healthy = False
        self._sick_since = time.time()

    def usable(self) -> bool:
        if self.healthy:
            return True
        # после сбоя даём каналу минуту на восстановление и пробуем снова
        return (self._sick_since
                and time.time() - self._sick_since > UNHEALTHY_COOLDOWN_SEC)

    async def _reconnect(self) -> None:
        try:
            self._connection = await self._client.websocket_api.create_connection()
            self.healthy = True
            log.info("торговый вебсокет переподключён")
        except Exception as e:
            log.warning("торговый вебсокет не переподключился (%s)", str(e)[:80])
            raise

    # ---------- торговые вызовы ----------
    async def new_order(self, symbol: str, side: str, qty: float,
                        price: float | None, client_id: str,
                        post_only: bool = True) -> dict:
        """Лимитная заявка (вход). GTX = только поставщик. Возвращает
        поля результата (order_id, status, executed_qty...)."""
        if not self.usable():
            raise RuntimeError("ws-trade: канал нездоров")
        if not self._connection:
            await self._reconnect()
        try:
            resp = await self._connection.new_order(
                symbol=symbol,
                side=_enum(NewOrderSideEnum, side.upper()),
                type=_enum(NewOrderTypeEnum, "LIMIT"),
                time_in_force=_enum(NewOrderTimeInForceEnum,
                                    "GTX" if post_only else "GTC"),
                quantity=float(qty),
                price=float(price) if price is not None else None,
                new_client_order_id=client_id,
            )
        except Exception as e:
            self.mark_unhealthy()
            raise
        result = getattr(resp, "result", None)
        self.orders_sent += 1
        if not self.healthy:
            self.healthy = True
        out = {}
        if result is not None:
            for field in ("order_id", "status", "price", "orig_qty",
                          "executed_qty", "client_order_id"):
                v = getattr(result, field, None)
                if v is not None:
                    out[field] = v
        return out

    async def new_algo_stop(self, symbol: str, side: str,
                            stop_price: float, client_id: str,
                            qty: float | None = None,
                            position_side: str | None = None) -> dict:
        """Биржевой стоп: условный STOP_MARKET (тот же ордер, что ставит
        боевой REST-путь; closePosition когда qty не задан)."""
        if not self.usable():
            raise RuntimeError("ws-trade: канал нездоров")
        if not self._connection:
            await self._reconnect()
        kwargs = dict(
            algo_type=_enum(NewAlgoOrderAlgoTypeEnum, "CONDITIONAL"),
            symbol=symbol,
            side=_enum(NewAlgoOrderSideEnum, side.upper()),
            type=_enum(NewAlgoOrderTypeEnum, "STOP_MARKET"),
            stop_price=float(stop_price),
            working_type=_enum(NewAlgoOrderWorkingTypeEnum, "CONTRACT_PRICE"),
            client_algo_id=client_id,
        )
        if qty is None:
            try:
                kwargs["close_position"] = _enum(
                    NewAlgoOrderClosePositionEnum, "true")
            except KeyError:
                kwargs["close_position"] = True
        else:
            kwargs["quantity"] = float(qty)
        if position_side:
            kwargs["position_side"] = position_side
        try:
            resp = await self._connection.new_algo_order(**kwargs)
        except Exception as e:
            self.mark_unhealthy()
            raise
        result = getattr(resp, "result", None)
        if not self.healthy:
            self.healthy = True
        out = {}
        if result is not None:
            for field in ("order_id", "algo_id", "status", "client_algo_id"):
                v = getattr(result, field, None)
                if v is not None:
                    out[field] = v
        return out

    async def cancel_order(self, symbol: str, order_id: int) -> dict | None:
        if not self.usable() or not self._connection:
            raise RuntimeError("ws-trade: канал нездоров")
        try:
            resp = await self._connection.cancel_order(symbol=symbol,
                                                       order_id=int(order_id))
        except Exception as e:
            self.mark_unhealthy()
            raise
        result = getattr(resp, "result", None)
        return {"status": getattr(result, "status", None)} if result else None

    async def cancel_algo_order(self, symbol: str, algo_id) -> dict | None:
        if not self.usable() or not self._connection:
            raise RuntimeError("ws-trade: канал нездоров")
        try:
            resp = await self._connection.cancel_algo_order(
                symbol=symbol, algo_id=int(algo_id))
        except Exception as e:
            self.mark_unhealthy()
            raise
        result = getattr(resp, "result", None)
        return {"status": getattr(result, "status", None)} if result else None

    async def query_order(self, symbol: str, order_id: int) -> dict | None:
        if not self.usable() or not self._connection:
            raise RuntimeError("ws-trade: канал нездоров")
        resp = await self._connection.query_order(symbol=symbol,
                                                  order_id=int(order_id))
        result = getattr(resp, "result", None)
        if result is None:
            return None
        return {
            "status": getattr(result, "status", None),
            "avg_price": getattr(result, "avg_price", None),
            "executed_qty": getattr(result, "executed_qty", None),
            "order_id": getattr(result, "order_id", None),
        }

    async def account_information(self) -> dict | None:
        if not self.usable() or not self._connection:
            raise RuntimeError("ws-trade: канал нездоров")
        resp = await self._connection.account_information()
        result = getattr(resp, "result", None)
        return result

    def status(self) -> str:
        return (f"торговый вебсокет: {'здоров' if self.healthy else 'БОЛЕН'} | "
                f"заявок отправлено {self.orders_sent}")
