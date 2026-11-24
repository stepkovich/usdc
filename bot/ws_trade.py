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

from binance_common.configuration import (ConfigurationWebSocketAPI,
                                          WebsocketMode)
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

# Транспортные маркеры смерти канала (паттерн newsetka, обкатан живыми
# деньгами): если текст ошибки содержит одно из этих слов — транспорт
# закрыт, дальнейшие вызовы бессмысленны до переподключения
TRANSPORT_MARKERS = [
    "cannot write to closing transport",
    "connection closed",
    "ws closed",
    "connection reset",
    "session is closing",
    "server disconnected",
    "clientconnectorerror",
    "connectionlost",
    "asyncio.timeout",
    "timeouterror",
]


def _is_transport_error(error: Exception) -> bool:
    """Смерть транспорта? (паттерн newsetka — транспорт ниже SDK)."""
    s = str(error).lower()
    return any(m in s for m in TRANSPORT_MARKERS)


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
        # Значения конфигурации — из newsetka (боевые, обкатаны): тайм-аут
        # 10с, переподключение через 3с, перелогин сессии, лимиты в ответах
        config = ConfigurationWebSocketAPI(
            api_key=self.api_key,
            api_secret=self.api_secret,
            stream_url=DERIVATIVES_TRADING_USDS_FUTURES_WS_API_PROD_URL,
            timeout=10000,
            reconnect_delay=3000,
            session_re_logon=True,
            return_rate_limits=True,
        )
        self._client = DerivativesTradingUsdsFutures(config_ws_api=config)
        self._connection = await self._client.websocket_api.create_connection()
        self.healthy = True
        log.info("торговый вебсокет подключён")

    def mark_unhealthy(self, error: Exception | None = None) -> None:
        self.healthy = False
        self._sick_since = time.time()
        if error is not None and _is_transport_error(error):
            log.critical("ws-trade: СМЕРТЬ ТРАНСПОРТА (%s) — канал закрыт "
                         "до переподключения", str(error)[:100])

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
            self.mark_unhealthy(e)
            raise
        result = resp.data().result
        self.orders_sent += 1
        if not self.healthy:
            self.healthy = True
        out = {}
        if result is None:
            return out
        if isinstance(result, dict):
            for k in ("order_id", "orderId", "status", "price", "orig_qty",
                      "origQty", "executed_qty", "executedQty",
                      "client_order_id", "clientOrderId"):
                if result.get(k) is not None:
                    out[k.replace("orderId", "order_id")
                         .replace("origQty", "orig_qty")
                         .replace("executedQty", "executed_qty")
                         .replace("clientOrderId", "client_order_id")] = result[k]
        else:
            for field in ("order_id", "status", "price", "orig_qty",
                          "executed_qty", "client_order_id"):
                v = getattr(result, field, None)
                if v is not None:
                    out[field] = v
        return out

    async def market_order(self, symbol: str, side: str, qty: float) -> dict:
        """Рыночная заявка (для тестов и аварийных закрытий)."""
        if not self.usable() or not self._connection:
            raise RuntimeError("ws-trade: канал нездоров")
        try:
            resp = await self._connection.new_order(
                symbol=symbol, side=_enum(NewOrderSideEnum, side.upper()),
                type=_enum(NewOrderTypeEnum, "MARKET"), quantity=float(qty),
            )
        except Exception as e:
            self.mark_unhealthy(e)
            raise
        result = getattr(resp, "result", None)
        self.orders_sent += 1
        out = {}
        if result is not None:
            for f in ("order_id", "status", "executed_qty", "avg_price"):
                v = getattr(result, f, None)
                if v is not None:
                    out[f] = v
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
            trigger_price=float(stop_price),
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
            self.mark_unhealthy(e)
            raise
        result = resp.data().result
        if not self.healthy:
            self.healthy = True
        out = {}
        if result is None:
            return out
        if isinstance(result, dict):
            for k in ("order_id", "orderId", "algo_id", "algoId", "status",
                      "client_algo_id", "clientAlgoId"):
                if result.get(k) is not None:
                    out[k.replace("algoId", "algo_id")
                         .replace("clientAlgoId", "client_algo_id")] = result[k]
        else:
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
            self.mark_unhealthy(e)
            raise
        result = resp.data().result
        if result is None:
            return None
        if isinstance(result, dict):
            return {"status": result.get("status")}
        return {"status": getattr(result, "status", None)}

    async def cancel_algo_order(self, symbol: str, algo_id) -> dict | None:
        # symbol НЕ пересылаем: SDK-метод /algoOrder.cancel принимает
        # только id/algo_id/client_algo_id (экзамен 24.11 поймал TypeError)
        if not self.usable() or not self._connection:
            raise RuntimeError("ws-trade: канал нездоров")
        try:
            resp = await self._connection.cancel_algo_order(
                algo_id=int(algo_id))
        except Exception as e:
            self.mark_unhealthy(e)
            raise
        result = resp.data().result
        return {"status": getattr(result, "status", None)} if result else None

    async def query_order(self, symbol: str, order_id: int) -> dict | None:
        if not self.usable() or not self._connection:
            raise RuntimeError("ws-trade: канал нездоров")
        resp = await self._connection.query_order(symbol=symbol,
                                                  order_id=int(order_id))
        result = resp.data().result
        if result is None:
            return None
        if isinstance(result, dict):
            return {"status": result.get("status"),
                    "avg_price": result.get("avg_price"),
                    "executed_qty": result.get("executed_qty")}
        return {"status": getattr(result, "status", None),
                "avg_price": getattr(result, "avg_price", None),
                "executed_qty": getattr(result, "executed_qty", None)}

    async def account_information(self) -> dict | None:
        if not self.usable() or not self._connection:
            raise RuntimeError("ws-trade: канал нездоров")
        resp = await self._connection.account_information()
        result = getattr(resp, "result", None)
        return result

    def status(self) -> str:
        return (f"торговый вебсокет: {'здоров' if self.healthy else 'БОЛЕН'} | "
                f"заявок отправлено {self.orders_sent}")
