"""Исполнитель: ордера на демо (REST), режим позиции (хедж/оневей), плечо,
реконсиляция. Биржа — единственный источник правды."""
from __future__ import annotations

import asyncio
import json
import time
import logging
from decimal import Decimal

from decimal import ROUND_DOWN, ROUND_UP

from binance_common.configuration import ConfigurationRestAPI
from binance_common.errors import Error as BinanceError
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)
from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
    NewAlgoOrderAlgoTypeEnum,
    NewAlgoOrderSideEnum,
    NewAlgoOrderClosePositionEnum,
    NewAlgoOrderTypeEnum,
    NewAlgoOrderWorkingTypeEnum,
    NewOrderSideEnum,
    NewOrderTimeInForceEnum,
    NewOrderTypeEnum,
    NewOrderReduceOnlyEnum,
)

from bot.markets import SymbolFilters

log = logging.getLogger("executor")


def api_code(e: Exception) -> int | None:
    """Код ошибки биржи (-4067, -2011, ...): у SDK он лежит в .status_code."""
    sc = getattr(e, "status_code", None)
    if isinstance(sc, int) and sc < 0:
        return sc
    msg = str(e)
    try:
        body = msg[msg.index("{"):msg.rindex("}") + 1]
        return int(json.loads(body).get("code"))
    except Exception:
        return None


def unwrap(model):
    """Автогенерированные one-of ответы: реальные поля в .actual_instance."""
    inst = getattr(model, "actual_instance", None)
    return inst if inst is not None else model


class Executor:
    def __init__(self, cfg, filters: dict[str, SymbolFilters]):
        self.cfg = cfg
        self.filters = filters
        self.hedge_mode = False            # подтверждается в ensure_position_mode()
        self.leverage_set: dict[str, int] = {}
        self.client = DerivativesTradingUsdsFutures(
            config_rest_api=ConfigurationRestAPI(
                api_key=cfg.api_key, api_secret=cfg.api_secret,
                base_path=cfg.exec_rest_url,
                timeout=5000, retries=3, backoff=1000))
        # РЫНОЧНЫЕ данные — всегда мейннет (публичные): свечи и стакан
        # для признаков/решений должны совпадать с обучением на реале.
        # Исполнение (ордера/позиции/баланс) — по режиму (демо).
        self.market_client = DerivativesTradingUsdsFutures(
            config_rest_api=ConfigurationRestAPI(
                api_key=cfg.api_key, api_secret=cfg.api_secret,
                base_path="https://fapi.binance.com",
                timeout=5000, retries=3, backoff=1000))

    def _run(self, fn, *a, **kw):
        return asyncio.to_thread(fn, *a, **kw)

    def _pside(self, side: str) -> str | None:
        """Hedge mode: LONG/SHORT (сторона позиции). One-way: параметр не шлём."""
        if not self.hedge_mode:
            return None
        return "LONG" if side == "BUY" else "SHORT"

    # ---------- режим позиции ----------
    def ensure_position_mode(self, want_hedge: bool = True) -> bool:
        r = self.client.rest_api.get_current_position_mode().data()
        current = bool(r.dual_side_position)
        if current == want_hedge:
            self.hedge_mode = current
            log.info("режим позиции: %s (уже установлен)",
                     "HEDGE" if current else "ONE-WAY")
            return current
        try:
            self.client.rest_api.change_position_mode(
                dual_side_position="true" if want_hedge else "false")
            self.hedge_mode = want_hedge
            log.info("режим позиции переключён: %s",
                     "HEDGE" if want_hedge else "ONE-WAY")
        except BinanceError as e:
            code = api_code(e)
            if code in (-4067, -4068):     # есть открытые заявки/позиции
                self.hedge_mode = current
                log.warning("сменить режим нельзя (%s): есть позиции/заявки. "
                            "Работаем в текущем режиме %s",
                            code, "HEDGE" if current else "ONE-WAY")
            else:
                raise
        return self.hedge_mode

    # ---------- плечо ----------
    def verify_and_set_leverage(self, symbols: list[str]) -> None:
        lev = int(getattr(self.cfg, "leverage", 3) or 3)
        conf = self.client.rest_api.futures_account_configuration().data()
        conf_d = conf.model_dump(by_alias=True) if hasattr(conf, "model_dump") else {}
        current = {p.get("symbol"): int(p.get("leverage", 0))
                   for p in conf_d.get("positions", [])}
        for s in symbols:
            if current.get(s) == lev:
                self.leverage_set[s] = lev
                continue
            try:
                self.client.rest_api.change_initial_leverage(
                    symbol=s, leverage=lev)
                self.leverage_set[s] = lev
            except BinanceError as e:
                code = api_code(e)
                log.error("плечо %s -> %sx не установлено (код %s): %s",
                          s, lev, code, e)
                if s in current:
                    self.leverage_set[s] = current[s]
            except Exception as e:
                log.error("плечо %s: %s", s, e)
        log.info("плечо: установлено/подтверждено для %d/%d пар",
                 len(self.leverage_set), len(symbols))

    def setup_symbol(self, symbol: str) -> None:
        try:
            self.client.rest_api.change_margin_type(
                symbol=symbol, margin_type="ISOLATED")
        except BinanceError as e:
            if api_code(e) != -4046:       # -4046: уже изолированная
                log.warning("margin_type %s: %s", symbol, e)
        except Exception as e:
            log.warning("margin_type %s: %s", symbol, e)

    # ---------- ордера ----------
    def _grid(self, symbol: str, price, side: str):
        """Цена строго на тик-сетке монеты: BUY вниз, SELL вверх
        (не пересекать спред). Лечит -4014 'not increased by tick size'."""
        fl = self.filters.get(symbol)
        if fl is None or price is None:
            return price
        d = price if isinstance(price, Decimal) else Decimal(str(price))
        tick = getattr(fl, "tick_size", None) or Decimal("0.0001")
        q = (d / tick).to_integral_value(
            rounding=ROUND_DOWN if side == "BUY" else ROUND_UP) * tick
        return q if q > 0 else d

    @staticmethod
    def _plain(v) -> str:
        """Обычная десятичная запись: 9.8e-06 -> '0.0000098'. Биржа
        отвергает «научную запись» в параметрах (-1102 malformed)."""
        d = v if isinstance(v, Decimal) else Decimal(str(v))
        return format(d, "f")

    def place_entry_limit(self, symbol: str, side: str, price: Decimal,
                          qty: Decimal, client_id: str,
                          post_only: bool = False) -> int | None:
        """Входная лимитка. post_only=True -> GTX: биржа ОТКЛОНИТ заявку,
        если она сразу исполнилась бы как тейкер (-4131), — тогда вернём None.
        Так вход гарантированно мейкерский (на USDC-M мейкер = 0 по промо)."""
        tif = "GTX" if post_only else "GTC"
        try:
            r = self.client.rest_api.new_order(
                symbol=symbol,
                side=NewOrderSideEnum[side].value,
                type=NewOrderTypeEnum["LIMIT"].value,
                position_side=self._pside(side),
                time_in_force=NewOrderTimeInForceEnum[tif].value,
                quantity=self._plain(qty),
                price=self._plain(self._grid(symbol, price, side)),
                new_client_order_id=client_id)
        except BinanceError as e:
            if post_only and api_code(e) in (-4131, -5022):
                # не мейкер (цена ушла сквозь уровень): демо отвечает -5022,
                # прод -4131. Вход тейкером не делаем — сигнал устарел.
                log.info("%s: GTX-вход отклонён (%s) — цена ушла, пропускаем",
                         symbol, api_code(e))
                return None
            raise
        oid = int(r.data().order_id)
        log.info("заявка %s %s %s @%s id=%s (tif=%s)", symbol, side, qty, price,
                 oid, tif)
        return oid

    def place_tp_limit(self, symbol: str, side: str, price: Decimal,
                       qty: Decimal, client_id: str,
                       pos_side: str | None = None) -> int | None:
        # -2022: гонка сразу после исполнения входа — ретраим с паузой
        # pos_side = сторона ПОЗИЦИИ (LONG/SHORT), не ордера.
        # В HEDGE: positionSide = сторона позиции (закрываем LONG -> positionSide=LONG)
        # В ONE-WAY: reduceOnly=true, positionSide не шлём
        for attempt in range(4):
            try:
                kw = dict(
                    symbol=symbol,
                    side=NewOrderSideEnum[side].value,
                    type=NewOrderTypeEnum["LIMIT"].value,
                    time_in_force=NewOrderTimeInForceEnum["GTC"].value,
                    quantity=self._plain(qty),
                    price=self._plain(self._grid(symbol, price, side)),
                    new_client_order_id=client_id if attempt == 0 else f"{client_id}r{attempt}")
                if self.hedge_mode:
                    # pos_side = сторона ПОЗИЦИИ, обязательна. Фолбэк на сторону
                    # ордера ЗАПРЕЩЁН: для закрывающего ордера она обратна стороне
                    # позиции, и молчаливый вывод делал TP открывающим (баг 22.09,
                    # LTCUSDC: позиция удвоилась). Нет стороны — падаем громко.
                    if not pos_side:
                        raise ValueError(
                            f"{symbol}: place_tp_limit в HEDGE требует pos_side "
                            "(сторона ПОЗИЦИИ), получен None")
                    kw["position_side"] = pos_side
                else:
                    kw["reduce_only"] = NewOrderReduceOnlyEnum["TRUE"].value
                r = self.client.rest_api.new_order(**kw)
                oid = int(r.data().order_id)
                break
            except BinanceError as e:
                if api_code(e) in (-4509, -1021) and attempt < 3:
                    # -2022 (ReduceOnly) сюда НЕ входит: позиции нет —
                    # это окончательный ответ, ретраить бессмысленно
                    if api_code(e) == -1021:
                        # часы разошлись с биржей — внеплановый ресинк и повтор
                        # (функция синхронная, крутится в to_thread — блокировка ок)
                        import bot.timesync as timesync
                        off = timesync.measure(self.client.rest_api)
                        log.warning("время: -1021, ресинк по бирже, офсет %+d мс", off)
                    time.sleep(2)
                    continue
                raise
        log.info("TP %s %s %s @%s id=%s", symbol, side, qty, price, oid)
        return oid

    def place_stop_market(self, symbol: str, side: str, stop_price: Decimal,
                          client_id: str, qty: Decimal | None = None) -> int | None:
        """STOP_MARKET через условные ордера. closePosition=true когда позиция
        открыта (страховка: закрывает весь объём стороны)."""
        close_pos = qty is None
        kw = dict(
            algo_type=NewAlgoOrderAlgoTypeEnum["CONDITIONAL"].value,
            symbol=symbol,
            side=NewAlgoOrderSideEnum[side].value,
            type=NewAlgoOrderTypeEnum["STOP_MARKET"].value,
            trigger_price=float(stop_price),
            working_type=NewAlgoOrderWorkingTypeEnum["CONTRACT_PRICE"].value,
            client_algo_id=client_id)
        if self.hedge_mode:
            # HEDGE: positionSide обязателен для algo-ордеров
            kw["position_side"] = "LONG" if side == "SELL" else "SHORT"
        if close_pos:
            kw["close_position"] = NewAlgoOrderClosePositionEnum["TRUE"].value
        else:
            kw["quantity"] = float(qty)
        for attempt in range(5):
            try:
                r = self.client.rest_api.new_algo_order(**kw)
                break
            except BinanceError as e:
                if api_code(e) in (-4509, -2022, -1021) and attempt < 4:
                    if api_code(e) == -1021:
                        import bot.timesync as timesync
                        off = timesync.measure(self.client.rest_api)
                        log.warning("время: -1021, ресинк по бирже, офсет %+d мс", off)
                    time.sleep(3)
                    continue
                raise
        else:
            raise RuntimeError(f"стоп {symbol} не размещён после ретраев")
        d = r.data().model_dump(by_alias=True) if hasattr(r.data(), "model_dump") else {}
        oid = d.get("orderId") or d.get("algoId") or d.get("order_id")
        log.info("стоп %s %s @%s id=%s closePos=%s",
                 symbol, side, stop_price, oid, close_pos)
        return int(oid) if oid is not None else None

    def place_market(self, symbol: str, side: str, qty: Decimal,
                     pos_side: str | None = None,
                     reduce_only: bool = False) -> int | None:
        for attempt in range(2):
            try:
                kw = dict(symbol=symbol, side=NewOrderSideEnum[side].value,
                          type=NewOrderTypeEnum["MARKET"].value,
                          quantity=self._plain(qty))
                if self.hedge_mode:
                    kw["position_side"] = pos_side or self._pside(side)
                if reduce_only:
                    # закрытие: позволяет номинал < 5 (пыль) — биржа иначе
                    # отвечает -4164 и позиция не закрывается никогда
                    kw["reduce_only"] = NewOrderReduceOnlyEnum.TRUE
                r = self.client.rest_api.new_order(**kw)
                return int(r.data().order_id)
            except BinanceError as e:
                if api_code(e) == -1021 and attempt == 0:
                    import bot.timesync as ts
                    ts.measure(self.client.rest_api)
                    continue
                log.warning("маркет %s %s: %s", symbol, side, e)
                return None
        return None

    def query_order_full(self, symbol: str, order_id: int) -> dict | None:
        for _ in range(3):
            try:
                r = self.client.rest_api.query_order(symbol=symbol,
                                                     order_id=order_id)
                d = r.data()
                return d.model_dump(by_alias=True) \
                    if hasattr(d, "model_dump") else d
            except BinanceError as e:
                if api_code(e) == -1021:
                    import bot.timesync as ts
                    ts.measure(self.client.rest_api)
                    continue
                return None
            except Exception:
                return None
        return None

    def position_information_for(self, symbols: list[str]) -> list:
        pos = unwrap(self.client.rest_api.position_information_v3().data())
        rows = getattr(pos, "root", None) or pos
        out = []
        for p in rows:
            d = p.model_dump(by_alias=True) if hasattr(p, "model_dump") else p
            if d.get("symbol") in symbols:
                out.append(d)
        return out

    def realized_pnl_since(self, symbol: str, since_ms: int) -> Decimal | None:
        """Биржа — источник правды: реализованный PnL + комиссии по символу
        с момента since_ms. None — если биржа не ответила (не фантазируем)."""
        try:
            r = self.client.rest_api.get_income_history(
                symbol=symbol, income_type="REALIZED_PNL",
                start_time=since_ms, limit=1000).data()
            rows = getattr(r, "root", None) or r
            total = Decimal("0")
            for e in rows:
                d = e.model_dump(by_alias=True) \
                    if hasattr(e, "model_dump") else e
                total += Decimal(str(d.get("income", "0")))
            try:
                r2 = self.client.rest_api.get_income_history(
                    symbol=symbol, income_type="COMMISSION",
                    start_time=since_ms, limit=1000).data()
                rows2 = getattr(r2, "root", None) or r2
                for e in rows2:
                    d = e.model_dump(by_alias=True) \
                        if hasattr(e, "model_dump") else e
                    total += Decimal(str(d.get("income", "0")))
            except Exception as e:
                log.warning("%s: комиссии из истории не получены: %s",
                            symbol, e)
            return total
        except Exception as e:
            log.warning("%s: реализованный PnL не получен: %s", symbol, e)
            return None

    def order_book_top(self, symbol: str) -> tuple[float, float] | None:
        try:
            r = self.market_client.rest_api.order_book(symbol=symbol, limit=5).data()
            d = r.model_dump(by_alias=True) if hasattr(r, "model_dump") else {}
            bids = d.get("bids") or []
            asks = d.get("asks") or []
            # свежие листинги бывают на демо, но ещё нет на боевом REST:
            # там приходит пустой стакан — это не авария, символа просто
            # нет; тихо возвращаем None (движки пропустят вход)
            if not bids or not asks:
                return None
            return float(bids[0][0]), float(asks[0][0])
        except Exception as e:
            log.warning("стакан %s: %s", symbol, e)
            return None

    def fetch_klines(self, symbol: str, interval: str, limit: int,
                     end_ms: int | None = None) -> list:
        kw = dict(symbol=symbol, interval=interval, limit=limit)
        if end_ms:
            kw["end_time"] = end_ms
        try:
            resp = self.market_client.rest_api.kline_candlestick_data(**kw)
            rows = getattr(resp.data(), "root", None) or resp.data()
            return list(rows)
        except Exception as e:
            log.warning("klines %s: %s", symbol, e)
            return []

    def build_filters(self, symbols: list[str]) -> None:
        info = self.client.rest_api.exchange_information().data()
        d = info.model_dump(by_alias=True)
        from bot.markets import parse_filters
        for s in d.get("symbols", []):
            if s.get("symbol") in symbols and s.get("status") == "TRADING":
                try:
                    self.filters[s["symbol"]] = parse_filters(s["symbol"],
                                                              s["filters"])
                except Exception as e:
                    log.warning("фильтры %s: %s", s["symbol"], e)

    def cancel_order(self, symbol: str, order_id: int) -> bool:
        try:
            self.client.rest_api.cancel_order(symbol=symbol, order_id=order_id)
            log.info("отменена заявка %s id=%s", symbol, order_id)
            return True
        except BinanceError as e:
            if api_code(e) == -2011:       # заявки уже нет (исполнилась/отменена)
                return False
            log.warning("cancel %s id=%s: %s", symbol, order_id, e)
            return False

    def query_order_status(self, symbol: str, order_id: int) -> str | None:
        """Статус заявки от биржи (источник правды): NEW / FILLED / CANCELED..."""
        try:
            r = self.client.rest_api.query_order(symbol=symbol, order_id=order_id)
            d = r.data().model_dump(by_alias=True) if hasattr(r.data(), "model_dump") else {}
            return d.get("status")
        except BinanceError as e:
            if api_code(e) == -2013:       # Order does not exist
                return None
            log.warning("query_order %s id=%s: %s", symbol, order_id, e)
            return None

    def cancel_algo_order(self, symbol: str, algo_id: int) -> bool:
        try:
            self.client.rest_api.cancel_algo_order(algo_id=algo_id)
            log.info("отменён стоп %s id=%s", symbol, algo_id)
            return True
        except BinanceError as e:
            log.warning("cancel_algo %s id=%s: %s", symbol, algo_id, e)
            return False

    # ---------- дневной PnL по направлениям (биржа = источник правды) ----------
    def daily_directional_pnl(self, start_ms: int) -> dict:
        """Авторитетный дневной PnL по направлениям: income history (окно) +
        userTrades по символам с ненулевым результатом (там есть side).
        realizedPnl начисляется на ЗАКРЫВАЮЩЕЙ стороне:
        SELL-заполнение закрыло лонг, BUY-заполнение закрыло шорт."""
        income = unwrap(self.client.rest_api.get_income_history(
            income_type="REALIZED_PNL", start_time=start_ms, limit=1000).data())
        rows = getattr(income, "root", None) or income
        by_symbol = {}
        for r in rows:
            d = r.model_dump(by_alias=True)
            sym, inc = d.get("symbol", "?"), Decimal(str(d.get("income", "0")))
            if inc != 0:
                by_symbol[sym] = by_symbol.get(sym, Decimal(0)) + inc
        out = {"LONG": Decimal(0), "SHORT": Decimal(0), "symbols": len(by_symbol)}
        for sym in by_symbol:
            try:
                trades = unwrap(self.client.rest_api.account_trade_list(
                    symbol=sym, start_time=start_ms, limit=1000).data())
                trows = getattr(trades, "root", None) or trades
                for t in trows:
                    d = t.model_dump(by_alias=True) if hasattr(t, "model_dump") else t
                    rp = Decimal(str(d.get("realizedPnl", "0")))
                    if rp == 0:
                        continue
                    bucket = "LONG" if d.get("side") == "SELL" else "SHORT"
                    out[bucket] += rp
            except Exception as e:
                log.warning("daily_directional_pnl %s: %s", sym, e)
        return out

    # ---------- реконсиляция ----------
    def snapshot(self) -> dict:
        oo = unwrap(self.client.rest_api.current_all_open_orders().data())
        ao = unwrap(self.client.rest_api.current_all_algo_open_orders().data())
        pos = unwrap(self.client.rest_api.position_information_v3().data())

        def rows(x):
            return getattr(x, "root", None) or x

        return {
            "orders": [o.model_dump(by_alias=True) for o in rows(oo)],
            "algo": [o.model_dump(by_alias=True) for o in rows(ao)],
            "positions": [p.model_dump(by_alias=True) for p in rows(pos)],
        }

    def account_wallet_balance(self, asset: str = "USDC") -> Decimal:
        """Кошельковый баланс (включая маржу открытых позиций) — база сайзинга."""
        bal = unwrap(self.client.rest_api.futures_account_balance_v3().data())

        def rows(x):
            return getattr(x, "root", None) or x
        for a in rows(bal):
            d = a.model_dump(by_alias=True) if hasattr(a, "model_dump") else a
            if d.get("asset") == asset:
                return Decimal(str(d.get("balance", d.get("availableBalance", "0"))))
        return Decimal("0")
