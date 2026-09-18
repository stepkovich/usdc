"""Исполнитель: ордера на демо (REST), режим позиции (хедж/оневей), плечо,
реконсиляция. Биржа — единственный источник правды."""
from __future__ import annotations

import asyncio
import json
import time
import logging
from decimal import Decimal

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
        conf = self.client.rest_api.futures_account_configuration().data()
        conf_d = conf.model_dump(by_alias=True) if hasattr(conf, "model_dump") else {}
        current = {p.get("symbol"): int(p.get("leverage", 0))
                   for p in conf_d.get("positions", [])}
        for s in symbols:
            if current.get(s) == self.cfg.leverage:
                self.leverage_set[s] = self.cfg.leverage
                continue
            try:
                self.client.rest_api.change_initial_leverage(
                    symbol=s, leverage=self.cfg.leverage)
                self.leverage_set[s] = self.cfg.leverage
            except BinanceError as e:
                code = api_code(e)
                log.error("плечо %s -> %sx не установлено (код %s): %s",
                          s, self.cfg.leverage, code, e)
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
    def place_entry_limit(self, symbol: str, side: str, price: Decimal,
                          qty: Decimal, client_id: str) -> int | None:
        r = self.client.rest_api.new_order(
            symbol=symbol,
            side=NewOrderSideEnum[side].value,
            type=NewOrderTypeEnum["LIMIT"].value,
            position_side=self._pside(side),
            time_in_force=NewOrderTimeInForceEnum["GTC"].value,
            quantity=float(qty), price=float(price),
            new_client_order_id=client_id)
        oid = int(r.data().order_id)
        log.info("заявка %s %s %s @%s id=%s", symbol, side, qty, price, oid)
        return oid

    def place_tp_limit(self, symbol: str, side: str, price: Decimal,
                       qty: Decimal, client_id: str) -> int | None:
        # -2022: гонка сразу после исполнения входа — ретраим с паузой
        for attempt in range(4):
            try:
                r = self.client.rest_api.new_order(
                    symbol=symbol,
                    side=NewOrderSideEnum[side].value,
                    type=NewOrderTypeEnum["LIMIT"].value,
                    position_side=self._pside(side),
                    time_in_force=NewOrderTimeInForceEnum["GTC"].value,
                    quantity=float(qty), price=float(price),
                    reduce_only=NewOrderReduceOnlyEnum["TRUE"].value,
                    new_client_order_id=client_id if attempt == 0 else f"{client_id}r{attempt}")
                oid = int(r.data().order_id)
                break
            except BinanceError as e:
                if api_code(e) in (-2022, -4509) and attempt < 3:
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
        if close_pos:
            kw["close_position"] = NewAlgoOrderClosePositionEnum["TRUE"].value
        else:
            kw["quantity"] = float(qty)
        for attempt in range(5):
            try:
                r = self.client.rest_api.new_algo_order(**kw)
                break
            except BinanceError as e:
                if api_code(e) in (-4509, -2022) and attempt < 4:
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
