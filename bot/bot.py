"""Сборка: фид (прод-данные) + стратегия + исполнитель (демо) + user-data + watchdog.
Запуск: PYTHONPATH=.. python3 bot/bot.py  (см. systemd-юнит)."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from decimal import Decimal
from pathlib import Path

from binance_common.configuration import ConfigurationRestAPI, ConfigurationWebSocketStreams
from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)

from bot.config import Mode, BotConfig
from bot.executor import Executor
from bot.feed import MarketFeed
from bot.ledger import Ledger
from bot.markets import parse_filters
from bot.user_stream import RawUserStream
from bot.strategy import (
    CancelEntry,
    CancelExit,
    PlaceEntry,
    PlaceStop,
    PlaceTp,
    Strategy,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("bot")


class Bot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        import os
        journal_path = Path(os.environ.get("BOT_JOURNAL_PATH",
                                           ROOT / "bot" / "journal.db"))
        self.ledger = Ledger(journal_path)
        self.feed = MarketFeed(cfg)
        self.strategy = Strategy(cfg)
        self.exec: Executor | None = None
        self.filters: dict[str, object] = {}
        self.symbols: list[str] = []
        self.exits: dict[str, dict[str, int | None]] = {}
        self._acc: dict[str, dict] = {}   # symbol -> {tp: id, stop: id}
        self.paused: set[str] = set()
        self.balance_snapshot: Decimal | None = None
        self._user_conn = None
        self._listen_key_task: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    # ---------------- инициализация окружения ----------------
    async def setup(self) -> None:
        cfg = self.cfg
        # 1) фильтры биржи с демо (исполнение) — источник правды по шагам
        probe = DerivativesTradingUsdsFutures(config_rest_api=ConfigurationRestAPI(
            api_key=cfg.api_key, api_secret=cfg.api_secret, base_path=cfg.exec_rest_url))
        info = await asyncio.to_thread(probe.rest_api.exchange_information)
        d = info.data().model_dump(by_alias=True)
        usdc = {s["symbol"]: s["filters"] for s in d["symbols"]
                if s["symbol"].endswith("USDC") and s.get("status") == "TRADING"}
        self.symbols = cfg.symbols or sorted(usdc)
        for s in list(self.symbols):
            if s not in usdc:
                log.warning("символ %s недоступен на %s — исключён", s, cfg.mode.value)
                self.symbols.remove(s)
                continue
            self.filters[s] = parse_filters(s, usdc[s])
        # пары, где 50 USDC < минимального лота — не торгуем
        self.exec = Executor(cfg, self.filters)
        self.symbols = list(usdc.keys() & set(self.symbols))
        log.info("режим=%s пар=%d исполнение=%s рынок=%s",
                 cfg.mode.value, len(self.symbols), cfg.exec_rest_url,
                 cfg.market_streams_url)
        if not cfg.dry_run:
            self.balance_snapshot = await asyncio.to_thread(
                self.exec.account_wallet_balance, "USDC")
            log.info("снимок баланса: %s USDC -> риск-бюджет одной сделки %s USDC "
                     "(%s%% баланса)", self.balance_snapshot,
                     (self.balance_snapshot * cfg.risk_pct).quantize(Decimal("0.0001")),
                     cfg.risk_pct * 100)
        if not cfg.dry_run:
            # режим позиции (хедж/оневей) — до любых ордеров
            await asyncio.to_thread(self.exec.ensure_position_mode, True)
            # плечо: проверить текущее и установить нужное
            await asyncio.to_thread(self.exec.verify_and_set_leverage, self.symbols)
            await asyncio.gather(*[asyncio.to_thread(self.exec.setup_symbol, s)
                                   for s in self.symbols])
            log.info("маржа/плечо настроены")

    # ---------------- исполнение действий стратегии ----------------
    async def apply(self, acts: list) -> None:
        for a in acts:
            try:
                match a:
                    case PlaceEntry():
                        f = self.filters[a.symbol]
                        price = f.round_price(a.price)
                        stop_dist = a.atr0 * self.cfg.stop_atr_mult
                        budget = self.balance_snapshot * self.cfg.risk_pct
                        natural = budget / stop_dist
                        floor = f.min_notional * self.cfg.notional_buffer
                        if natural < floor:
                            log.warning("%s: пропуск — размер %.2f ниже пола %.2f "
                                        "(стоп %.2f%%)", a.symbol, natural, floor,
                                        stop_dist * 100)
                            self.strategy.state(a.symbol).pending = None
                            continue
                        size = min(natural, self.cfg.max_notional)
                        qty = f.qty_for_notional(size, price)
                        if qty is None:
                            log.warning("%s: лот не сходится (size=%.2f) — пропуск",
                                        a.symbol, size)
                            self.strategy.state(a.symbol).pending = None
                            continue
                        log.info("САЙЗИНГ %s: бюджет %s / стоп %.2f%% = %.2f "
                                 "-> торгуем %.2f (вилка [%s, %s])", a.symbol,
                                 budget.quantize(Decimal("0.0001")),
                                 stop_dist * 100, natural, size,
                                 floor.quantize(Decimal("0.01")),
                                 self.cfg.max_notional)
                        self.ledger.event("signal_entry", a.symbol, {
                            "side": a.side.value, "price": str(price),
                            "qty": str(qty), "atr0": str(a.atr0), "cid": a.client_id})
                        if self.cfg.dry_run:
                            log.info("[DRY] вход %s %s %s @%s", a.symbol,
                                     a.side.value, qty, price)
                            continue
                        oid = await asyncio.to_thread(
                            self.exec.place_entry_limit, a.symbol,
                            "BUY" if a.side.value == "LONG" else "SELL", price, qty,
                            a.client_id)
                        self.strategy.entry_placed(a.symbol, oid or 0, qty)
                    case CancelEntry():
                        if not self.cfg.dry_run and a.order_id:
                            await asyncio.to_thread(
                                self.exec.cancel_order, a.symbol, a.order_id)
                        self.ledger.event("cancel_entry", a.symbol,
                                          {"order_id": a.order_id, "reason": a.reason})
                    case PlaceTp():
                        f = self.filters[a.symbol]
                        price = f.round_price(a.price)
                        self.ledger.event("tp_placed", a.symbol, {"price": str(price)})
                        if self.cfg.dry_run:
                            log.info("[DRY] TP %s %s %s @%s", a.symbol,
                                     a.side.value, a.qty, price)
                            continue
                        oid = await asyncio.to_thread(
                            self.exec.place_tp_limit, a.symbol,
                            "SELL" if a.side.value == "LONG" else "BUY", price,
                            a.qty, a.client_id)
                        self.exits.setdefault(a.symbol, {})["tp"] = oid
                    case PlaceStop():
                        f = self.filters[a.symbol]
                        sp = f.round_price(a.stop_price)
                        self.ledger.event("stop_placed", a.symbol, {"price": str(sp)})
                        if self.cfg.dry_run:
                            log.info("[DRY] STOP %s %s @%s", a.symbol, a.side.value, sp)
                            continue
                        oid = await asyncio.to_thread(
                            self.exec.place_stop_market, a.symbol,
                            "SELL" if a.side.value == "LONG" else "BUY", sp, a.client_id)
                        self.exits.setdefault(a.symbol, {})["stop"] = oid
                    case CancelExit():
                        ids = self.exits.get(a.symbol, {})
                        if not self.cfg.dry_run:
                            if a.kind == "stop" and ids.get("stop"):
                                await asyncio.to_thread(
                                    self.exec.cancel_algo_order, a.symbol, ids["stop"])
                            if a.kind == "tp" and ids.get("tp"):
                                await asyncio.to_thread(
                                    self.exec.cancel_order, a.symbol, ids["tp"])
                        self.ledger.event("cancel_exit", a.symbol,
                                          {"kind": a.kind, "reason": a.reason})
            except Exception:
                log.exception("ошибка исполнения действия %r", a)

    # ---------------- user data stream (исполнение) ----------------
    async def start_user_stream(self) -> None:
        if self.cfg.dry_run:
            log.info("dry-run: user-data стрим не нужен")
            return

        def new_key() -> str:
            return self.exec.client.rest_api.start_user_data_stream().data().listen_key

        def keepalive() -> None:
            self.exec.client.rest_api.keepalive_user_data_stream()

        self._raw_stream = RawUserStream(
            new_key, keepalive,
            lambda d: asyncio.create_task(self.on_user_event(d)))
        self._tasks.append(asyncio.create_task(self._raw_stream.run()))

    async def on_user_event(self, d: dict) -> None:
        try:
            ev = d.get("e")
            if ev == "ORDER_TRADE_UPDATE":
                o = d.get("o") or {}
                sym, status, cid = o.get("s"), o.get("X"), str(o.get("c", ""))
                role = cid.split("-")[1] if cid.startswith("scr-") else "?"
                self.ledger.fill(sym, role, o)
                if status not in ("FILLED", "PARTIALLY_FILLED", "NEW"):
                    return
                D0 = Decimal(0)
                acc = self._acc.setdefault(
                    sym, {"fees": D0, "maker": D0, "pnl": D0, "orders": {},
                          "done": True, "open": False})
                # новая сделка начинается с NEW входной заявки: обнуляем накопители
                if role == "E" and status == "NEW" and not acc.get("open"):
                    acc.update({"fees": D0, "maker": D0, "pnl": D0,
                                "orders": {}, "done": False, "open": True})
                # rp кумулятивен по ордеру -> учитываем дельту
                oid = str(o.get("i", ""))
                rp = Decimal(str(o.get("rp") or "0"))
                acc["pnl"] += rp - acc["orders"].get(oid, D0)
                acc["orders"][oid] = rp
                comm = Decimal(str(o.get("n") or "0"))
                acc["fees"] += comm
                if role in ("E", "T"):            # мейкерские исполнения
                    acc["maker"] += comm
                if role == "E" and status == "FILLED":
                    qty = Decimal(o.get("z") or "0")
                    avg = Decimal(o.get("ap") or "0")
                    acc.update(entry_ts=time.time(), entry_px=avg, qty=qty,
                               side=o.get("S"))
                    acts = self.strategy.entry_filled(sym, qty, avg, int(o.get("i", 0)))
                    await self.apply(acts)
                elif role in ("T", "S") and status == "FILLED":
                    exit_kind = "tp" if role == "T" else "stop"
                    exit_px = Decimal(str(o.get("ap") or o.get("L") or "0"))
                    acc["open"] = False
                    self.ledger.trade_closed(
                        sym, acc.get("side", "?"), acc.get("entry_ts", 0),
                        time.time(), acc.get("entry_px", Decimal(0)), exit_px,
                        acc.get("qty", Decimal(0)), exit_kind,
                        acc["fees"], acc["fees"] - acc["maker"], acc["pnl"])
                    acts = self.strategy.exit_filled(sym, exit_kind)
                    await self.apply(acts)
            self.ledger.event("user_event", "", {"e": ev})
        except Exception:
            log.exception("user event error")

    # ---------------- реконсиляция и базис ----------------
    async def reconciler(self) -> None:
        refresh_counter = 0
        while not self._stop.is_set():
            try:
                snap = await asyncio.to_thread(self.exec.snapshot)
                pos_syms = set()
                for p in snap["positions"]:
                    amt = Decimal(str(p.get("positionAmt", "0")))
                    if amt == 0:
                        continue
                    sym = p["symbol"]
                    pos_syms.add(sym)
                    if sym not in self.filters:
                        continue          # чужие позиции (USDT и пр.) — не трогаем
                    side = "LONG" if amt > 0 else "SHORT"
                    avg = Decimal(str(p.get("entryPrice", "0")))
                    st = self.strategy.state(sym)
                    if not st.position:
                        # позиция без состояния (рестарт/сбой) -> восстанавливаем защиту
                        log.warning("%s: позиция без состояния — восстанавливаю TP/стоп", sym)
                        st.position = {"side": side, "qty": abs(amt), "entry": avg,
                                       "tp_id": None, "stop_id": None,
                                       "tp_price": None, "repaired": True}
                        await self.repair_exits(sym, side, avg, abs(amt), snap=snap)
                    else:
                        await self.repair_exits(sym, side, st.position["entry"],
                                                st.position["qty"], snap=snap)
                for s in pos_syms - set(self.strategy.states):
                    if s not in self.filters:
                        self.strategy.force_flat(s)
                        log.warning("реконсиляция: чужая позиция %s — не трогаем", s)
                # позиция была нашей, но на бирже её нет -> закрыта ВРУЧНУЮ:
                # чистим хвосты (TP/стоп), ставим кулдаун, символ снова торгуем
                for s in list(self.strategy.states):
                    st = self.strategy.states[s]
                    if st.position and s in self.filters and s not in pos_syms:
                        log.warning("%s: позиция закрыта вручную — отмена хвостов, "
                                    "кулдаун, символ свободен", s)
                        self.ledger.event("manual_close", s, {})
                        ids = self.exits.get(s, {})
                        if not self.cfg.dry_run:
                            if ids.get("tp"):
                                await asyncio.to_thread(
                                    self.exec.cancel_order, s, ids["tp"])
                            if ids.get("stop"):
                                await asyncio.to_thread(
                                    self.exec.cancel_algo_order, s, ids["stop"])
                        st.cooldown_until = st.bars_seen + self.cfg.cool_bars
                        st.position = None
                        self.exits[s] = {}
                self.ledger.event("snapshot", "", {
                    "positions": len(pos_syms),
                    "orders": len(snap["orders"]), "algo": len(snap["algo"])})
                log.info("реконсиляция: позиций %d, заявок %d, стопов %d",
                         len(pos_syms), len(snap["orders"]), len(snap["algo"]))
                refresh_counter += 1
                if refresh_counter >= 360:   # фильтры биржи — раз в ~6ч
                    refresh_counter = 0
                    await self.refresh_filters()
            except Exception:
                log.exception("реконсиляция")
            await asyncio.sleep(60)

    async def repair_exits(self, sym: str, side: str, avg: Decimal, qty: Decimal,
                           snap: dict) -> None:
        """Гарантирует, что у открытой позиции висят TP-лимитка и STOP_MARKET."""
        orders = [o for o in snap["orders"] if o.get("symbol") == sym]
        algo = [o for o in snap["algo"] if o.get("symbol") == sym]
        f = self.filters[sym]
        exit_side = "SELL" if side == "LONG" else "BUY"
        st = self.strategy.state(sym)
        entry = st.position["entry"] if st.position else avg
        tp_px = f.round_price(entry * (1 + self.cfg.target_pct) if side == "LONG"
                              else entry * (1 - self.cfg.target_pct))
        if not any(str(o.get("clientOrderId", "")).startswith("scr-T") for o in orders):
            log.warning("%s: TP отсутствует — ставлю @%s", sym, tp_px)
            oid = await asyncio.to_thread(self.exec.place_tp_limit, sym, exit_side,
                                          tp_px, abs(qty), f"scr-T-{sym}-repair")
            self.exits.setdefault(sym, {})["tp"] = oid
            if st.position:
                st.position["tp_id"] = oid
        if not any(str(a.get("clientAlgoId", "")).startswith("scr-S") for a in algo):
            atr = (self.feed.hist[sym].atr_frac
                   if sym in self.feed.hist and self.feed.hist[sym].ready
                   else Decimal("0.001"))
            stop_px = f.round_price(entry * (1 - self.cfg.stop_atr_mult * atr)
                                    if side == "LONG"
                                    else entry * (1 + self.cfg.stop_atr_mult * atr))
            log.warning("%s: стоп отсутствует — ставлю @%s", sym, stop_px)
            oid = await asyncio.to_thread(self.exec.place_stop_market, sym, exit_side,
                                          stop_px, f"scr-S-{sym}-repair")
            self.exits.setdefault(sym, {})["stop"] = oid
            if st.position:
                st.position["stop_id"] = oid

    async def refresh_filters(self) -> None:
        probe = DerivativesTradingUsdsFutures(config_rest_api=ConfigurationRestAPI(
            api_key=self.cfg.api_key, api_secret=self.cfg.api_secret,
            base_path=self.cfg.exec_rest_url))
        info = (await asyncio.to_thread(probe.rest_api.exchange_information)).data()
        d = info.model_dump(by_alias=True)
        for s in d["symbols"]:
            if s["symbol"] in self.filters and s.get("status") == "TRADING":
                self.filters[s["symbol"]] = parse_filters(s["symbol"], s["filters"])
        log.info("фильтры биржи обновлены")

    # ---------------- watchdog ----------------
    async def watchdog(self) -> None:
        bal_ts = time.time()
        while not self._stop.is_set():
            await asyncio.sleep(30)
            if time.time() - bal_ts >= 86400:     # снимок баланса раз в сутки
                try:
                    self.balance_snapshot = await asyncio.to_thread(
                        self.exec.account_wallet_balance, "USDC")
                    log.info("снимок баланса обновлён: %s USDC", self.balance_snapshot)
                    bal_ts = time.time()
                except Exception:
                    log.exception("обновление снимка баланса")
            if self.feed.stale(180):
                log.error("фид молчит >180с — переподключение стримов")
                try:
                    await self.feed.start_streams(self.symbols)
                except Exception:
                    log.exception("перезапуск стримов не удался")

    # ---------------- главный цикл ----------------
    async def run(self) -> None:
        await self.setup()
        await self.feed.warmup(self.symbols)
        await self.feed.start_streams(self.symbols)
        await self.start_user_stream()
        self._tasks = [asyncio.create_task(self.reconciler()),
                       asyncio.create_task(self.watchdog())]
        log.info("бот запущен: пары=%d dry_run=%s цель=%s", len(self.symbols),
                 self.cfg.dry_run, self.cfg.target_pct)
        while not self._stop.is_set():
            sym, bar = await self.feed.closed_bars()
            h = self.feed.hist[sym]
            if h.ready:
                acts = self.strategy.on_closed_bar(h, bar)
                if acts:
                    await self.apply(acts)

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()


def main() -> None:
    import os
    log_path = Path(os.environ.get("BOT_LOG_PATH", ROOT / "bot" / "bot.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_path)])
    cfg = BotConfig.from_env(ROOT)
    bot = Bot(cfg)

    async def runner():
        try:
            await bot.run()
        except asyncio.CancelledError:
            pass
        finally:
            await bot.stop()

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        log.info("остановлено вручную")


if __name__ == "__main__":
    main()
