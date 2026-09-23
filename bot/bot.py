"""Сборка: фид (прод-данные) + стратегия + исполнитель (демо) + user-data + watchdog.
Запуск: PYTHONPATH=.. python3 bot/bot.py  (см. systemd-юнит)."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
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
from bot.executor import Executor, unwrap
from bot.feed import MarketFeed
from bot.ledger import Ledger
from bot.markets import parse_filters
from bot.user_stream import RawUserStream, WS_BASE_DEMO, WS_BASE_MAINNET
import bot.telegram as tg
import bot.timesync as timesync
from bot.structure import Structure15m
from bot.strategy import (
    CancelEntry,
    CancelExit,
    PlaceEntry,
    PlaceStop,
    PlaceTp,
    RearmEntry,
    Strategy,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("bot")


class Bot:
    def __init__(self, cfg: BotConfig):
        # эксклюзивный лок журнала: второй экземпляр бота не сможет стартовать
        # (второй писец в journal.db = причина дрейфа учёта)
        import fcntl
        self._lock_file = open(ROOT / "bot" / "bot.lock", "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("второй экземпляр бота уже работает (лок journal.db) — отказ от старта")
        self.cfg = cfg
        import os
        journal_path = Path(os.environ.get("BOT_JOURNAL_PATH",
                                           ROOT / "bot" / "journal.db"))
        self.ledger = Ledger(journal_path, env=cfg.mode.value)
        self.feed = MarketFeed(cfg)
        self.strategy = Strategy(cfg)
        self.exec: Executor | None = None
        self.filters: dict[str, object] = {}
        self.symbols: list[str] = []
        self.exits: dict[str, dict[str, int | None]] = {}
        self._acc: dict[str, dict] = {}   # symbol -> {tp: id, stop: id}
        self.paused: set[str] = set()
        self.balance_snapshot: Decimal | None = None
        # дневные счётчики PnL по направлениям (сброс при суточном снимке баланса)
        self.daily_pnl: dict[str, Decimal] = {"LONG": Decimal(0), "SHORT": Decimal(0)}
        self.daily_locked: dict[str, bool] = {"LONG": False, "SHORT": False}
        self.daily_total_pnl: Decimal = Decimal(0)
        self.daily_total_locked: bool = False
        self._day_key = datetime.now(timezone.utc).date()
        self._day_start_ms = int(datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        self._pnl_check_counter = 0
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
        # синхронизация времени с биржей ДО первого подписанного запроса:
        # защита от -1021 «Timestamp outside recvWindow»
        timesync.install()
        off = await asyncio.to_thread(timesync.measure, self.exec.client.rest_api)
        log.info("время: офсет к бирже %+d мс (round-trip %.0f мс)",
                 off, timesync.LAST_RTT_MS)
        self.symbols = list(usdc.keys() & set(self.symbols))
        self.smc_struct = ({s: Structure15m(s) for s in self.symbols}
                           if cfg.smc_filter else {})
        log.info("режим=%s пар=%d исполнение=%s рынок=%s",
                 cfg.mode.value, len(self.symbols), cfg.exec_rest_url,
                 cfg.market_streams_url)
        try:
            self.balance_snapshot = await asyncio.to_thread(
                self.exec.account_wallet_balance, "USDC")
        except Exception as e:
            if cfg.dry_run:
                self.balance_snapshot = Decimal("1000")   # виртуальный баланс dry-run
                log.warning("dry-run: баланс биржи недоступен (%s) — "
                            "виртуальный снимок %s USDC", e, self.balance_snapshot)
            else:
                raise
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
                        # общий дневной кап (V-образный отскок): суммарный убыток дня
                        if (self.cfg.total_daily_loss_pct > 0 and self.balance_snapshot
                                and self.daily_total_pnl
                                <= -self.balance_snapshot * self.cfg.total_daily_loss_pct):
                            log.warning("%s: сигнал пропущен — общий дневной кап "
                                        "(сегодня всего %s)", a.symbol,
                                        self.daily_total_pnl.quantize(Decimal("0.01")))
                            self.ledger.event("total_daily_limit_skip", a.symbol, {})
                            self.strategy.state(a.symbol).pending = None
                            continue
                        if self.balance_snapshot is None:
                            log.warning("%s: снимок баланса не готов — сигнал пропущен",
                                        a.symbol)
                            self.strategy.state(a.symbol).pending = None
                            continue
                        # направленный дневной лимит убытка
                        if self.cfg.daily_loss_pct > 0 and \
                                self.daily_locked.get(a.side.value):
                            log.warning("%s: сигнал %s пропущен — дневной лимит %s "
                                        "исчерпан (сегодня %s)", a.symbol,
                                        a.side.value, self.cfg.daily_loss_pct * 100,
                                        self.daily_pnl[a.side.value].quantize(Decimal("0.01")))
                            self.ledger.event("daily_limit_skip", a.symbol,
                                              {"side": a.side.value})
                            self.strategy.state(a.symbol).pending = None
                            continue
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
                        book = self.feed.book.get(a.symbol)
                        spread_bp = None
                        if book and book[0] > 0:
                            mid = (book[0] + book[1]) / 2
                            spread_bp = float((book[1] - book[0]) / mid * 10000)
                        # квартиль волатильности ЗАМОРАЖИВАЕТСЯ в момент сделки
                        # (пары мигрируют между квартилями при пересчёте)
                        vol_q = None
                        try:
                            import bisect
                            stops = sorted(self.cfg.stop_atr_mult *
                                           self.feed.hist[s].atr_frac
                                           for s in self.symbols
                                           if s in self.feed.hist and self.feed.hist[s].ready
                                           and self.feed.hist[s].atr_frac > 0)
                            my = self.cfg.stop_atr_mult * a.atr0
                            vol_q = 1 + bisect.bisect_left(stops, my)
                        except Exception:
                            pass
                        self.ledger.event("signal_entry", a.symbol, {
                            "side": a.side.value, "price": str(price),
                            "qty": str(qty), "atr0": str(a.atr0), "cid": a.client_id,
                            "spread_bp": spread_bp, "vol_quartile": vol_q})
                        if self.cfg.dry_run:
                            log.info("[DRY] вход %s %s %s @%s", a.symbol,
                                     a.side.value, qty, price)
                            continue
                        tg.fire(f"{'🟢' if a.side.value == 'LONG' else '🔴'} "
                                f"<b>ЗАЯВКА {a.symbol} {a.side.value}</b>\n"
                                f"{size:.2f} USDC @ {price} — ждёт отката к уровню")
                        oid = await asyncio.to_thread(
                            self.exec.place_entry_limit, a.symbol,
                            "BUY" if a.side.value == "LONG" else "SELL", price, qty,
                            a.client_id)
                        self.strategy.entry_placed(a.symbol, oid or 0, qty)
                    case CancelEntry():
                        self.ledger.event("cancel_entry", a.symbol,
                                          {"order_id": a.order_id, "reason": a.reason,
                                           "min_gap_bp": round(a.gap_bp, 1)})
                        if a.reason == "timeout":
                            log.info("%s: заявка не исполнена за %d мин; ближайший "
                                     "подход к уровню %.0f бп", a.symbol,
                                     self.cfg.wait_bars, a.gap_bp)
                        if not self.cfg.dry_run and a.order_id:
                            await asyncio.to_thread(
                                self.exec.cancel_order, a.symbol, a.order_id)
                        self.ledger.event("cancel_entry", a.symbol,
                                          {"order_id": a.order_id, "reason": a.reason})
                    case RearmEntry():
                        # перестановка заявки на новый экстремум (BOT_REARM=1).
                        # Порядок жёсткий: отмена старой (правда — ответ биржи),
                        # проверка гонок, только потом новая заявка.
                        st2 = self.strategy.state(a.symbol)
                        f = self.filters[a.symbol]
                        price = f.round_price(a.level)
                        stop_dist = a.atr0 * self.cfg.stop_atr_mult
                        if self.balance_snapshot is None or stop_dist <= 0:
                            continue
                        budget = self.balance_snapshot * self.cfg.risk_pct
                        natural = budget / stop_dist
                        floor = f.min_notional * self.cfg.notional_buffer
                        # направленный дневной лимит: перестановка = новое входное решение
                        if self.cfg.daily_loss_pct > 0 and \
                                self.daily_locked.get(a.side.value):
                            log.warning("%s: rearm пропущен — дневной лимит %s",
                                        a.symbol, a.side.value)
                            self.ledger.event("rearm_entry", a.symbol,
                                              {"result": "skip_lock",
                                               "old_order_id": a.old_order_id})
                            if not self.cfg.dry_run and a.old_order_id:
                                await asyncio.to_thread(self.exec.cancel_order,
                                                        a.symbol, a.old_order_id)
                            st2.pending = None
                            continue
                        if natural < floor:
                            log.warning("%s: rearm отменён — размер %.2f ниже "
                                        "пола %.2f (стоп %.2f%%)", a.symbol,
                                        natural, floor, stop_dist * 100)
                            self.ledger.event("rearm_entry", a.symbol,
                                              {"result": "skip_floor",
                                               "old_order_id": a.old_order_id})
                            if not self.cfg.dry_run and a.old_order_id:
                                await asyncio.to_thread(self.exec.cancel_order,
                                                        a.symbol, a.old_order_id)
                            st2.pending = None
                            continue
                        qty = f.qty_for_notional(
                            min(natural, self.cfg.max_notional), price)
                        if qty is None:
                            log.warning("%s: rearm — лот не сходится (size=%.2f)",
                                        a.symbol, natural)
                            self.ledger.event("rearm_entry", a.symbol,
                                              {"result": "skip_lot",
                                               "old_order_id": a.old_order_id})
                            if not self.cfg.dry_run and a.old_order_id:
                                await asyncio.to_thread(self.exec.cancel_order,
                                                        a.symbol, a.old_order_id)
                            st2.pending = None
                            continue
                        if self.cfg.dry_run:
                            log.info("[DRY] rearm %s -> уровень %s", a.symbol, price)
                            continue
                        # 1) гасим старую заявку; -2011 => её уже нет — уточняем
                        cancelled = await asyncio.to_thread(
                            self.exec.cancel_order, a.symbol, a.old_order_id) \
                            if a.old_order_id else False
                        if not cancelled and a.old_order_id:
                            status = await asyncio.to_thread(
                                self.exec.query_order_status, a.symbol, a.old_order_id)
                            if status in ("FILLED", "PARTIALLY_FILLED"):
                                # старая успела исполниться — позицию ведёт
                                # обычный цикл, перестановка отменяется
                                log.info("%s: rearm отменён — заявка уже "
                                         "исполнилась (%s)", a.symbol, status)
                                self.ledger.event("rearm_entry", a.symbol,
                                                  {"result": "race_filled",
                                                   "old_order_id": a.old_order_id})
                                continue
                        # 2) pending всё ещё наш? (fill по стриму мог прийти в фоне)
                        if not st2.pending or \
                                st2.pending.get("order_id") != a.old_order_id:
                            log.info("%s: rearm отменён — состояние заявки "
                                     "изменилось", a.symbol)
                            continue
                        # 3) новая заявка на свежем уровне
                        oid = await asyncio.to_thread(
                            self.exec.place_entry_limit, a.symbol,
                            "BUY" if a.side.value == "LONG" else "SELL", price, qty,
                            a.client_id)
                        # 4) гонка: старая могла исполниться, ПОКА мы ставили
                        # новую (fill-событие обнуляет pending). Тогда новую
                        # заявку немедленно гасим — позиции быть не должно.
                        if not st2.pending or \
                                st2.pending.get("order_id") != a.old_order_id:
                            if oid and not self.cfg.dry_run:
                                await asyncio.to_thread(
                                    self.exec.cancel_order, a.symbol, oid)
                            self.ledger.event("rearm_entry", a.symbol,
                                              {"result": "race_filled_late",
                                               "order_id": oid or 0})
                            log.warning("%s: rearm — старая заявка исполнилась "
                                        "в момент перестановки, новая отменена",
                                        a.symbol)
                            continue
                        st2.pending.update(level=a.level, atr0=a.atr0,
                                           client_id=a.client_id, order_id=oid or 0,
                                           qty=qty,
                                           deadline=st2.bars_seen + self.cfg.wait_bars)
                        self.ledger.event("rearm_entry", a.symbol,
                                          {"level": str(a.level), "qty": str(qty),
                                           "order_id": oid or 0,
                                           "old_order_id": a.old_order_id})
                        tg.fire(f"🔁 <b>ПЕРЕСТАНОВКА {a.symbol} {a.side.value}</b>\n"
                                f"{float(qty) * float(price):.2f} USDC @ {price}")
                        log.info("rearm %s: новая заявка id=%s @%s", a.symbol,
                                 oid, price)
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
                            a.qty, a.client_id, pos_side=a.side.value)
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

        ws_base = WS_BASE_MAINNET if self.cfg.mode is Mode.MAINNET else WS_BASE_DEMO
        self._raw_stream = RawUserStream(
            new_key, keepalive,
            lambda d: asyncio.create_task(self.on_user_event(d)), ws_base=ws_base)
        self._tasks.append(asyncio.create_task(self._raw_stream.run()))

    async def on_user_event(self, d: dict) -> None:
        try:
            ev = d.get("e")
            if ev == "ORDER_TRADE_UPDATE":
                o = d.get("o") or {}
                sym, status, cid = o.get("s"), o.get("X"), str(o.get("c", ""))
                role = cid.split("-")[1] if cid.startswith("scr-") else "?"
                # в журнал — только реальные исполнения: NEW/CANCELED с нулевым
                # количеством писались призрачными строками и портили счётчики
                if status in ("FILLED", "PARTIALLY_FILLED"):
                    self.ledger.fill(sym, role, o)
                if status not in ("FILLED", "PARTIALLY_FILLED", "NEW"):
                    return
                D0 = Decimal(0)
                acc = self._acc.setdefault(
                    sym, {"fees": D0, "maker": D0, "pnl": D0, "orders": {},
                          "done": True, "open": False})
                # новая сделка начинается с NEW входной заявки: обнуляем накопители
                # протрузия: насколько цена прошла СКВОЗЬ уровень в минуту исполнения
                fb = self.feed.forming.get(sym)
                if fb and role in ("E", "T") and status in ("FILLED", "PARTIALLY_FILLED"):
                    try:
                        hi, lo = fb[1], fb[2]
                        if role == "E" and self.strategy.state(sym).pending:
                            lvl = self.strategy.state(sym).pending["level"]
                            depth = (lvl - lo) / lvl if o.get("S") == "BUY" \
                                else (hi - lvl) / lvl
                        elif role == "T":
                            stp = self.strategy.state(sym).position or {}
                            tp = stp.get("tp_price")
                            side = stp.get("side")
                            if tp is None:
                                tp = (Decimal(str(o.get("ap") or 0))
                                      * (1 + self.cfg.target_pct))
                                side = "LONG" if o.get("S") == "SELL" else "SHORT"
                            depth = (hi - tp) / tp if side == "LONG" \
                                else (tp - lo) / tp
                        else:
                            depth = None
                        if depth is not None and depth >= 0:
                            self.ledger.event("fill_protrusion", sym, {
                                "kind": "entry" if role == "E" else "tp",
                                "depth_bp": round(float(depth) * 10000, 1)})
                    except Exception:
                        pass                      # измерение не должно ломать торговлю
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
                # возврат по СТАВКЕ: мейкер (<=3 бп) -> промо обнулит; тейкер остаётся
                try:
                    _px = Decimal(str(o.get("ap") or o.get("L") or "0"))
                    _qt = Decimal(str(o.get("l") or o.get("z") or "0"))
                    _rate_bp = (comm / (_px * _qt) * 10000) if _px * _qt > 0 else Decimal(0)
                except Exception:
                    _rate_bp = Decimal(0)
                if role in ("E", "T") and _rate_bp <= 3:
                    acc["maker"] += comm
                if role == "E" and status == "FILLED":
                    qty = Decimal(o.get("z") or "0")
                    avg = Decimal(o.get("ap") or "0")
                    acc.update(entry_ts=time.time(), entry_px=avg, qty=qty,
                               side=o.get("S"))
                    # настоящий «ВХОД» = позиция открыта (исполнение заявки)
                    d_side = "LONG" if o.get("S") == "BUY" else "SHORT"
                    tg.fire(f"{'🟢' if d_side == 'LONG' else '🔴'} "
                            f"<b>ВХОД {sym} {d_side}</b>\n"
                            f"{float(qty) * float(avg):.2f} USDC @ {avg}")
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
                    # дневные счётчики по направлениям + срабатывание лимита
                    d_side = {"BUY": "LONG", "SELL": "SHORT"}.get(acc.get("side", "?"))
                    if d_side:
                        self.daily_total_pnl += acc["pnl"]
                        self.daily_pnl[d_side] += acc["pnl"]
                        limit = self.balance_snapshot * self.cfg.daily_loss_pct
                        if (self.cfg.daily_loss_pct > 0 and not self.daily_locked[d_side]
                                and self.daily_pnl[d_side] <= -limit
                                and self.balance_snapshot):
                            self.daily_locked[d_side] = True
                            self.ledger.event("daily_limit_hit", sym,
                                              {"side": d_side,
                                               "pnl": str(self.daily_pnl[d_side]),
                                               "limit": str(limit)})
                            tg.fire(f"🛑 <b>Дневной лимит {d_side}</b>: убыток "
                                    f"{self.daily_pnl[d_side].quantize(Decimal('0.01')):+.2f} "
                                    f"USDC — новые входы {d_side} приостановлены "
                                    f"до 00:00 UTC")
                            log.warning("ДНЕВНОЙ ЛИМИТ %s: убыток %s <= -%s баланса — "
                                        "новые входы %s приостановлены до суточного "
                                        "сброса", d_side, self.daily_pnl[d_side].quantize(
                                            Decimal("0.01")),
                                        (self.cfg.daily_loss_pct * 100), d_side)
                    exit_pnl = float(acc.get("pnl", 0))
                    dur_h = (time.time() - acc.get("entry_ts", time.time())) / 3600
                    if exit_kind == "tp":
                        tg.fire(f"✅ <b>TP {sym}</b> {exit_pnl:+.2f} USDC ({dur_h:.1f} ч)")
                    else:
                        tg.fire(f"🔴 <b>СТОП {sym}</b> {exit_pnl:+.2f} USDC ({dur_h:.1f} ч)")
                    acts = self.strategy.exit_filled(sym, exit_kind)
                    await self.apply(acts)
            self.ledger.event("user_event", "", {"e": ev})
        except Exception:
            log.exception("user event error")

    # ---------------- суточный цикл (граница 00:00 UTC) ----------------
    def reset_daily(self) -> None:
        """Новый торговый день: свежий снимок баланса, обнуление счётчиков,
        новая граница дня (00:00 UTC — устойчива к рестартам процесса)."""
        self._day_key = datetime.now(timezone.utc).date()
        self._day_start_ms = int(datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        had = (self.daily_pnl["LONG"] or self.daily_pnl["SHORT"]
               or self.daily_locked["LONG"] or self.daily_locked["SHORT"])
        if had:
            log.info("суточный сброс (00:00 UTC): счётчики обнулены "
                     "(было: LONG %s, SHORT %s, блокировки %s/%s)",
                     self.daily_pnl["LONG"].quantize(Decimal("0.01")),
                     self.daily_pnl["SHORT"].quantize(Decimal("0.01")),
                     self.daily_locked["LONG"], self.daily_locked["SHORT"])
        self.daily_pnl = {"LONG": Decimal(0), "SHORT": Decimal(0)}
        self.daily_locked = {"LONG": False, "SHORT": False}
        self.daily_total_pnl = Decimal(0)
        self.daily_total_locked = False

    def _refresh_locks(self) -> None:
        """Пересчёт блокировок направленного дневного лимита по текущим цифрам."""
        if not self.balance_snapshot or self.cfg.daily_loss_pct <= 0:
            return
        limit = self.balance_snapshot * self.cfg.daily_loss_pct
        for side in ("LONG", "SHORT"):
            want = self.daily_pnl[side] <= -limit
            if want != self.daily_locked[side]:
                self.daily_locked[side] = want
                if want:
                    self.ledger.event("daily_limit_hit", "",
                                      {"side": side, "pnl": str(self.daily_pnl[side])})
                log.warning("дневной лимит %s: %s (pnl %s, лимит -%s)",
                            side, "ВКЛЮЧЁН" if want else "выключен",
                            self.daily_pnl[side].quantize(Decimal("0.01")),
                            limit.quantize(Decimal("0.01")))

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
                # сироты-входы: заявки scr-E-* без живого pending-состояния.
                # Рестарт теряет st.pending, а GTC-лимитка остаётся на книге и
                # может наполниться через часы на неуправляемый объём
                # (демо 22.09: SHIB/PEPE со вчера). Сверяем по client_id,
                # который появляется в pending ещё ДО выставления заявки.
                live_cids = {st.pending["client_id"]
                             for st in self.strategy.states.values()
                             if st.pending and st.pending.get("client_id")}
                for o in snap["orders"]:
                    cid = str(o.get("clientOrderId", ""))
                    if not cid.startswith("scr-E-") or cid in live_cids:
                        continue
                    log.warning("%s: сирота-заявка входа id=%s отменена "
                                "(состояние потеряно)", o["symbol"], cid)
                    self.ledger.event("cancel_entry", o["symbol"],
                                      {"reason": "orphan", "order_id": o["orderId"]})
                    if not self.cfg.dry_run:
                        await asyncio.to_thread(self.exec.cancel_order,
                                                o["symbol"], int(o["orderId"]))
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
                # авторитетная сверка дневного PnL с биржей (первый цикл + ежечасно)
                self._pnl_check_counter += 1
                if self._pnl_check_counter % 5 == 1:
                    auth = await asyncio.to_thread(
                        self.exec.daily_directional_pnl, self._day_start_ms)
                    dl = auth["LONG"] - self.daily_pnl["LONG"]
                    ds = auth["SHORT"] - self.daily_pnl["SHORT"]
                    if abs(dl) > Decimal("0.01") or abs(ds) > Decimal("0.01"):
                        log.warning("коррекция дневного PnL по бирже: LONG %s->%s, "
                                    "SHORT %s->%s",
                                    self.daily_pnl["LONG"].quantize(Decimal("0.01")),
                                    auth["LONG"].quantize(Decimal("0.01")),
                                    self.daily_pnl["SHORT"].quantize(Decimal("0.01")),
                                    auth["SHORT"].quantize(Decimal("0.01")))
                        self.daily_pnl = {"LONG": auth["LONG"], "SHORT": auth["SHORT"]}
                        self._refresh_locks()
                    else:
                        log.info("сверка дневного PnL с биржей: расхождений нет "
                                 "(LONG %s, SHORT %s)",
                                 self.daily_pnl["LONG"].quantize(Decimal("0.01")),
                                 self.daily_pnl["SHORT"].quantize(Decimal("0.01")))
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
        pos_side = side  # сторона ПОЗИЦИИ (LONG/SHORT)
        if not any(str(o.get("clientOrderId", "")).startswith("scr-T") for o in orders):
            log.warning("%s: TP отсутствует — ставлю @%s", sym, tp_px)
            oid = await asyncio.to_thread(self.exec.place_tp_limit, sym, exit_side,
                                          tp_px, abs(qty), f"scr-T-{sym}-repair",
                                          pos_side=pos_side)
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
            # HEDGE mode: stop для LONG-позиции закрывается SELL, positionSide=LONG
            # stop_market передаёт side ордера, positionSide выводится из него
            self.exits.setdefault(sym, {})["stop"] = oid
            if st.position:
                st.position["stop_id"] = oid

    async def send_hourly_report(self) -> None:
        """Почасовой отчёт в Telegram: чистый PnL дня и за всё время, позиции.
        «Чистыми» = все типы income (реализованный PnL, комиссии, фандинг,
        страховочный клиринг) — то, что реально видно по балансу счёта.
        Все числа — из income history биржи (источник правды)."""
        if not tg.TOKEN or not tg.CHAT_ID:
            return
        try:
            def fetch_income(start_ms, income_type=None):
                # пагинация: биржа отдаёт максимум 1000 строк за запрос
                cur, out = start_ms, []
                for _ in range(50):
                    kw = dict(start_time=cur, limit=1000)
                    if income_type:
                        kw["income_type"] = income_type
                    rp = unwrap(self.exec.client.rest_api.get_income_history(**kw).data())
                    rows = getattr(rp, "root", None) or rp
                    if not rows:
                        break
                    out.extend(r.model_dump(by_alias=True) for r in rows)
                    if len(rows) < 1000:
                        break
                    cur = int(out[-1].get("time", cur)) + 1
                return [r for r in out if str(r.get("symbol", "")).endswith("USDC")]

            def fl(r):
                return float(r.get("income", 0) or 0)

            # сегодня (с 00:00 UTC): все типы income по USDC-парам
            day_start = int(datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
            day_rows = fetch_income(day_start)
            day_realized = sum(fl(r) for r in day_rows
                               if r.get("incomeType") == "REALIZED_PNL")
            day_wins = sum(1 for r in day_rows
                           if r.get("incomeType") == "REALIZED_PNL" and fl(r) > 0)
            day_losses = sum(1 for r in day_rows
                             if r.get("incomeType") == "REALIZED_PNL" and fl(r) < 0)
            day_comm = sum(fl(r) for r in day_rows
                           if r.get("incomeType") == "COMMISSION")
            day_net = sum(fl(r) for r in day_rows)
            day_other = day_net - day_realized - day_comm

            # за всё время: чистое изменение счёта по income (с пагинацией)
            bot_start = int(datetime(2026, 9, 17, 11, 0, tzinfo=timezone.utc).timestamp() * 1000)
            total_net = sum(fl(r) for r in fetch_income(bot_start))

            # позиции: направления и нереализованный PnL
            snap = await asyncio.to_thread(self.exec.snapshot)
            longs, shorts = 0, 0
            upl = 0.0
            for p in snap["positions"]:
                amt = float(p.get("positionAmt", 0) or 0)
                if amt == 0 or not p.get("symbol", "").endswith("USDC"):
                    continue
                if amt > 0: longs += 1
                else: shorts += 1
                upl += float(p.get("unRealizedProfit", 0) or 0)

            env_tag = f"[{self.cfg.mode.value.upper()}]"
            lines = [f"📊 Отчёт {datetime.now(timezone.utc).strftime('%H:%M UTC')}"]
            lines.append(f"Сегодня чистыми: {day_net:+.2f} USDC (реализ. {day_realized:+.2f} "
                         f"[{day_wins}W/{day_losses}L], комисс. {day_comm:+.2f}, "
                         f"прочее {day_other:+.2f})")
            lines.append(f"Позиции: {longs} лонг / {shorts} шорт, нереализ. {upl:+.2f}")
            lines.append(f"За всё время чистыми: {total_net:+.2f} USDC "
                         f"— тот рост, что виден по балансу счёта")
            tg.fire("\n".join(lines))
        except Exception:
            log.exception("send_hourly_report")

    def log_universe_state(self) -> None:
        """П1: сколько пар торгуемо при текущем бюджете, минимум для юниверса."""
        if not self.balance_snapshot or self.balance_snapshot <= 0:
            return
        budget = self.balance_snapshot * self.cfg.risk_pct
        tradable, required = [], {}
        for s in self.symbols:
            h = self.feed.hist.get(s)
            if not h or not h.ready:
                continue
            try:
                stop = self.cfg.stop_atr_mult * h.atr_frac
            except (ValueError, ArithmeticError):
                continue
            if stop <= 0:
                continue
            need = self.filters[s].min_notional * self.cfg.notional_buffer \
                * stop / self.cfg.risk_pct
            required[s] = need
            if budget / stop >= self.filters[s].min_notional * self.cfg.notional_buffer:
                tradable.append(s)
        if not required:
            return
        min_sym = min(required, key=required.get)
        min_need = required[min_sym].quantize(Decimal("1"))
        if tradable:
            log.info("юниверс: торгуемых %d/%d при балансе %s; порог входа на самую "
                     "дешёвую пару %s USDC (на %s)", len(tradable), len(self.symbols),
                     self.balance_snapshot.quantize(Decimal("1")), min_need, min_sym)
        else:
            log.warning("юниверс: НЕ ХВАТАЕТ НИ НА ЧТО — бюджет %s USDC меньше пола "
                        "любой пары (минимум для входа: %s USDC на %s); бот ждёт "
                        "пополнения или снижения стопов", budget.quantize(Decimal("0.01")),
                        min_need, min_sym)

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
        # отчёт шлём на РОВНЫЙ ЧАС: при старте запоминаем текущий час
        # (не стреляя сразу), отчёт уходит при смене часа — :00 по UTC
        # (= :00 по МСК, UTC+3)
        last_report_hour = datetime.now(timezone.utc).hour
        last_timesync = 0.0
        while not self._stop.is_set():
            await asyncio.sleep(30)
            # почасовой отчёт в Telegram (на ровном часе)
            now_ts = time.time()
            if datetime.now(timezone.utc).hour != last_report_hour:
                last_report_hour = datetime.now(timezone.utc).hour
                try:
                    await self.send_hourly_report()
                except Exception:
                    log.exception("hourly telegram report")
            if datetime.now(timezone.utc).date() != self._day_key:
                # новый торговый день (00:00 UTC): свежий баланс + сброс счётчиков
                try:
                    await self.reset_daily()
                    self.balance_snapshot = await asyncio.to_thread(
                        self.exec.account_wallet_balance, "USDC")
                    log.info("снимок баланса обновлён: %s USDC", self.balance_snapshot)
                    self.log_universe_state()
                except Exception:
                    log.exception("суточный сброс")
            # синхронизация времени с биржей каждые 15 мин (анти -1021)
            if now_ts - last_timesync >= 900:
                last_timesync = now_ts
                try:
                    off = await asyncio.to_thread(
                        timesync.measure, self.exec.client.rest_api)
                    if abs(off) > 1000:
                        log.warning("время: офсет к бирже %+d мс — крупный дрейф",
                                    off)
                except Exception:
                    log.exception("timesync")
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
        self.log_universe_state()
        await self.feed.start_streams(self.symbols)
        await self.start_user_stream()
        self._tasks = [asyncio.create_task(self.reconciler()),
                       asyncio.create_task(self.watchdog())]
        log.info("бот запущен: пары=%d dry_run=%s цель=%s", len(self.symbols),
                 self.cfg.dry_run, self.cfg.target_pct)
        bars_seen = 0
        last_report = time.time()
        while not self._stop.is_set():
            sym, bar = await self.feed.closed_bars()
            bars_seen += 1
            if time.time() - last_report >= 300:      # пульс: бары капают из стрима
                log.info("пульс: %d закрытых баров за 5 мин", bars_seen)
                bars_seen = 0
                last_report = time.time()
            if self.cfg.smc_filter and sym in self.smc_struct:
                self.smc_struct[sym].push_1m(
                    bar.open_time, float(bar.high), float(bar.low),
                    float(bar.close), 0.0)
            h = self.feed.hist[sym]
            if h.ready:
                acts = self.strategy.on_closed_bar(h, bar)
                if acts and self.cfg.smc_filter and sym in self.smc_struct:
                    bias = self.smc_struct[sym].bias
                    kept = []
                    for a in acts:
                        if isinstance(a, PlaceEntry) and bias is not None:
                            want = 1 if a.side.value == "LONG" else -1
                            if bias != want:
                                self.strategy.state(sym).pending = None
                                self.ledger.event("smc_filter_skip", sym,
                                                  {"side": a.side.value,
                                                   "bias": bias})
                                log.warning("%s: SMC-фильтр — вход %s против "
                                            "структуры %s пропущен", sym,
                                            a.side.value, bias)
                                continue
                        kept.append(a)
                    acts = kept
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
    # Telegram инициализируется ПОСЛЕ загрузки .env (нужен cfg.mode)
    tg_token = os.environ.get("TELEGRAM_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if tg_token and tg_chat:
        tg.init(tg_token, tg_chat, env=cfg.mode.value)
        tg.fire(f"🤖 Бот запущен. Среда: {cfg.mode.value.upper()}")
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
