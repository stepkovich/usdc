"""Двигатель ML-5Ч: топ-20 USDT, сигнал каждые 30 минут, холд 5 часов.

Каждые 30 минут (закрытие бара) пересчитываем признаки по всей истории
в памяти и спрашиваем модель. p > gate -> LONG: вход post-only по биду
(TTL), не взяли — маркет (сигнал «купить сейчас»). Выход мейкером
за минуту до горизонта, фолбэк маркетом.
Сайзинг от потери: notional = min(2% баланса, риск 0.15% / буфер 2%).
Дневной кап 0.5% баланса. Decimal + тик/шаг биржи.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from binance_common.errors import Error as BinanceError
from bot.executor import api_code

import bot.telegram as tg
from ml5h.features import feature_row

log = logging.getLogger("ml5h")

MIN_PRICE = Decimal("0.0001")   # ниже — цена не сериализуется в API
                                # без «научной записи», биржа даёт -1102

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "n", "taker_buy_base",
              "taker_buy_quote", "ignore"]


class Ml5hEngine:
    def __init__(self, cfg, ex):
        self.cfg = cfg
        self.ex = ex
        self.c = cfg.ml5h
        self.model: lgb.Booster | None = None
        self.feats: list[str] = []
        self.symbols: list[str] = []
        self.bars: dict[str, pd.DataFrame] = {}
        self.btc_close = None
        self.pos: dict[str, dict] = {}
        self.pending: dict[str, dict] = {}
        self.day_pnl = Decimal(0)
        self.day_key = datetime.now(timezone.utc).date()
        self.balance = Decimal(0)
        self._last_bar_ms = 0

    # ---------- запуск ----------
    async def setup(self) -> None:
        meta = json.loads(self.c.meta_path.read_text())
        self.feats = meta["features"]
        self.symbols = self.c.symbols or meta["symbols"]
        self.model = lgb.Booster(model_file=str(self.c.model_path))
        log.info("модель %s: %d фич, %d символов, gate %.2f",
                 self.c.model_path.name, len(self.feats), len(self.symbols),
                 meta.get("gate", self.c.gate))
        # биржа — источник правды: символы, которых нет на демо, отсеиваем
        await asyncio.to_thread(self.ex.build_filters, self.symbols)
        live = [s for s in self.symbols if s in self.ex.filters]
        for s in self.symbols:
            if s not in live:
                log.warning("%s: нет на демо-бирже — исключён", s)
        self.symbols = live
        await asyncio.to_thread(self.ex.verify_and_set_leverage, self.symbols)
        for s in self.symbols:
            await asyncio.to_thread(self.ex.setup_symbol, s)
        self.balance = await asyncio.to_thread(
            self.ex.account_wallet_balance, "USDT")
        await self._warmup()
        log.info("тёплый старт: %d/%d символов, баланс %s USDT",
                 len(self.bars), len(self.symbols), self.balance)

    async def _warmup(self) -> None:
        for sym in self.symbols:
            rows = await asyncio.to_thread(
                self.ex.fetch_klines, sym, "30m", self.c.warmup_bars)
            rows = rows[:-1]                      # последний бар ещё идёт
            if len(rows) < 300:
                log.warning("%s: мало истории (%d)", sym, len(rows))
                continue
            df = pd.DataFrame(rows, columns=KLINE_COLS)
            for c in ("open", "high", "low", "close", "volume",
                      "quote_volume", "taker_buy_base"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["open_time"] = df["open_time"].astype(np.int64)
            df.index = pd.to_datetime(df["open_time"], unit="ms")
            self.bars[sym] = df[["open_time", "open", "high", "low", "close",
                                 "volume", "quote_volume",
                                 "taker_buy_base"]]
        btc = self.bars.get("BTCUSDT")
        self.btc_close = btc["close"] if btc is not None else None
        # сверхдешёвые монеты: цена не проходит сериализацию API (-1102)
        dead = [s for s in list(self.bars)
                if float(self.bars[s]["close"].iloc[-1]) < float(MIN_PRICE)]
        for s in dead:
            self.bars.pop(s, None)
            if s in self.symbols:
                self.symbols.remove(s)
        if dead:
            log.info("исключены сверхдешёвые (цена <%s): %s",
                     MIN_PRICE, ", ".join(dead))

    # ---------- цикл ----------
    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("tick")
            await asyncio.sleep(2)

    def _model_mtime(self) -> float:
        try:
            return self.c.model_path.stat().st_mtime
        except Exception:
            return 0.0

    def maybe_reload_model(self) -> None:
        mt = self._model_mtime()
        if mt and mt != getattr(self, "_model_loaded_mtime", -1):
            try:
                import lightgbm as lgb
                self.model = lgb.Booster(model_file=str(self.c.model_path))
                # мета едет ВМЕСТЕ с моделью: признаки и вселенная могут
                # измениться (иначе предсказание падает по числу колонок)
                meta = json.loads(self.c.meta_path.read_text())
                self.feats = meta["features"]
                new_syms = [s for s in (self.c.symbols or meta["symbols"])
                            if s not in self.symbols]
                self.symbols = self.c.symbols or meta["symbols"]
                self._model_loaded_mtime = mt
                import bot.telegram as tg
                tg.fire(f"🧠 <b>ML-5ч</b>: новая модель получена "
                        f"({len(self.feats)} признаков, "
                        f"{len(self.symbols)} монет) — подхвачена "
                        f"без рестарта")
                log.info("модель перезагружена (обновлена %s): %d фич, "
                         "%d символов (+%d новых)",
                         datetime.fromtimestamp(mt, timezone.utc)
                         .strftime("%H:%M UTC"), len(self.feats),
                         len(self.symbols), len(new_syms))
            except Exception as e:
                log.warning("перезагрузка модели не удалась: %s", e)

    async def tick(self) -> None:
        now = datetime.now(timezone.utc)
        self.maybe_reload_model()
        if now.date() != self.day_key:
            self.day_key = now.date()
            self.day_pnl = Decimal(0)
            self.balance = await asyncio.to_thread(
                self.ex.account_wallet_balance, "USDT")
        if now.date() != self.day_key or True:
            pass
        if time.time() - getattr(self, "_last_recon", 0) >= 600:
            self._last_recon = time.time()
            await self.recon()
        day_cap = (min(self.c.day_cap_usdc, self.balance * self.cfg.daily_cap_pct)
                   if self.c.day_cap_usdc > 0
                   else self.balance * self.cfg.daily_cap_pct)
        if self.day_pnl <= -day_cap:
            await self.manage()
            return
        await self.manage()
        # новый закрытый 30м бар?
        bar_ms = int(now.timestamp() * 1000) // (self.c.bar_minutes * 60_000)
        if bar_ms == self._last_bar_ms:
            return
        prev_close_boundary = bar_ms * self.c.bar_minutes * 60_000
        if now.timestamp() * 1000 - prev_close_boundary < 5_000:
            return                    # ждём 5с после границы — честное закрытие
        self._last_bar_ms = bar_ms
        await self.on_bar()
        # новые символы из обновлённой меты: тёплый старт истории
        new = [s for s in self.symbols if s not in self.bars
               and s not in getattr(self, "_warming", set())]
        if new:
            self._warming = getattr(self, "_warming", set()) | set(new)
            try:
                await self._warmup_new(new)
            finally:
                self._warming -= set(new)

    async def _warmup_new(self, syms: list[str]) -> None:
        """Тёплый старт для символов, появившихся с новой метой."""
        log.info("прогрев %d новых символов...", len(syms))
        for sym in syms:
            rows = await asyncio.to_thread(
                self.ex.fetch_klines, sym, "30m", self.c.warmup_bars)
            rows = rows[:-1]
            if len(rows) < 300:
                continue
            df = pd.DataFrame(rows, columns=KLINE_COLS)
            for c in ("open", "high", "low", "close", "volume",
                      "quote_volume", "taker_buy_base"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["open_time"] = df["open_time"].astype(np.int64)
            df.index = pd.to_datetime(df["open_time"], unit="ms")
            self.bars[sym] = df[["open_time", "open", "high", "low", "close",
                                 "volume", "quote_volume",
                                 "taker_buy_base"]]
        btc = self.bars.get("BTCUSDT")
        self.btc_close = btc["close"] if btc is not None else None
        log.info("прогрев завершён: %d символов в работе", len(self.bars))

    async def on_bar(self) -> None:
        for sym in self.symbols:
            rows = await asyncio.to_thread(
                self.ex.fetch_klines, sym, "30m", 3)
            if not rows:
                continue
            rows = rows[:-1]
            r = rows[-1]
            df = self.bars.get(sym)
            if df is None:
                continue
            if int(r[0]) <= int(df["open_time"].iloc[-1]):
                continue              # бар уже есть
            new = pd.DataFrame([r], columns=KLINE_COLS)
            for c in ("open", "high", "low", "close", "volume",
                      "quote_volume", "taker_buy_base"):
                new[c] = pd.to_numeric(new[c], errors="coerce")
            new["open_time"] = new["open_time"].astype(np.int64)
            new.index = pd.to_datetime(new["open_time"], unit="ms")
            self.bars[sym] = pd.concat([df, new])[-1200:]
        btc = self.bars.get("BTCUSDT")
        self.btc_close = btc["close"] if btc is not None else None
        slots = len(self.pos) + len(self.pending)
        for sym in self.symbols:
            if sym not in self.bars or sym in self.pos or sym in self.pending:
                continue
            if slots >= self.c.max_slots:
                break
            x = await asyncio.to_thread(feature_row, self.bars[sym],
                                        self.btc_close, sym, self.feats)
            if not x:
                continue
            p = float(self.model.predict(np.array([[x[f] for f in self.feats]]))[0])
            if p > self.c.gate:
                await self.open_long(sym, p)
                slots += 1
            elif p < 1 - self.c.gate:
                await self.open_short(sym, p)
                slots += 1
            else:
                log.info("сигнал %s p=%.3f — мимо порога", sym, p)

    # ---------- сделки ----------
    def risk_budget(self) -> Decimal:
        """Гибрид: денежный риск (если задан) с процентным потолком.
        Демо-репетиция реала-100: риск 0.10 → позиции ~5 USDC."""
        base = self.balance * self.c.risk_pct
        if self.c.risk_usdc > 0:
            return min(self.c.risk_usdc, base)
        return base

    def notional(self) -> Decimal:
        by_risk = self.risk_budget() / self.c.buffer_pct
        by_cap = self.balance * self.c.notional_cap_pct
        return max(min(by_risk, by_cap), Decimal("5"))   # пол = минимум биржи

    def qty_for(self, sym: str, px: Decimal) -> Decimal:
        """qty_for_notional принимает НОТИОНАЛ в USDT и сам делит на цену
        (баг 28.09: двойное деление давало заявки в 4+ раза больше расчёта)."""
        fl = self.ex.filters.get(sym)
        if not fl or px <= 0:
            return Decimal(0)
        return fl.qty_for_notional(self.notional(), px) or Decimal(0)

    async def open_long(self, sym: str, p: float) -> None:
        top = await asyncio.to_thread(self.ex.order_book_top, sym)
        if not top:
            return
        bb, _ba = top
        if bb < float(MIN_PRICE):
            return
        df = self.bars[sym]
        tick = float(df["close"].iloc[-1]) * 0  # цена — с биржи, не из баров
        qty = self.qty_for(sym, Decimal(str(bb)))
        if qty <= 0:
            log.warning("%s: лот не сошёлся — символ исключён", sym)
            if sym in self.symbols:
                self.symbols.remove(sym)
            return
        try:
            oid = await asyncio.to_thread(self.ex.place_entry_limit, sym,
                                          "BUY", Decimal(str(bb)), qty,
                                          f"ml5-{sym}-{int(time.time())}",
                                          post_only=True)
        except BinanceError as e:
            if api_code(e) in (-4411, -1121):
                log.warning("%s: биржа требует соглашения TradFi — символ "
                            "исключён", sym)
                if sym in self.symbols:
                    self.symbols.remove(sym)
                return
            raise
        if oid is None:
            # GTX отклонён (цена ушла сквозь) — сигнал «купить сейчас»:
            # исполняемся маркетом, как договорено механикой сигнала
            oid = await asyncio.to_thread(self.ex.place_market, sym, "BUY",
                                          qty)
            if oid is None:
                return
            info = await asyncio.to_thread(self.ex.query_order_full, sym, oid)
            px = float((info or {}).get("avgPrice") or 0)
            self.pos[sym] = {"qty": qty, "px": px, "entry_ts": time.time(),
                             "p": p, "maker_entry": False, "side": "LONG"}
            tg.fire(f"🟢 <b>ML5 ВХОД {sym}</b> {qty} @ {px} (taker) p={p:.2f}")
            return
        self.pending[sym] = {"oid": oid, "qty": qty, "px": bb, "p": p,
                             "ts": time.time(), "side": "LONG"}
        log.info("%s: GTX-вход выставлен @%s qty=%s p=%.3f", sym, bb, qty, p)

    async def open_short(self, sym: str, p: float) -> None:
        """Зеркало open_long: SELL post-only по аску, taker-фолбэк."""
        top = await asyncio.to_thread(self.ex.order_book_top, sym)
        if not top:
            return
        _bb, ba = top
        if ba < float(MIN_PRICE):
            return
        qty = self.qty_for(sym, Decimal(str(ba)))
        if qty <= 0:
            log.warning("%s: лот не сошёлся — символ исключён", sym)
            if sym in self.symbols:
                self.symbols.remove(sym)
            return
        try:
            oid = await asyncio.to_thread(self.ex.place_entry_limit, sym,
                                          "SELL", Decimal(str(ba)), qty,
                                          f"ml5-{sym}-{int(time.time())}",
                                          post_only=True)
        except BinanceError as e:
            if api_code(e) in (-4411, -1121):
                log.warning("%s: биржа требует соглашения TradFi — символ "
                            "исключён", sym)
                if sym in self.symbols:
                    self.symbols.remove(sym)
                return
            raise
        if oid is None:
            oid = await asyncio.to_thread(self.ex.place_market, sym, "SELL",
                                          qty)
            if oid is None:
                return
            info = await asyncio.to_thread(self.ex.query_order_full, sym, oid)
            px = float((info or {}).get("avgPrice") or 0)
            self.pos[sym] = {"qty": qty, "px": px, "entry_ts": time.time(),
                             "p": p, "maker_entry": False, "side": "SHORT"}
            tg.fire(f"🔴 <b>ML5 ВХОД {sym}</b> {qty} @ {px} (taker) p={p:.2f}")
            return
        self.pending[sym] = {"oid": oid, "qty": qty, "px": ba, "p": p,
                             "ts": time.time(), "side": "SHORT"}
        log.info("%s: GTX-шорт выставлен @%s qty=%s p=%.3f", sym, ba, qty, p)

    async def close_short(self, sym: str, pos: dict, maker: bool) -> None:
        """Зеркало close_long: BUY лимиткой по биду, фолбэк маркет BUY."""
        top = await asyncio.to_thread(self.ex.order_book_top, sym)
        if maker and top:
            bb, _ba = top
            try:
                oid = await asyncio.to_thread(
                    self.ex.place_tp_limit, sym, "BUY", Decimal(str(bb)),
                    pos["qty"], f"ml5x-{sym}-{int(time.time())}",
                    pos_side="SHORT" if self.ex.hedge_mode else None)
            except BinanceError as e:
                if api_code(e) == -2022:
                    log.warning("%s: -2022 при выходе — позиция уже закрыта, "
                                "снимаю с учёта", sym)
                    self.pos.pop(sym, None)
                    return
                raise
            if oid is not None:
                pos["exit_oid"] = oid
                pos["exit_deadline"] = time.time() + self.c.exit_ttl_s
                return
        oid = await asyncio.to_thread(self.ex.place_market, sym, "BUY",
                                      pos["qty"])
        info = await asyncio.to_thread(self.ex.query_order_full, sym,
                                       oid) if oid else None
        exit_px = float((info or {}).get("avgPrice") or 0)
        await self.finish(sym, pos, exit_px, "taker")

    async def close_any(self, sym: str, pos: dict, maker: bool) -> None:
        if pos.get("side") == "SHORT":
            await self.close_short(sym, pos, maker)
        else:
            await self.close_long(sym, pos, maker)

    async def close_long(self, sym: str, pos: dict, maker: bool) -> None:
        top = await asyncio.to_thread(self.ex.order_book_top, sym)
        if maker and top:
            _bb, ba = top
            try:
                oid = await asyncio.to_thread(
                    self.ex.place_tp_limit, sym, "SELL", Decimal(str(ba)),
                    pos["qty"], f"ml5x-{sym}-{int(time.time())}",
                    pos_side="LONG" if self.ex.hedge_mode else None)
            except BinanceError as e:
                if api_code(e) == -2022:
                    # позиции на бирже уже нет (закрыта вне цикла/ранее)
                    log.warning("%s: -2022 при выходе — позиция уже закрыта, "
                                "снимаю с учёта", sym)
                    self.pos.pop(sym, None)
                    return
                raise
            if oid is not None:
                pos["exit_oid"] = oid
                pos["exit_deadline"] = time.time() + self.c.exit_ttl_s
                return
        oid = await asyncio.to_thread(self.ex.place_market, sym, "SELL",
                                      pos["qty"])
        info = await asyncio.to_thread(self.ex.query_order_full, sym,
                                       oid) if oid else None
        exit_px = float((info or {}).get("avgPrice") or 0)
        await self.finish(sym, pos, exit_px, "taker")

    async def finish(self, sym: str, pos: dict, exit_px: float,
                     exit_fee: str, allow_zero: bool = False) -> None:
        if exit_px <= 0:
            if allow_zero:
                exit_px = 0.0
            else:
                top = await asyncio.to_thread(self.ex.order_book_top, sym)
                exit_px = (top[0] + top[1]) / 2 if top else pos["px"]
                log.warning("%s: цена выхода не от биржи — мид %s", sym,
                            exit_px)
        q = Decimal(str(pos["qty"]))
        mult = Decimal(1) if pos.get("side", "LONG") == "LONG" else Decimal(-1)
        pnl = (Decimal(str(exit_px)) - Decimal(str(pos["px"]))) * q * mult
        self.day_pnl += pnl
        self.pos.pop(sym, None)
        tg.fire(f"{'✅' if pnl > 0 else '🔻'} <b>ML5 ВЫХОД {sym}</b> "
                f"{pnl:.2f} USDT ({exit_fee})")
        log.info("ЗАКРЫТО %s @%s pnl=%.3f (день %.2f)", sym, exit_px, pnl,
                 self.day_pnl)

    async def recon(self) -> None:
        """Биржа — источник правды: раз в 10 минут сверяем позиции.
        Чужая (после рестарта) — усыновляем и немедленно закрываем;
        наша, которой на бирже нет — фиксируем внешнее закрытие."""
        try:
            rows = await asyncio.to_thread(
                self.ex.position_information_for, self.symbols)
            live = {d["symbol"]: d for d in rows
                    if Decimal(str(d.get("positionAmt", "0"))) != 0}
        except Exception:
            log.exception("recon")
            return
        for sym, d in live.items():
            if sym in self.pos or sym in self.pending:
                continue
            amt = Decimal(str(d.get("positionAmt", "0")))
            qty = abs(amt)
            log.warning("%s: позиция без состояния (%s) — усыновляю и "
                        "закрываю", sym, d.get("entryPrice"))
            self.pos[sym] = {"qty": qty, "px": float(d.get("entryPrice", 0)),
                             "entry_ts": 0, "adopted": True,
                             "side": "LONG" if amt > 0 else "SHORT"}
        for sym in list(self.pos):
            if sym not in live:
                pos = self.pos[sym]
                log.warning("%s: закрыта вне бота — фиксирую", sym)
                await self.finish(sym, pos, 0.0, "external", allow_zero=True)

    async def manage(self) -> None:
        for sym, pend in list(self.pending.items()):
            if time.time() - pend["ts"] > self.c.entry_ttl_s:
                await asyncio.to_thread(self.ex.cancel_order, sym,
                                        pend["oid"])
                info = await asyncio.to_thread(self.ex.query_order_full, sym,
                                               pend["oid"])
                if (info or {}).get("status") == "FILLED":
                    px = float((info or {}).get("avgPrice") or pend["px"])
                    self.pos[sym] = {**pend, "px": px,
                                     "entry_ts": time.time(),
                                     "maker_entry": True}
                    self.pending.pop(sym, None)
                    tg.fire(f"🟢 <b>ML5 ВХОД {sym}</b> {pend['qty']} @ {px} "
                            f"(мейкер) p={pend['p']:.2f}")
                    continue
                self.pending.pop(sym, None)
                info = await asyncio.to_thread(self.ex.query_order_full, sym,
                                               pend["oid"])
                exq = Decimal(str((info or {}).get("executedQty") or 0))
                if exq > 0:
                    # ЧАСТИЧНОЕ исполнение: остаток позиции усыновляем
                    px = float((info or {}).get("avgPrice") or pend["px"])
                    self.pos[sym] = {"qty": exq, "px": px,
                                     "entry_ts": time.time(),
                                     "maker_entry": True, "partial": True}
                    tg.fire(f"🟡 <b>ML5 {sym}</b> частичный вход {exq} @ {px} "
                            f"— веду остаток")
                    await self.close_any(sym, self.pos[sym], maker=True)
                else:
                    self._fallback_market(sym, pend)
                continue
            info = await asyncio.to_thread(self.ex.query_order_full, sym,
                                           pend["oid"])
            if (info or {}).get("status") == "FILLED":
                px = float((info or {}).get("avgPrice") or pend["px"])
                self.pending.pop(sym, None)
                self.pos[sym] = {**pend, "px": px, "entry_ts": time.time(),
                                 "maker_entry": True,
                                 "side": pend.get("side", "LONG")}
                tg.fire(f"🟢 <b>ML5 ВХОД {sym}</b> {pend['qty']} @ {px} "
                        f"(мейкер) p={pend['p']:.2f}")
        for sym, pos in list(self.pos.items()):
            # АВАРИЙНЫЙ ВЫХОД (предрегистрация 28.09 по MAE-исследованию:
            # глубже -15% ныряют 0.27% сделок, их финал -11.7%, выживают 6%).
            # Закрываем по рынку, поверх остаётся биржевая ликвидация.
            top = await asyncio.to_thread(self.ex.order_book_top, sym)
            if top:
                bid, ba = top
                px0 = Decimal(str(pos["px"]))
                if pos.get("side") == "SHORT":
                    loss = (Decimal(str(ba)) - px0) / px0
                else:
                    loss = (px0 - Decimal(str(bid))) / px0
                if loss >= self.c.emergency_stop_pct:
                    log.warning("%s: АВАРИЙНЫЙ ВЫХОД -%.1f%% (порог %s%%)",
                                sym, float(loss) * 100,
                                self.c.emergency_stop_pct * 100)
                    tg.fire(f"🛑 <b>ML5 АВАРИЙНЫЙ ВЫХОД {sym}</b> "
                            f"−{float(loss)*100:.1f}%")
                    await self.close_any(sym, pos, maker=False)
                    continue
            if "exit_oid" in pos:
                if time.time() > pos["exit_deadline"]:
                    await asyncio.to_thread(self.ex.cancel_order, sym,
                                            pos["exit_oid"])
                    info = await asyncio.to_thread(self.ex.query_order_full,
                                                   sym, pos["exit_oid"])
                    if (info or {}).get("status") == "FILLED":
                        await self.finish(sym, pos,
                                          float((info or {}).get("avgPrice")
                                                or 0), "maker")
                    else:
                        await self.close_any(sym, pos, maker=False)
                else:
                    info = await asyncio.to_thread(self.ex.query_order_full,
                                                   sym, pos["exit_oid"])
                    if (info or {}).get("status") == "FILLED":
                        await self.finish(sym, pos,
                                          float((info or {}).get("avgPrice")
                                                or 0), "maker")
                continue
            horizon = pos["entry_ts"] + self.c.hold_bars \
                * self.c.bar_minutes * 60 - 60
            if time.time() >= horizon:
                await self.close_any(sym, pos, maker=True)

    async def _fallback_market(self, sym: str, pend: dict) -> None:
        oid = await asyncio.to_thread(self.ex.place_market, sym, "BUY",
                                      pend["qty"])
        info = await asyncio.to_thread(self.ex.query_order_full, sym,
                                       oid) if oid else None
        px = float((info or {}).get("avgPrice") or 0)
        if px > 0:
            self.pos[sym] = {**pend, "px": px, "entry_ts": time.time(),
                             "maker_entry": False}
            tg.fire(f"🟢 <b>ML5 ВХОД {sym}</b> {pend['qty']} @ {px} "
                    f"(taker) p={pend['p']:.2f}")
