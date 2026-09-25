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
import lightgbm as lgb

import bot.telegram as tg
from ml5h.features import feature_row

log = logging.getLogger("ml5h")

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
        self.symbols = self.cfg.symbols or meta["symbols"]
        self.model = lgb.Booster(model_file=str(self.c.model_path))
        log.info("модель %s: %d фич, %d символов, gate %.2f",
                 self.c.model_path.name, len(self.feats), len(self.symbols),
                 meta.get("gate", self.c.gate))
        await asyncio.to_thread(self.ex.verify_and_set_leverage, self.symbols)
        for s in self.symbols:
            await asyncio.to_thread(self.ex.setup_symbol, s)
        await asyncio.to_thread(self.ex.build_filters, self.symbols)
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

    # ---------- цикл ----------
    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("tick")
            await asyncio.sleep(2)

    async def tick(self) -> None:
        now = datetime.now(timezone.utc)
        if now.date() != self.day_key:
            self.day_key = now.date()
            self.day_pnl = Decimal(0)
            self.balance = await asyncio.to_thread(
                self.ex.account_wallet_balance, "USDT")
        if self.day_pnl <= -self.balance * self.cfg.daily_cap_pct:
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
            else:
                log.info("сигнал %s p=%.3f — мимо порога", sym, p)

    # ---------- сделки ----------
    def notional(self) -> Decimal:
        by_risk = self.balance * self.c.risk_pct / self.c.buffer_pct
        by_cap = self.balance * self.c.notional_cap_pct
        return min(by_risk, by_cap)

    def qty_for(self, sym: str, px: Decimal) -> Decimal:
        fl = self.ex.filters.get(sym)
        if not fl or px <= 0:
            return Decimal(0)
        raw = self.notional() / px
        return fl.qty_for_notional(raw, px) or Decimal(0)

    async def open_long(self, sym: str, p: float) -> None:
        top = await asyncio.to_thread(self.ex.order_book_top, sym)
        if not top:
            return
        bb, _ba = top
        df = self.bars[sym]
        tick = float(df["close"].iloc[-1]) * 0  # цена — с биржи, не из баров
        qty = self.qty_for(sym, Decimal(str(bb)))
        if qty <= 0:
            log.warning("%s: лот не сошёлся", sym)
            return
        oid = await asyncio.to_thread(self.ex.place_entry_limit, sym,
                                      "BUY", Decimal(str(bb)), qty,
                                      f"ml5-{sym}-{int(time.time())}",
                                      post_only=True)
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
                             "p": p, "maker_entry": False}
            tg.fire(f"🟢 <b>ML5 ВХОД {sym}</b> {qty} @ {px} (taker) p={p:.2f}")
            return
        self.pending[sym] = {"oid": oid, "qty": qty, "px": bb, "p": p,
                             "ts": time.time()}
        log.info("%s: GTX-вход выставлен @%s qty=%s p=%.3f", sym, bb, qty, p)

    async def close_long(self, sym: str, pos: dict, maker: bool) -> None:
        top = await asyncio.to_thread(self.ex.order_book_top, sym)
        if maker and top:
            _bb, ba = top
            oid = await asyncio.to_thread(
                self.ex.place_tp_limit, sym, "SELL", Decimal(str(ba)),
                pos["qty"], f"ml5x-{sym}-{int(time.time())}",
                pos_side="LONG" if self.ex.hedge_mode else None)
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
                     exit_fee: str) -> None:
        if exit_px <= 0:
            top = await asyncio.to_thread(self.ex.order_book_top, sym)
            exit_px = (top[0] + top[1]) / 2 if top else pos["px"]
        q = Decimal(str(pos["qty"]))
        pnl = (Decimal(str(exit_px)) - Decimal(str(pos["px"]))) * q
        self.day_pnl += pnl
        self.pos.pop(sym, None)
        tg.fire(f"{'✅' if pnl > 0 else '🔻'} <b>ML5 ВЫХОД {sym}</b> "
                f"{pnl:.2f} USDT ({exit_fee})")
        log.info("ЗАКРЫТО %s @%s pnl=%.3f (день %.2f)", sym, exit_px, pnl,
                 self.day_pnl)

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
                self._fallback_market(sym, pend)
                continue
            info = await asyncio.to_thread(self.ex.query_order_full, sym,
                                           pend["oid"])
            if (info or {}).get("status") == "FILLED":
                px = float((info or {}).get("avgPrice") or pend["px"])
                self.pending.pop(sym, None)
                self.pos[sym] = {**pend, "px": px, "entry_ts": time.time(),
                                 "maker_entry": True}
                tg.fire(f"🟢 <b>ML5 ВХОД {sym}</b> {pend['qty']} @ {px} "
                        f"(мейкер) p={pend['p']:.2f}")
        for sym, pos in list(self.pos.items()):
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
                        await self.close_long(sym, pos, maker=False)
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
                await self.close_long(sym, pos, maker=True)

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
