"""Стаканный бот: демо-исполнение (USDT-M) + самообучение раз в сутки.

ПРИНЦИПЫ ПРОЕКТА (как в основном боте):
- биржа — единственный источник правды: статусы/цены только из ответов
  биржи; реконсиляция на старте и раз в 10 минут (сироты усыновляются,
  внешние закрытия фиксируются честной пометкой);
- Decimal везде в деньгах, количества и цены по stepSize/tickSize биржи;
- сайзинг от потери: риск сделки = 0.15% баланса USDT; оценочный
  убыток = максимум(3 x спред, 3 бп) — нотионал = риск/убыток, с полом
  и потолком;
- дневной кап: -0.5% баланса -> стоп торговли до полуночи;
- 24/7: запуск через watchdog-скрипт (автоподъём при падении).

Логика: каждые 2с свежая строка признаков из lob.db, модель LightGBM
даёт вероятность «через 5 минут дороже».
  p >= GATE   -> LONG: post-only BUY по биду (мейкер!), не взяли за 60с
                 — отмена; держим 5 минут; выход SELL-лимиткой на аске
                 (мейкер), не взяли за 45с — маркет-фолбэк.
  p <= 1-GATE -> SHORT зеркально.
Самообучение: 00:15 UTC subprocess train.py new --save -> model.txt ->
горячая подмена; первое обучение через 6 часов данных (до этого тень).
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_CEILING
from pathlib import Path

import numpy as np

import bot.timesync as timesync
from binance_common.errors import Error as BinanceError

log = logging.getLogger("lob.engine")
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "lob" / "lob.db"
MODEL = ROOT / "data" / "lob" / "model.txt"

GATE = 0.62
HOLD_S = 300
RISK_PCT = Decimal("0.0015")      # риск одной сделки, доля баланса
DAY_CAP_PCT = Decimal("0.005")    # дневной кап убытка, доля баланса
MAX_NOTIONAL = Decimal("300")
MIN_NOTIONAL = Decimal("8")
MAX_SLOTS = 3
ENTRY_TTL_S = 60
EXIT_TTL_S = 45
RECON_S = 600
FIRST_TRAIN_H = 6
RETRAIN_AT = (0, 15)              # 00:15 UTC

FEATS = ["mid", "spread_bp", "microprice", "imb5", "imb10", "imb20",
         "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
         "ntr10", "vpin10", "d30", "d120"]


def api_code(e):
    sc = getattr(e, "status_code", None)
    if isinstance(sc, int) and sc < 0:
        return sc
    try:
        msg = str(e)
        return int(json.loads(msg[msg.index("{"):msg.rindex("}") + 1]).get("code"))
    except Exception:
        return None


class LobBot:
    def __init__(self, cfg, ex):
        self.cfg = cfg
        self.lob = cfg.lob
        self.ex = ex
        self.client = ex.client
        self.SYMS = list(self.lob.symbols)
        self.hedge = True
        self.model = None
        self.hist: dict[str, list[float]] = {}
        self.pos: dict[str, dict] = {}
        self.pending: dict[str, dict] = {}
        self.filters: dict[str, dict] = {}
        self.balance = Decimal(0)
        self.ASSET = "USDC"
        self.day_pnl = Decimal(0)
        self.day_key = datetime.now(timezone.utc).date()
        self.start_ts = time.time()
        self.last_retrain_day = None
        self.last_recon = 0.0
        DB.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(DB, timeout=10)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                ts INTEGER, symbol TEXT, p REAL, mid REAL,
                acted TEXT, order_id INTEGER);
            CREATE TABLE IF NOT EXISTS trades (
                symbol TEXT, side TEXT, entry_ts INTEGER, exit_ts INTEGER,
                entry_px REAL, exit_px REAL, qty REAL, pnl REAL,
                entry_fee TEXT, exit_fee TEXT, note TEXT);
        """)

    # ---------- биржа: источник правды ----------
    def resync(self) -> None:
        try:
            off = timesync.measure(self.client.rest_api)
            log.info("время: ресинк по бирже, офсет %+d мс", off)
        except Exception:
            pass

    def setup(self) -> None:
        try:
            r = self.client.rest_api.get_current_position_mode().data()
            self.hedge = bool(r.dual_side_position)
            log.info("режим позиции: %s", "HEDGE" if self.hedge else "ONE-WAY")
        except Exception as e:
            log.warning("режим позиции не прочитан (%s) — считаем HEDGE", e)
        info = self.client.rest_api.exchange_information().data()
        d = info.model_dump(by_alias=True)
        if self.cfg.lob.symbols:          # вселенная задана руками
            pass
        else:                              # НА ВСЕХ: все живые USDC-перпетуалы
            self.SYMS = sorted(s["symbol"] for s in d.get("symbols", [])
                               if s.get("symbol", "").endswith("USDC")
                               and s.get("status") == "TRADING"
                               and s.get("contractType") == "PERPETUAL")
            log.info("вселенная стакана: автообнаружено %d USDC-пар",
                     len(self.SYMS))
        for s in d.get("symbols", []):
            if s.get("symbol") in self.SYMS:
                fl = {}
                for f in s.get("filters", []):
                    t = f.get("filterType")
                    if t == "LOT_SIZE":
                        fl["step"] = Decimal(str(f["stepSize"]))
                        fl["min_qty"] = Decimal(str(f["minQty"]))
                    elif t == "PRICE_FILTER":
                        fl["tick"] = Decimal(str(f["tickSize"]))
                    elif t == "MIN_NOTIONAL":
                        fl["min_notional"] = Decimal(str(f.get("notional", 0)))
                self.filters[s["symbol"]] = fl
        for sym in self.SYMS:
            for attempt in range(2):
                try:
                    self.client.rest_api.change_initial_leverage(
                        symbol=sym, leverage=3)
                    break
                except BinanceError as e:
                    if api_code(e) == -1021 and attempt == 0:
                        self.resync()
                        continue
                    if api_code(e) != -4046:
                        log.warning("плечо %s: %s", sym, e)
            for attempt in range(2):
                try:
                    self.client.rest_api.change_margin_type(
                        symbol=sym, margin_type="ISOLATED")
                    break
                except BinanceError as e:
                    if api_code(e) == -1021 and attempt == 0:
                        self.resync()
                        continue
                    if api_code(e) != -4046:
                        log.warning("маржа %s: %s", sym, e)
        self.refresh_balance()
        log.info("настроено: фильтры %d/%d, баланс %s USDT",
                 len(self.filters), len(self.SYMS), self.balance)

    def refresh_balance(self) -> None:
        bal = self.client.rest_api.futures_account_balance_v3().data()
        rows = getattr(bal, "root", None) or bal
        for a in rows:
            d = a.model_dump(by_alias=True) if hasattr(a, "model_dump") else a
            if d.get("asset") == self.ASSET:
                self.balance = Decimal(str(d.get("balance", "0")))
                return
        log.warning("баланс USDT не найден")
        self.balance = Decimal(0)
        self.ASSET = "USDC"

    def recon(self) -> None:
        """Биржа против локального состояния: сироты усыновляем, внешние
        закрытия фиксируем. Каждые RECON_S и на старте."""
        try:
            pos = self.client.rest_api.position_information_v3().data()
            rows = getattr(pos, "root", None) or pos
            live: dict[str, dict] = {}
            for p in rows:
                d = p.model_dump(by_alias=True) if hasattr(p, "model_dump") else p
                sym = d.get("symbol")
                amt = Decimal(str(d.get("positionAmt", "0")))
                if sym in self.SYMS and amt != 0:
                    live[sym] = d
            for sym, d in live.items():
                if sym in self.pos or sym in self.pending:
                    continue
                amt = Decimal(str(d.get("positionAmt", "0")))
                log.warning("%s: позиция без состояния (%s %s) — усыновляю "
                            "и закрываю следующим циклом выхода", sym,
                            "LONG" if amt > 0 else "SHORT",
                            d.get("entryPrice"))
                self.pos[sym] = {"pside": "LONG" if amt > 0 else "SHORT",
                                 "entry_ts": 0.0,
                                 "px": float(d.get("entryPrice", 0)),
                                 "qty": abs(float(amt)),
                                 "adopted": True}
            for sym in list(self.pos) + list(self.pending):
                if sym not in live and sym not in self.pending:
                    pos_d = self.pos.get(sym)
                    if pos_d:
                        log.warning("%s: позиция закрыта ВНЕ бота — "
                                    "фиксирую с нулевой ценой выхода", sym)
                        self.finish(sym, pos_d, 0.0, "external",
                                    allow_zero=True)
                elif sym in self.pending and sym not in live:
                    self.cancel(sym, self.pending[sym]["oid"])
                    self.pending.pop(sym, None)
        except Exception:
            log.exception("реконсиляция")

    # ---------- сайзинг от потери ----------
    def notional_for(self, spread_bp: float) -> Decimal:
        if self.balance <= 0:
            return Decimal(0)
        risk = self.balance * RISK_PCT
        buffer = max(Decimal(str(spread_bp)) * 3,
                     Decimal("3")) / Decimal(10000)   # оценочный убыток
        notional = risk / buffer
        return min(max(notional, MIN_NOTIONAL), MAX_NOTIONAL)

    def qty_for(self, sym: str, px: Decimal, notional: Decimal) -> Decimal:
        fl = self.filters.get(sym, {})
        step = fl.get("step", Decimal("0.001"))
        min_qty = fl.get("min_qty", Decimal("0.001"))
        min_not = fl.get("min_notional", Decimal("5"))
        raw = notional / px
        if raw < min_qty or raw * px < min_not:
            raw = max(min_qty, (min_not / px).quantize(step, ROUND_DOWN))
        q = (raw / step).to_integral_value(rounding=ROUND_DOWN) * step
        return q

    def px_on_grid(self, sym: str, px: float, side: str) -> float:
        """Цена на тик-сетке биржи: BUY вниз, SELL вверх."""
        tick = self.filters.get(sym, {}).get("tick", Decimal("0.0001"))
        d = Decimal(str(px))
        q = (d / tick).to_integral_value(
            rounding=ROUND_DOWN if side == "BUY" else ROUND_CEILING) * tick
        return float(q)

    def order(self, sym: str, side: str, ptype: str, qty: Decimal,
              price: float | None = None, tif: str | None = None,
              pos_side: str | None = None) -> int | None:
        kw = dict(symbol=sym, side=side, type=ptype, quantity=float(qty))
        if self.hedge:
            kw["position_side"] = pos_side or ("LONG" if side == "BUY"
                                               else "SHORT")
        if ptype == "LIMIT":
            kw["price"] = price
            kw["time_in_force"] = tif or "GTC"
        for attempt in range(2):
            try:
                r = self.client.rest_api.new_order(**kw)
                return int(r.data().order_id)
            except BinanceError as e:
                if api_code(e) == -4131:
                    return None
                if api_code(e) == -1021 and attempt == 0:
                    self.resync()
                    continue
                log.warning("ордер %s %s: %s", sym, side, e)
                return None
        return None

    def order_info(self, sym: str, oid: int) -> dict | None:
        for _ in range(3):
            try:
                r = self.client.rest_api.query_order(symbol=sym, order_id=oid)
                d = r.data()
                return d.model_dump(by_alias=True) if hasattr(d, "model_dump") else d
            except Exception:
                time.sleep(0.3)
        return None

    def cancel(self, sym: str, oid: int) -> None:
        try:
            self.client.rest_api.cancel_order(symbol=sym, order_id=oid)
        except Exception:
            pass

    def book_top(self, sym: str) -> tuple[float, float] | None:
        try:
            r = self.client.rest_api.order_book(symbol=sym, limit=5).data()
            d = r.model_dump(by_alias=True) if hasattr(r, "model_dump") else {}
            return float(d["bids"][0][0]), float(d["asks"][0][0])
        except Exception as e:
            log.warning("стакан %s: %s", sym, e)
            return None

    # ---------- фичи ----------
    def feat_row(self, sym: str):
        cur = self.conn.execute(
            "select ts,mid,spread_bp,microprice,imb5,imb10,imb20,flow10_buy,"
            "flow10_sell,flow60_buy,flow60_sell,ntr10,vpin10 from feat "
            "where symbol=? order by ts desc limit 1", (sym,))
        row = cur.fetchone()
        if not row:
            return None
        (ts, mid, sp, micro, i5, i10, i20, f10b, f10s, f60b, f60s,
         ntr, vpin) = row
        h = self.hist.setdefault(sym, [])
        if not h or h[-1][0] != ts:
            h.append((ts, mid))
            if len(h) > 600:
                del h[:-600]
        def past_mid(lag_ms: int):
            target = ts - lag_ms
            for t, m in reversed(h):
                if abs(t - target) <= 2500:
                    return m
                if t < target - 2500:
                    break
            return None
        m30, m120 = past_mid(30_000), past_mid(120_000)
        d30 = mid / m30 - 1 if m30 else None
        d120 = mid / m120 - 1 if m120 else None
        x = dict(zip(FEATS, [mid, sp, micro, i5, i10, i20, f10b, f10s,
                             f60b, f60s, ntr, vpin, d30, d120]))
        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in x.values()):
            return None
        return x, mid, ts

    # ---------- модель ----------
    def maybe_train(self, now: datetime) -> None:
        """Сервер НЕ обучается (458 МБ RAM — обучение убивало контейнер
        OOM-киллером). Модель едет файлом с локальной машины; здесь только
        горячая подгрузка по mtime (как у ml5h)."""
        try:
            mt = MODEL.stat().st_mtime
        except Exception:
            return
        if mt != getattr(self, "_model_mtime", -1):
            self._model_mtime = mt
            self.try_load_model()

    def try_load_model(self) -> None:
        if self.model is None and MODEL.exists():
            try:
                import lightgbm as lgb
                self.model = lgb.Booster(model_file=str(MODEL))
                import bot.telegram as tg
                tg.fire("🧠 <b>Стакан</b>: модель получена — начинаю "
                        "оценивать очереди")
                log.info("модель загружена из файла")
            except Exception as e:
                log.warning("модель не загрузилась: %s", e)

    # ---------- сделки ----------
    def log_signal(self, sym: str, p: float, mid: float, acted: str,
                   oid: int | None) -> None:
        self.conn.execute("insert into signals values (?,?,?,?,?,?)",
                          (int(time.time() * 1000), sym, p, mid, acted, oid))
        self.conn.commit()

    def open_position(self, sym: str, side: str, p: float, mid: float,
                      spread_bp: float) -> None:
        top = self.book_top(sym)
        if not top:
            return
        bb, ba = top
        pside = "LONG" if side == "BUY" else "SHORT"
        px_f = self.px_on_grid(sym, bb if side == "BUY" else ba, side)
        notional = self.notional_for(spread_bp)
        qty = self.qty_for(sym, Decimal(str(px_f)), notional)
        if qty <= 0:
            return
        oid = self.order(sym, side, "LIMIT", qty, price=px_f, tif="GTX",
                         pos_side=pside)
        if oid is None:
            self.log_signal(sym, p, mid, "gtx_reject", None)
            return
        self.pending[sym] = {"pside": pside, "oid": oid, "qty": qty,
                             "ts": time.time(), "px": px_f}
        self.log_signal(sym, p, mid, pside, oid)

    def close_position(self, sym: str, pos: dict, maker: bool) -> None:
        side = "SELL" if pos["pside"] == "LONG" else "BUY"
        if maker:
            top = self.book_top(sym)
            if top:
                bb, ba = top
                px = self.px_on_grid(sym, ba if side == "SELL" else bb, side)
                oid = self.order(sym, side, "LIMIT", pos["qty"], price=px,
                                 tif="GTX", pos_side=pos["pside"])
                if oid is not None:
                    pos["exit_oid"] = oid
                    pos["exit_deadline"] = time.time() + EXIT_TTL_S
                    return
        oid = self.order(sym, side, "MARKET", pos["qty"],
                         pos_side=pos["pside"])
        info = self.order_info(sym, oid) if oid else None
        exit_px = float((info or {}).get("avgPrice") or 0)
        self.finish(sym, pos, exit_px, "taker_fb")

    def finish(self, sym: str, pos: dict, exit_px: float, exit_fee: str,
               allow_zero: bool = False) -> None:
        if exit_px <= 0:
            if allow_zero:
                exit_px = 0.0
            else:
                top = self.book_top(sym)
                exit_px = (top[0] + top[1]) / 2 if top else pos["px"]
                log.warning("%s: цена выхода не от биржи — взят мид %s",
                            sym, exit_px)
        q = Decimal(str(pos["qty"]))
        mult = Decimal(1) if pos["pside"] == "LONG" else Decimal(-1)
        pnl = (Decimal(str(exit_px)) - Decimal(str(pos["px"]))) * q * mult
        self.day_pnl += pnl
        self.conn.execute(
            "insert into trades values (?,?,?,?,?,?,?,?,?,?,?)",
            (sym, pos["pside"], int(pos["entry_ts"] * 1000),
             int(time.time() * 1000), pos["px"], exit_px, pos["qty"],
             float(pnl), "maker", exit_fee,
             "adopted" if pos.get("adopted") else ""))
        self.conn.commit()
        self.pos.pop(sym, None)
        log.info("ЗАКРЫТО %s %s @%s pnl=%+.3f (день %+.2f)",
                 sym, pos["pside"], exit_px, pnl, self.day_pnl)

    def manage(self, sym: str) -> None:
        pend = self.pending.get(sym)
        if pend:
            if time.time() - pend["ts"] > ENTRY_TTL_S:
                self.cancel(sym, pend["oid"])
                self.pending.pop(sym, None)
                return
            info = self.order_info(sym, pend["oid"])
            st = (info or {}).get("status")
            if st == "FILLED":
                px = float(info.get("avgPrice") or pend["px"])
                self.pending.pop(sym, None)
                self.pos[sym] = {**pend, "entry_ts": time.time(), "px": px}
                log.info("ОТКРЫТО %s %s @%s", sym, pend["pside"], px)
            elif st in ("CANCELED", "EXPIRED", "REJECTED"):
                self.pending.pop(sym, None)
            return
        pos = self.pos.get(sym)
        if not pos:
            return
        if "exit_oid" in pos:
            if time.time() > pos["exit_deadline"]:
                self.cancel(sym, pos["exit_oid"])
                self.close_position(sym, pos, maker=False)
                return
            info = self.order_info(sym, pos["exit_oid"])
            st = (info or {}).get("status")
            if st == "FILLED":
                self.finish(sym, pos, float(info.get("avgPrice") or 0),
                            "maker")
            elif st in ("CANCELED", "EXPIRED"):
                pos.pop("exit_oid", None)
            return
        if time.time() - pos["entry_ts"] >= HOLD_S:
            self.close_position(sym, pos, maker=True)

    # ---------- главный шаг ----------
    def step(self) -> None:
        now = datetime.now(timezone.utc)
        if now.date() != self.day_key:
            self.day_key = now.date()
            self.day_pnl = Decimal(0)
            self.refresh_balance()
        self.maybe_train(now)
        if time.time() - self.last_recon >= RECON_S:
            self.last_recon = time.time()
            self.recon()
        if self.day_pnl <= -self.balance * DAY_CAP_PCT:
            return
        for sym in self.SYMS:
            self.manage(sym)
        self.try_load_model()
        if self.model is None:
            return
        slots = len(self.pos) + len(self.pending)
        if slots >= MAX_SLOTS:
            return
        for sym in self.SYMS:
            if sym in self.pos or sym in self.pending or slots >= MAX_SLOTS:
                continue
            fr = self.feat_row(sym)
            if not fr:
                continue
            x, mid, _ts = fr
            p = float(self.model.predict(np.array([[x[f] for f in FEATS]]))[0])
            acted = ""
            if p >= GATE:
                self.open_position(sym, "BUY", p, mid, x["spread_bp"])
                acted = "LONG"
                slots += 1
            elif p <= 1 - GATE:
                self.open_position(sym, "SELL", p, mid, x["spread_bp"])
                acted = "SHORT"
                slots += 1
            elif p >= 0.58 or p <= 0.42:
                self.log_signal(sym, p, mid, "", None)


    async def run_loop(self) -> None:
        """Боевой цикл: шаг каждые 2 секунды, дневной кап 0.5% баланса."""
        while True:
            try:
                now = datetime.now(timezone.utc)
                if now.date() != self.day_key:
                    self.day_key = now.date()
                    self.day_pnl = Decimal(0)
                    self.refresh_balance()
                self.maybe_train(now)
                if time.time() - getattr(self, "_last_recon", 0) >= 600:
                    self._last_recon = time.time()
                    self.recon()
                if self.day_pnl <= -self.balance * Decimal("0.005"):
                    await asyncio.sleep(2)
                    continue
                for sym in self.SYMS:
                    self.manage(sym)
                self.try_load_model()
                if self.model is None:
                    await asyncio.sleep(2)
                    continue
                slots = len(self.pos) + len(self.pending)
                if slots >= self.lob.max_slots:
                    await asyncio.sleep(2)
                    continue
                for sym in self.SYMS:
                    if sym in self.pos or sym in self.pending \
                            or slots >= self.lob.max_slots:
                        continue
                    fr = self.feat_row(sym)
                    if not fr:
                        continue
                    x, mid, _ts = fr
                    p = float(self.model.predict(
                        np.array([[x[f] for f in FEATS]]))[0])
                    acted = ""
                    if p >= self.GATE:
                        self.open_position(sym, "BUY", p, mid, x["spread_bp"])
                        acted = "LONG"
                        slots += 1
                    elif p <= 1 - self.GATE:
                        self.open_position(sym, "SELL", p, mid,
                                           x["spread_bp"])
                        acted = "SHORT"
                        slots += 1
                    elif p >= 0.58 or p <= 0.42:
                        self.log_signal(sym, p, mid, "", None)
            except Exception:
                log.exception("step")
            await asyncio.sleep(2)


async def main() -> None:
    timesync.install()
    bot = LobBot()
    try:
        off = await asyncio.to_thread(timesync.measure, bot.client.rest_api)
        log.info("время: офсет к бирже %+d мс", off)
    except Exception as e:
        log.warning("timesync: %s", e)
    bot.setup()
    bot.recon()
    bot.try_load_model()
    log.info("лоб-бот запущен: gate %.2f, hold %ds, риск %s/сделка, "
             "модель %s", GATE, HOLD_S, RISK_PCT * 100,
             "есть" if bot.model else "ждёт данных")
    while True:
        try:
            bot.step()
        except Exception:
            log.exception("step")
        await asyncio.sleep(2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
