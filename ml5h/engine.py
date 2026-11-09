"""Двигатель ML-5Ч: все живые USDT-перпетуалы (~515), сигнал каждые
30 минут, холд 5 часов.

Каждые 30 минут (закрытие бара) пересчитываем признаки по всей истории
в памяти и спрашиваем модель. p > gate -> LONG: вход post-only по биду
(TTL), не взяли — маркет (сигнал «купить сейчас»). Выход мейкером
за минуту до горизонта, фолбэк маркетом.
Сайзинг (владелец v3 02.11): notional = баланс × 50% × плечо 3 × 10%,
но не ниже мин-ордера монеты × 1.1; слоты 10; дневной кап 4% баланса.
Дневной кап 0.5% баланса. Decimal + тик/шаг биржи.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
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
        # ЖУРНАЛ СДЕЛОК: источник для сигнала переобучения (последние 30
        # сделок) и живой статистики. Биржа — источник правды по факту,
        # журнал — наш слой с уверенностью модели в момент входа.
        self.trade_db = self.cfg.data_dir / "ml5h_trades.db"
        self.trade_db.parent.mkdir(parents=True, exist_ok=True)
        self.tconn = sqlite3.connect(self.trade_db, timeout=30)
        self.tconn.execute("""CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY, ts REAL, symbol TEXT, side TEXT,
            entry_px REAL, exit_px REAL, qty REAL, pnl REAL,
            confidence REAL, exit_kind TEXT)""")
        self.tconn.commit()
        self._last_bar_ms = 0

    # ---------- запуск ----------
    def _seed_model(self) -> None:
        """Первый запуск на чистом сервере: модель из образа едет на том
        данных (дальше живёт там и обновляется без пересборки)."""
        if self.c.model_path.exists() and self.c.meta_path.exists():
            return
        self.c.model_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        base = Path(__file__).resolve().parent.parent / "models"
        src_m, src_j = base / "ml5h.txt", base / "ml5h_meta.json"
        if not (src_m.exists() and src_j.exists()):
            raise RuntimeError(
                "модель не найдена ни на томе данных, ни в образе — "
                "положи ml5h.txt и ml5h_meta.json в data/models/")
        shutil.copy(src_m, self.c.model_path)
        shutil.copy(src_j, self.c.meta_path)
        log.info("модель засеяна из образа на том данных")

    async def setup(self) -> None:
        self._seed_model()
        meta = json.loads(self.c.meta_path.read_text())
        self.feats = meta["features"]
        self.symbols = self.c.symbols or meta["symbols"]
        # порог принадлежит МОДЕЛИ (у модели-бури 0.65, у старой 0.55):
        # едет в артефакте вместе с ней, чтобы они не разъехались
        if meta.get("gate"):
            self.c.gate = float(meta["gate"])
        self.model = lgb.Booster(model_file=str(self.c.model_path))
        # калибровка (07.11): карта «сырая уверенность -> настоящий шанс».
        # Файл может отсутствовать — тогда работает сырой гейт 0.65.
        self._cal_x = self._cal_y = None
        try:
            cal = json.loads((self.c.model_path.parent
                              / "calibration.json").read_text())
            self._cal_x = np.array(cal["iso_x"], dtype="float64")
            self._cal_y = np.array(cal["iso_y"], dtype="float64")
            log.info("калибровка загружена (%d точек) — гейт в настоящих %%",
                     len(self._cal_x))
        except Exception as e:
            log.info("калибровки нет (%s) — гейт по сырой уверенности",
                     str(e)[:50])
        log.info("модель %s: %d фич, %d символов, gate %.2f",
                 self.c.model_path.name, len(self.feats), len(self.symbols),
                 self.c.gate)
        self.meta = None
        self.meta_veto = 0
        self.meta_pass = 0
        if self.c.meta_gate:
            try:
                self.meta = lgb.Booster(
                    model_file=str(self.c.meta_model_path))
                log.info("мета: замок загружен (%s) — вето при p_win<0.5",
                         self.c.meta_model_path.name)
                self._meta_loaded_mtime = self._meta_mtime()
            except Exception as e:
                log.warning("мета: не загрузилась (%s) — торгую без неё",
                            str(e)[:80])
        # биржа — источник правды: символы, которых нет на демо, отсеиваем
        await asyncio.to_thread(self.ex.build_filters, self.symbols)
        live = [s for s in self.symbols if s in self.ex.filters]
        for s in self.symbols:
            if s not in live:
                log.warning("%s: нет на бирже (делистнута) — исключён", s)
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

    def _meta_mtime(self) -> float:
        try:
            return self.c.meta_model_path.stat().st_mtime
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
                if meta.get("gate"):
                    self.c.gate = float(meta["gate"])
                new_syms = [s for s in (self.c.symbols or meta["symbols"])
                            if s not in self.symbols]
                self.symbols = self.c.symbols or meta["symbols"]
                try:
                    cal = json.loads((self.c.model_path.parent
                                      / "calibration.json").read_text())
                    self._cal_x = np.array(cal["iso_x"], dtype="float64")
                    self._cal_y = np.array(cal["iso_y"], dtype="float64")
                    log.info("калибровка перечитана (%d точек)",
                             len(self._cal_x))
                except Exception as e:
                    log.warning("калибровка при перезагрузке: %s", str(e)[:60])
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
        # мета меняется ОТДЕЛЬНО от Бури (конвейер меты копирует только
        # meta.txt) — свой mtime-трекер, иначе она живёт лишь до рестарта
        # (помощник, 09.11)
        mt_meta = self._meta_mtime()
        if self.c.meta_gate and mt_meta \
                and mt_meta != getattr(self, "_meta_loaded_mtime", -1):
            try:
                import lightgbm as lgb
                self.meta = lgb.Booster(
                    model_file=str(self.c.meta_model_path))
                self._meta_loaded_mtime = mt_meta
                import bot.telegram as tg
                tg.fire("⚖️ <b>Мета</b>: новый замок подхвачен без рестарта")
                log.info("мета перезагружена (обновлена %s)",
                         datetime.fromtimestamp(mt_meta, timezone.utc)
                         .strftime("%H:%M UTC"))
            except Exception as e:
                log.warning("перезагрузка меты не удалась: %s", e)

    async def _refresh_day_xchg(self) -> None:
        """БИРЖА — ИСТОЧНИК ПРАВДЫ (владелец, 15.10): день = ВСЕ потоки
        с полуночи UTC (сделки + комиссии + фандинг). Кэш 120с; при сбое
        остаётся прошлое значение, локальный счёт — запасной."""
        if time.time() - getattr(self, "_day_xchg_ts", 0) < 120:
            return
        self._day_xchg_ts = time.time()
        if not hasattr(self, "_day_xchg"):
            self._day_xchg = None
        try:
            midnight = int(datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
            r = await asyncio.to_thread(
                self.ex.client.rest_api.get_income_history,
                start_time=midnight, limit=1000)
            inst = getattr(r.data(), "actual_instance", r.data())
            rows_l = inst if isinstance(inst, list) else []
            # день = торговля (сделки+комиссии+фандинг); переводы —
            # движение денег, не заработок, иначе пополнение картинка дня
            s = sum(float(getattr(x, "income", 0) or 0) for x in rows_l
                    if getattr(x, "income_type", "") != "TRANSFER")
            self._day_xchg = Decimal(str(round(s, 6)))
        except Exception as e:
            log.warning("день с биржи не получен (%s) — счёт локальный",
                        str(e)[:60])

    async def tick(self) -> None:
        now = datetime.now(timezone.utc)
        self.maybe_reload_model()
        if now.minute == 0 and now.hour != getattr(self, "_bal_hour", -1):
            self._bal_hour = now.hour   # депозит днём сразу влияет на сайзинг
            self.balance = await asyncio.to_thread(
                self.ex.account_wallet_balance, "USDT")
        if now.date() != self.day_key:
            self.day_key = now.date()
            self.day_pnl = Decimal(0)
            self.balance = await asyncio.to_thread(
                self.ex.account_wallet_balance, "USDT")
        if time.time() - getattr(self, "_last_recon", 0) >= 600:
            self._last_recon = time.time()
            await self.recon()
        await self._refresh_day_xchg()
        if self._day_xchg is not None:
            self.day_pnl = self._day_xchg
        # баланс == 0 (старт при бане/сбое) — перечитываем раз в 120с,
        # иначе дневной кап от нулевого баланса ложно душит торговлю
        if self.balance <= 0 and time.time() - getattr(
                self, "_last_bal_try", 0) >= 120:
            self._last_bal_try = time.time()
            try:
                self.balance = await asyncio.to_thread(
                    self.ex.account_wallet_balance, "USDT")
                log.info("баланс перечитан: %s USDT", self.balance)
            except Exception as e:
                log.warning("баланс не читается (%s)", str(e)[:60])
        day_cap = (min(self.c.day_cap_usdc, self.balance * self.cfg.daily_cap_pct)
                   if self.c.day_cap_usdc > 0
                   else self.balance * self.cfg.daily_cap_pct)
        if self.day_pnl <= -day_cap:
            # кап: отменяем ещё висящие входные заявки (иначе лазейка —
            # заявка, поставленная до капа, исполняется после него)
            for psym, pend in list(self.pending.items()):
                await asyncio.to_thread(self.ex.cancel_order, psym,
                                        pend["oid"])
                self.pending.pop(psym, None)
                log.info("%s: входная заявка отменена (дневной кап)", psym)
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
        # 1) оценка ВСЕХ доступных символов бара (только закрытые бары в df)
        scored = []
        atrs = []          # atr_pct всей вселенной этого бара — для фильтра
        for sym in self.symbols:
            if sym not in self.bars or sym in self.pos or sym in self.pending:
                continue
            try:
                x = await asyncio.to_thread(feature_row, self.bars[sym],
                                            self.btc_close, sym, self.feats)
                if not x:
                    continue
                p = float(self.model.predict(
                    np.array([[x[f] for f in self.feats]]))[0])
            except Exception as e:
                log.warning("%s: ошибка оценки (%s) — символ исключён",
                            sym, str(e)[:60])
                continue
            atr = x.get("atr_pct")
            if atr is not None and np.isfinite(atr):
                atrs.append(float(atr))
            if p > self.c.gate or p < 1 - self.c.gate:
                if self.meta is not None and atr is not None \
                        and np.isfinite(atr):
                    # мета обучена на сигналах с конечной atr — на свежих
                    # листингах без истории воздерживается (режет их и так
                    # фильтр «тихая треть»)
                    vec = [x.get(f) for f in self.feats] + [p, atr]
                    vec = [v if v is not None and np.isfinite(v) else 0.0
                           for v in vec]
                    try:
                        pw = float(self.meta.predict(np.array([vec]))[0])
                    except Exception:
                        pw = 1.0
                    if pw < 0.5:
                        self.meta_veto += 1
                        log.info("мета: вето %s p=%.3f p_win=%.2f",
                                 sym, p, pw)
                        continue
                    self.meta_pass += 1
                # калибровка: истинный шанс роста рядом с сырой уверенностью
                scored.append([abs(p - 0.5), p, sym, atr,
                               self.true_prob(p)])
        # 1.5) фильтр «тихая треть» (рентген 12.10: сделки в нижней
        #      терцили ATR вселенной −3.2бп на 21.8к сделок, летучая
        #      треть +85.7). Порог = живая 1/3-квантиль atr_pct вселенной
        #      этого бара — каузально и без констант из прошлого.
        if scored and self.c.quiet_filter and len(atrs) >= 30:
            q33 = float(np.quantile(atrs, 1 / 3))
            before = len(scored)
            scored = [s for s in scored
                      if s[3] is not None and np.isfinite(s[3])
                      and s[3] >= q33]
            if len(scored) < before:
                log.info("тихая-треть: отсечено %d из %d кандидатов "
                         "(порог atr %.5f)", before - len(scored), before,
                         q33)
        # 2) сортировка по уверенности (расстояние от 50%): самые уверенные
        #    входят первыми. Раньше при нехватке мест входили первые ПО
        #    АЛФАВИТУ — уверенность игнорировалась (дыра найдена владельцем).
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored:
            log.info("бар: кандидатов %d, лучшая уверенность %.3f (%s)",
                     len(scored), scored[0][1], scored[0][2])
        if self.meta is not None:
            log.info("мета: счёт принято/вето %d/%d", self.meta_pass,
                     self.meta_veto)
        slots = len(self.pos) + len(self.pending)
        # СТОЛ САМ ПОДСТРАИВАЕТСЯ ПОД ДЕНЬГИ (владелец 02.11): слотов не
        # больше, чем свободной маржи хватает по марже одной позиции.
        # При полном столе / просадке — сам сжимается, при пополнении —
        # сам раздвигается до max_slots. Отказов биржи по марже нет.
        avail = await asyncio.to_thread(self.ex.available_balance, "USDT")
        margin_one = self.notional(scored[0][2]) \
            / Decimal(self.c.leverage) if scored else Decimal(0)
        slot_cap = self.c.max_slots
        if margin_one > 0 and avail < margin_one:
            slot_cap = min(self.c.max_slots,
                           int(avail // margin_one))
            if slot_cap < slots:
                slot_cap = slots     # уже открытое не считаем лишним
        for conf, p, sym, _atr, t_true in scored:
            if slots >= slot_cap:
                log.info("очередь: %s p=%.3f — слотов нет (%d/%d, своб. маржа %s) — "
                         "ждёт следующего бара", sym, p, slots, slot_cap, avail)
                continue
            try:
                if p > self.c.gate:
                    if t_true < self.c.gate_true:
                        log.info("%s: честный шанс %.0f%% < %.0f%% — пропуск",
                                 sym, t_true * 100, self.c.gate_true * 100)
                        continue
                    await self.open_long(sym, p)
                    slots += 1
                else:
                    if 1 - t_true < self.c.gate_true:
                        log.info("%s: честный шанс падения %.0f%% < %.0f%% — пропуск",
                                 sym, (1 - t_true) * 100,
                                 self.c.gate_true * 100)
                        continue
                    await self.open_short(sym, p)
                    slots += 1
            except BinanceError as e:
                log.warning("%s: биржа отказала (%s) — сигнал пропущен",
                            sym, str(e)[:80])
                continue

    # ---------- сделки ----------
    def notional(self, sym: str) -> Decimal:
        """Сайзинг владельца (v3, 02.11): номинал = 10% от ПЛЕЧЕВОЙ
        ЁМКОСТИ половины баланса: баланс × 50% × плечо 3 × 10% (= 15%
        баланса), но никогда не ниже мин-ордера ЭТОЙ монеты × 1.1.
        Полный стол 10 позиций = ровно 50% баланса маржей."""
        base = self.balance * self.c.margin_pool_pct \
            * Decimal(self.c.leverage) * self.c.notional_pct
        fl = self.ex.filters.get(sym)
        if fl is not None:
            base = max(base, fl.min_notional * self.c.min_notional_buffer)
        return base

    def qty_for(self, sym: str, px: Decimal) -> Decimal:
        """qty_for_notional принимает НОТИОНАЛ в USDT и сам делит на цену
        (баг 28.09: двойное деление давало заявки в 4+ раза больше расчёта)."""
        fl = self.ex.filters.get(sym)
        if not fl or px <= 0:
            return Decimal(0)
        return fl.qty_for_notional(self.notional(sym), px) or Decimal(0)

    def true_prob(self, p: float) -> float:
        """Настоящий шанс роста по карте калибровки (np.interp по
        изотоническим порогам). Без карты — сырая уверенность как есть."""
        if self._cal_x is None:
            return p
        return float(np.interp(p, self._cal_x, self._cal_y))

    async def place_stop(self, sym: str, pos: dict) -> None:
        """БИРЖЕВОЙ СТОП (владелец 02.11): защита позиции живёт на бирже,
        а не в нашем polling. STOP_MARKET closePosition по порогу -15%
        от входа (тот же, что у прежней ботовой полиции); сработает
        миллисекундно и переживёт падение бота. Цена триггера — к тику
        монеты, в сторону срабатывания (чуть раньше — консервативно)."""
        try:
            # после рестарта наш стоп мог выжить на бирже — усыновляем его
            existing = await asyncio.to_thread(self.ex.find_own_stop, sym)
            if existing is not None:
                pos["sl_oid"] = existing
                log.info("%s: усыновляю существующий биржевой стоп (id=%s)",
                         sym, existing)
                return
            px = Decimal(str(pos["px"]))
            short = pos.get("side") == "SHORT"
            stop_px = px * (1 + self.c.emergency_stop_pct if short
                            else 1 - self.c.emergency_stop_pct)
            fl = self.ex.filters.get(sym)
            if fl is not None:
                stop_px = fl.round_price(stop_px)
            oid = await asyncio.to_thread(
                self.ex.place_stop_market, sym,
                "BUY" if short else "SELL", stop_px,
                f"ml5s-{sym}-{int(time.time())}")
            if oid is not None:
                pos["sl_oid"] = oid
                pos["sl_px"] = float(stop_px)
                log.info("%s: биржевой стоп @%s (id=%s)", sym, stop_px, oid)
            else:
                log.warning("%s: биржевой стоп НЕ выставлен — "
                            "попробую на следующем цикле", sym)
        except Exception as e:
            from bot.executor import api_code
            if api_code(e) == -4130:
                # "closePosition-стоп в этом направлении уже стоит" — наш
                # или чужой: усыновить, дальше просто следить
                existing = await asyncio.to_thread(self.ex.find_own_stop, sym)
                if existing is not None:
                    pos["sl_oid"] = existing
                    log.info("%s: -4130 -> усыновляю стоп id=%s",
                             sym, existing)
                else:
                    log.warning("%s: -4130 чужой closePosition-стоп — "
                                "мой не поставлен, повторю позже", sym)
            else:
                log.exception("%s: ошибка установки биржевого стопа", sym)

    async def slot_cap(self) -> int:
        """Текущий потолок стола: max_slots, урезанный свободной маржей
        (та же логика, что в on_bar — для отчётов)."""
        avail = await asyncio.to_thread(self.ex.available_balance, "USDT")
        try:
            margin_one = self.notional(next(iter(self.bars))) \
                / Decimal(self.c.leverage)
        except Exception:
            return self.c.max_slots
        if margin_one <= 0 or avail >= margin_one * self.c.max_slots:
            return self.c.max_slots
        cap = int(avail // margin_one)
        return max(cap, len(self.pos) + len(self.pending), 0)

    async def open_long(self, sym: str, p: float) -> None:
        top = await asyncio.to_thread(self.ex.order_book_top, sym)
        if not top:
            return
        bb, _ba = top
        if bb <= 0:
            return
        df = self.bars[sym]
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
            # цена ушла сквозь уровень: экзамен считал мейкерский вход —
            # пропускаем сигнал (погоня тейкером = вход хуже обещанного)
            log.info("%s: цена ушла от уровня — сигнал пропущен", sym)
            return
        self.pending[sym] = {"oid": oid, "qty": qty, "px": bb, "p": p,
                             "ts": time.time(), "side": "LONG"}
        log.info("%s: GTX-вход выставлен @%s qty=%s p=%.3f", sym, bb, qty, p)

    async def open_short(self, sym: str, p: float) -> None:
        """Зеркало open_long: SELL post-only по аску; цена ушла сквозь
        уровень — сигнал пропускается (как у лонга, без тейкер-погони)."""
        top = await asyncio.to_thread(self.ex.order_book_top, sym)
        if not top:
            return
        _bb, ba = top
        if ba <= 0:
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
            # зеркало лонга: экзамен считал вход по цене лимитки —
            # погоня тейкером = вход хуже обещанного, сигнал пропускаем
            log.info("%s: цена ушла от уровня — сигнал пропущен", sym)
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
                                      pos["qty"], reduce_only=True)
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
                    # позиции на бирже уже нет (вероятно, биржевым стопом)
                    log.warning("%s: -2022 при выходе — позиция уже закрыта",
                                sym)
                    since_ms = int(max(pos.get("entry_ts", 0),
                                       time.time() - 6 * 3600) * 1000)
                    pnl_ex = await asyncio.to_thread(
                        self.ex.realized_pnl_since, sym, since_ms)
                    await self.finish(sym, pos, 0.0, "external",
                                      exchange_pnl=pnl_ex)
                    return
                raise
            if oid is not None:
                pos["exit_oid"] = oid
                pos["exit_deadline"] = time.time() + self.c.exit_ttl_s
                return
        oid = await asyncio.to_thread(self.ex.place_market, sym, "SELL",
                                      pos["qty"], reduce_only=True)
        info = await asyncio.to_thread(self.ex.query_order_full, sym,
                                       oid) if oid else None
        exit_px = float((info or {}).get("avgPrice") or 0)
        await self.finish(sym, pos, exit_px, "taker")

    async def finish(self, sym: str, pos: dict, exit_px: float,
                     exit_fee: str, exchange_pnl=None) -> None:
        """Биржа — источник правды: нулевая цена выхода запрещена. Если
        цена не восстановима (закрытие вне бота), в журнал и дневной счёт
        идёт реализованный PnL с биржи."""
        if pos.get("sl_oid"):
            # обычный выход: биржевой стоп больше не нужен (иначе останется
            # сиротой и перевернёт будущую позицию по этой монете)
            await asyncio.to_thread(self.ex.cancel_algo_order,
                                    sym, pos["sl_oid"])
            pos.pop("sl_oid", None)
        if exit_px <= 0 and exchange_pnl is not None:
            pnl = Decimal(str(exchange_pnl))
            exit_fee = "external_birzha"
        else:
            if exit_px <= 0:
                top = await asyncio.to_thread(self.ex.order_book_top, sym)
                exit_px = (top[0] + top[1]) / 2 if top else pos["px"]
                log.warning("%s: цена выхода не от биржи — мид %s", sym,
                            exit_px)
            q = Decimal(str(pos["qty"]))
            mult = Decimal(1) if pos.get("side", "LONG") == "LONG" \
                else Decimal(-1)
            pnl = (Decimal(str(exit_px)) - Decimal(str(pos["px"]))) * q * mult
        await self._refresh_day_xchg()
        if self._day_xchg is not None:
            self.day_pnl = self._day_xchg
        else:
            self.day_pnl += pnl
        try:
            self.tconn.execute(
                "INSERT INTO trades (ts,symbol,side,entry_px,exit_px,qty,"
                "pnl,confidence,exit_kind) VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), sym, pos.get("side", "LONG"), float(pos["px"]),
                 float(exit_px), float(pos["qty"]), float(pnl),
                 float(pos.get("p", 0) or 0), exit_fee))
            self.tconn.commit()
        except Exception:
            log.exception("журнал сделок: запись не удалась")
        # СИГНАЛ ПЕРЕОБУЧЕНИЯ (SYSTEM_RULES): последние 30 живых сделок
        # суммарно в минусе -> флаг переобучения (подхватит конвейер Кегла)
        try:
            n = self.tconn.execute("select count(*) from trades").fetchone()[0]
            if n >= 30:
                avg = self.tconn.execute(
                    "select avg(pnl) from (select pnl from trades "
                    "order by id desc limit 30)").fetchone()[0]
                if avg is not None and avg < 0:
                    flag = self.cfg.data_dir / "signals" / "retrain_needed"
                    flag.parent.mkdir(parents=True, exist_ok=True)
                    if not flag.exists():
                        flag.write_text(datetime.now(timezone.utc).isoformat())
                        tg.fire(f"🧠 Сигнал: последние 30 сделок в минусе "
                                f"(средняя {avg:+.2f} USDT) — запрошено "
                                f"переобучение")
        except Exception:
            log.exception("сигнал переобучения")
        # БИРЖЕВЫЕ ЧИСЛА СДЕЛКИ (владелец 31.10: в ТГ — как в приложении
        # биржи). income-записи закрытия доезжают с задержкой — коротко
        # ждём; не доехало — показываем расчёт по ценам, как раньше.
        entry_ms = int(float(pos.get("entry_ts") or 0) * 1000) - 2000
        net = None
        day_add = None
        if entry_ms > 0:
            midnight = int(datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0,
                microsecond=0).timestamp() * 1000)
            for _ in range(4):
                rows = await asyncio.to_thread(self.ex.income_rows,
                                               sym, entry_ms)
                if not any(r["type"] == "REALIZED_PNL" for r in rows):
                    await asyncio.sleep(1.5)
                    continue
                await asyncio.sleep(1.0)   # комиссия выхода доезжает следом
                rows = await asyncio.to_thread(self.ex.income_rows,
                                               sym, entry_ms)
                net = sum((Decimal(r["income"]) for r in rows
                           if r["type"] in ("REALIZED_PNL", "COMMISSION")),
                          Decimal(0))
                # вклад сделки в СЕГОДНЯшний день (позиция могла быть
                # открыта вчера — её вчерашние потоки в день не добавляем)
                day_add = sum((Decimal(r["income"]) for r in rows
                               if r["type"] != "TRANSFER"
                               and r["time"] >= midnight), Decimal(0))
                break
        self.pos.pop(sym, None)
        tg.fire(f"{'✅' if pnl > 0 else '🔻'} <b>ML5 ВЫХОД {sym} "
                f"{'ШОРТ' if pos.get('side', 'LONG') == 'SHORT' else 'ЛОНГ'}</b> "
                f"{pnl:+.2f} USDT"
                + (f" (чистыми {net:+.2f})" if net is not None else ""))
        # день в момент закрытия: биржевой снимок брался ДО записи этой
        # сделки — добавляем её вклад сами (биржа запишет то же число)
        day_disp = self.day_pnl + (day_add if day_add is not None else pnl) \
            if self._day_xchg is not None else self.day_pnl
        log.info("ЗАКРЫТО %s @%s pnl=%.3f (день %.2f)", sym, exit_px, pnl,
                 day_disp)

    async def recon(self) -> None:
        """Биржа — источник правды: раз в 10 минут сверяем позиции.
        Своя (метка ml5- на входном ордере) — усыновляем с настоящим
        временем входа и доезжаем до будильника; чужая (ручная) —
        усыновляем и немедленно закрываем; наша, которой на бирже
        нет — фиксируем внешнее закрытие."""
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
            # чья позиция? наш вход помечен ml5- в clientOrderId биржи.
            # Своя -> усыновляем с НАСТОЯЩИМ временем входа и доезжаем
            # до будильника (кромка в недосиженном хвосте). Чужая
            # (ручная) -> закрываем немедленно, как раньше.
            own = await asyncio.to_thread(self.ex.own_entry_for,
                                          sym, float(amt))
            if own:
                mins = int((time.time() - own["ts"]) / 60)
                log.warning("%s: своя позиция после рестарта (вход %d мин "
                            "назад) — доезжаю до будильника", sym, mins)
                self.pos[sym] = {"qty": qty,
                                 "px": own["px"] or float(d.get("entryPrice", 0)),
                                 "entry_ts": own["ts"], "adopted": True,
                                 "side": "LONG" if amt > 0 else "SHORT"}
                await self.place_stop(sym, self.pos[sym])
                continue
            log.warning("%s: чужая позиция (%s) — усыновляю и закрываю",
                        sym, d.get("entryPrice"))
            self.pos[sym] = {"qty": qty, "px": float(d.get("entryPrice", 0)),
                             "entry_ts": 0, "adopted": True,
                             "side": "LONG" if amt > 0 else "SHORT"}
        for sym in list(self.pos):
            if sym not in live:
                pos = self.pos[sym]
                # различаем честно (владелец 03.11): стоп исполнился или
                # позицию закрыл кто-то другой (владелец вручную / его ТП)
                if pos.get("sl_oid"):
                    done = await asyncio.to_thread(
                        self.ex.stop_executed, pos["sl_oid"])
                    if done is False:
                        tg.fire(f"✋ <b>ML5 {sym}</b> позиция закрыта "
                                f"вручную — снимаю биржевой стоп")
                        await asyncio.to_thread(
                            self.ex.cancel_algo_order, sym, pos["sl_oid"])
                        pos.pop("sl_oid", None)
                    elif done is True:
                        tg.fire(f"🛑 <b>ML5 СТОП {sym}</b> сработал на бирже "
                                f"без бота @ {pos.get('sl_px')}")
                        pos.pop("sl_oid", None)
                    # done is None: биржа не ответила — стоп снимет finish
                log.warning("%s: закрыта вне бота — результат беру с биржи",
                            sym)
                info = (await asyncio.to_thread(self.ex.query_order_full,
                        sym, pos["exit_oid"])) if pos.get("exit_oid") \
                    else None
                if (info or {}).get("status") == "FILLED":
                    await self.finish(sym, pos,
                                      float((info or {}).get("avgPrice")
                                            or 0), "maker")
                else:
                    since_ms = int(max(pos.get("entry_ts", 0),
                                       time.time() - 6 * 3600) * 1000)
                    pnl_ex = await asyncio.to_thread(
                        self.ex.realized_pnl_since, sym, since_ms)
                    await self.finish(sym, pos, 0.0, "external",
                                      exchange_pnl=pnl_ex)

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
                    await self.place_stop(sym, self.pos[sym])
                    tg.fire(f"🟢 <b>ML5 ВХОД {sym} "
                            f"{'ШОРТ' if pend.get('side') == 'SHORT' else 'ЛОНГ'}</b> "
                            f"{pend['qty']} @ {px} (мейкер) p={pend['p']:.2f}")
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
                    tg.fire(f"🟡 <b>ML5 {sym} "
                            f"{'ШОРТ' if pend.get('side') == 'SHORT' else 'ЛОНГ'}</b> "
                            f"частичный вход {exq} @ {px}")
                    await self.close_any(sym, self.pos[sym], maker=True)
                elif self.c.taker_fallback:
                    # демо-режим: на демо-бирже нет ликвидности под
                    # мейкер-заявки — берём по рынку, чтобы экзамен
                    # продолжал собирать статистику сделок
                    side = "SELL" if pend.get("side") == "SHORT" else "BUY"
                    oid = await asyncio.to_thread(
                        self.ex.place_market, sym, side,
                        Decimal(str(pend["qty"])))
                    if oid is not None:
                        info = await asyncio.to_thread(
                            self.ex.query_order_full, sym, oid)
                        px = float((info or {}).get("avgPrice")
                                   or pend["px"])
                        self.pos[sym] = {**pend, "px": px,
                                         "entry_ts": time.time(),
                                         "maker_entry": False}
                        self.pending.pop(sym, None)
                        tg.fire(f"🟢 <b>ML5 ВХОД {sym} "
                                f"{'ШОРТ' if pend.get('side') == 'SHORT' else 'ЛОНГ'}</b> "
                                f"{pend['qty']} @ {px} (рынок)")
                        log.info("%s: TTL истёк — вход по рынку (фолбэк)",
                                 sym)
                else:
                    # цена не дошла до уровня за отведённое время: экзамен
                    # считал мейкерский вход по уровню — погоню тейкером
                    # не делаем, сигнал пропускаем
                    log.info("%s: вход не исполнен за TTL — сигнал пропущен",
                             sym)
            info = await asyncio.to_thread(self.ex.query_order_full, sym,
                                           pend["oid"])
            if (info or {}).get("status") == "FILLED":
                px = float((info or {}).get("avgPrice") or pend["px"])
                self.pending.pop(sym, None)
                self.pos[sym] = {**pend, "px": px, "entry_ts": time.time(),
                                 "maker_entry": True,
                                 "side": pend.get("side", "LONG")}
                await self.place_stop(sym, self.pos[sym])
                tg.fire(f"🟢 <b>ML5 ВХОД {sym} "
                        f"{'ШОРТ' if pend.get('side') == 'SHORT' else 'ЛОНГ'}</b> "
                        f"{pend['qty']} @ {px} (мейкер) p={pend['p']:.2f}")
        for sym, pos in list(self.pos.items()):
            # АВАРИЙНАЯ ЗАЩИТА = БИРЖЕВОЙ СТОП (предрегистрация 28.09 по
            # MAE-исследованию: глубже -15% ныряют 0.27% сделок; владелец
            # 02.11: переносим её НА БИРЖУ — STOP_MARKET ставится при
            # входе/усиновлении и срабатывает без нашего сервера).
            # Здесь только гарантия наличия: не выставился при входе —
            # пробуем на каждом цикле, пока не стоит.
            if "sl_oid" not in pos:
                await self.place_stop(sym, pos)
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

