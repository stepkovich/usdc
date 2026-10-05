"""Рекордер стакана v1.2: штатный WebSocket-клиент SDK (по примерам
репозитория binance-connector-python, derivatives_trading_usds_futures).

ДАННЫЕ: мейннет (публичные рыночные стримы — бесплатно). ИСПОЛНЕНИЕ:
всегда демо (бот рядом, demo-fapi).

Стримы (примеры SDK: partial_book_depth_streams, aggregate_trade_streams):
  <sym>@depth20@500ms — снапшот 20 уровней 2 раза/сек (частота биржи);
  <sym>@aggTrade      — каждая сделка в реальном времени.

SDK сам держит соединение, пингует и переподключается (POOL, 10 попыток).
Запись строк (каждые 500 мс — частота пачек биржи) идёт в ОТДЕЛЬНОЙ
задаче, ничто её не блокирует.

Хранение: SQLite (горячий хвост 6 ч — читает бот) + часовые Parquet
(zstd, полный архив; на сервере живёт N_DAYS_KEEP дней, старое
вытесняется; локальная машина забирает архив себе ночью).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from binance_common.configuration import ConfigurationWebSocketStreams
from binance_common.constants import (
    DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
    WebsocketMode,
)
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)
from binance_sdk_derivatives_trading_usds_futures.websocket_streams.models import (
    PartialBookDepthStreamsLevelsEnum,
)

log = logging.getLogger("lob.recorder")
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "lob" / "lob.db"
PARQUET_DIR = ROOT / "data" / "lob" / "archive"
N_DAYS_KEEP = 5
HOT_HOURS = 6
ROW_MS = 500
UNIVERSE: list[str] = []          # задаётся из конфига (lob.engine)

SCHEMA = """
CREATE TABLE IF NOT EXISTS feat (
    ts INTEGER NOT NULL, symbol TEXT NOT NULL,
    mid REAL, spread_bp REAL, microprice REAL,
    imb5 REAL, imb10 REAL, imb20 REAL,
    bid_sum20 REAL, ask_sum20 REAL,
    flow10_buy REAL, flow10_sell REAL, flow60_buy REAL, flow60_sell REAL,
    ntr10 INTEGER, vpin10 REAL,
    imb1 REAL, slope_b REAL, slope_a REAL, wall_b REAL, wall_a REAL,
    PRIMARY KEY (ts, symbol));
"""
FEATURE_ROW = ("ts,symbol,mid,spread_bp,microprice,imb5,imb10,imb20,"
               "bid_sum20,ask_sum20,flow10_buy,flow10_sell,flow60_buy,"
               "flow60_sell,ntr10,vpin10,"
               "imb1,slope_b,slope_a,wall_b,wall_a")
NCOLS = 21
# Признаки лестницы (v2, 01.10): imb1 — перекос лучшего уровня;
# slope_b/a — где объём: глубина (уровни 6-20) против верха (1-5);
# wall_b/a — крупнейший уровень против среднего по 20 (стена).
RAW_DEPTH_ROW = (["bid_p_%d" % i for i in range(1, 21)]
                 + ["bid_q_%d" % i for i in range(1, 21)]
                 + ["ask_p_%d" % i for i in range(1, 21)]
                 + ["ask_q_%d" % i for i in range(1, 21)])
PARQUET_ROW = (FEATURE_ROW + "," + ",".join(RAW_DEPTH_ROW)).split(",")
SQL_N = len(FEATURE_ROW.split(","))          # первые 21 колонка буфера


def _ladder_feats(bids, asks, bq, aq):
    """Все новые признаки — безразмерные (отношения объёмов)."""
    tot1 = bq + aq
    imb1 = (bq - aq) / tot1 if tot1 > 0 else 0.0
    btop = sum(q for _, q in bids[:5])
    atop = sum(q for _, q in asks[:5])
    bdeep = sum(q for _, q in bids[5:20])
    adeep = sum(q for _, q in asks[5:20])
    slope_b = bdeep / btop if btop > 0 else 0.0
    slope_a = adeep / atop if atop > 0 else 0.0
    bq20 = [q for _, q in bids[:20]]
    aq20 = [q for _, q in asks[:20]]
    wall_b = (max(bq20) / (sum(bq20) / 20)) if bq20 and sum(bq20) > 0 else 0.0
    wall_a = (max(aq20) / (sum(aq20) / 20)) if aq20 and sum(aq20) > 0 else 0.0
    return imb1, slope_b, slope_a, wall_b, wall_a


class SymState:
    __slots__ = ("book", "trades", "msg_n")

    def __init__(self):
        self.book = None
        self.trades = deque()
        self.msg_n = 0


STATES = {s: SymState() for s in UNIVERSE}


def _to_dict(model):
    d = model.model_dump(by_alias=True) if hasattr(model, "model_dump") else model
    if isinstance(d, dict) and d.get("actual_instance") is not None:
        d = d["actual_instance"]
        if hasattr(d, "model_dump"):
            d = d.model_dump(by_alias=True)
    return d


def make_depth_handler(sym: str):
    def handler(model) -> None:
        try:
            st = STATES[sym]
            st.msg_n += 1
            d = _to_dict(model)
            bids = [(float(p), float(q))
                    for p, q in (d.get("b") or d.get("bids") or [])]
            asks = [(float(p), float(q))
                    for p, q in (d.get("a") or d.get("asks") or [])]
            if bids and asks:
                st.book = (bids, asks)
        except Exception:                      # noqa: BLE001
            pass
    return handler


def make_trade_handler(sym: str):
    def handler(model) -> None:
        try:
            st = STATES[sym]
            st.msg_n += 1
            d = _to_dict(model)
            ts = int(d.get("T") or d.get("time") or d.get("trade_time") or 0)
            q = float(d.get("q") or d.get("quantity") or 0)
            m = d.get("m", d.get("is_buyer_maker"))
            if ts and q:
                st.trades.append((ts, q, not bool(m)))
        except Exception:                      # noqa: BLE001
            pass
    return handler


def compute_row(st: SymState, now_ms: int) -> tuple | None:
    if not st.book:
        return None
    bids, asks = st.book
    bb, bq = bids[0]
    ba, aq = asks[0]
    mid = (bb + ba) / 2
    if mid <= 0:
        return None
    spread_bp = (ba - bb) / mid * 10000
    micro = (ba * bq + bb * aq) / (bq + aq) if (bq + aq) > 0 else mid

    def imb(k):
        bs = sum(q for _, q in bids[:k])
        as_ = sum(q for _, q in asks[:k])
        tot = bs + as_
        return (bs - as_) / tot if tot > 0 else 0.0, bs, as_

    imb5, _, _ = imb(5)
    imb10, _, _ = imb(10)
    imb20, bs20, as20 = imb(min(20, len(bids), len(asks)))
    while st.trades and st.trades[0][0] < now_ms - 60_000:
        st.trades.popleft()
    f10b = f10s = f60b = f60s = 0.0
    ntr = 0
    for t_ts, q, buy in st.trades:
        if t_ts >= now_ms - 10_000:
            if buy:
                f10b += q
            else:
                f10s += q
            ntr += 1
        if buy:
            f60b += q
        else:
            f60s += q
    tot10 = f10b + f10s
    vpin = abs(f10b - f10s) / tot10 if tot10 > 0 else 0.0
    l1, sl_b, sl_a, w_b, w_a = _ladder_feats(bids, asks, bq, aq)
    base = (now_ms, mid, spread_bp, micro, imb5, imb10, imb20,
            bs20, as20, f10b, f10s, f60b, f60s, ntr, vpin,
            l1, sl_b, sl_a, w_b, w_a)
    # сырые 20 уровней в архив (пересчитать любой новый признак потом
    # можно без ожидания новых дней); в SQLite сырьё не пишем — база
    # маленькая, ей нужна только горячая скорость
    def pad(levels):
        p = [0.0] * 20
        q = [0.0] * 20
        for i, (pr, qt) in enumerate(levels[:20]):
            p[i], q[i] = pr, qt
        return p, q
    bp, bq20_ = pad(bids)
    ap, aq20_ = pad(asks)
    return base + tuple(bp + bq20_ + ap + aq20_)


class Archive:
    def __init__(self):
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        self.buf: dict[str, list] = {}

    def add(self, rows: list) -> None:
        for r in rows:
            key = datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc) \
                .strftime("%Y%m%d_%H")
            self.buf.setdefault(key, []).append(r)

    def flush(self) -> None:
        import pandas as pd
        for key, rows in list(self.buf.items()):
            if not rows:
                continue
            out = PARQUET_DIR / f"feat_{key}.parquet"
            new = pd.DataFrame(rows, columns=PARQUET_ROW)
            if out.exists():
                old = pd.read_parquet(out)
                # старые строки часа без сырых уровней останутся с NaN —
                # норма: сырьё пишется только с версии depth20
                new = pd.concat([old, new]).drop_duplicates(
                    ["ts", "symbol"]).sort_values("ts")
            # атомарная запись: битых полузаписанных файлов больше не будет
            tmp = out.with_suffix(".tmp")
            new.to_parquet(tmp, compression="zstd", index=False)
            os.replace(tmp, out)
            del self.buf[key]

    def prune(self) -> None:
        cutoff = time.time() - N_DAYS_KEEP * 86400
        for f in PARQUET_DIR.glob("feat_*.parquet"):
            try:
                day = datetime.strptime(f.stem[5:13], "%Y%m%d").timestamp()
                if day < cutoff:
                    f.unlink()
                    log.info("архив устарел, удалён: %s", f.name)
            except Exception:
                pass


async def writer_task(conn_sql: sqlite3.Connection, arch: Archive) -> None:
    buf: list = []
    last_row = 0
    last_sql = time.time()
    last_arch = time.time()
    last_pulse = 0
    last_vacuum_day = None
    while True:
        await asyncio.sleep(0.2)
        try:
            now = int(time.time() * 1000)
            if now - last_row >= ROW_MS:
                last_row = now
                for s, st in STATES.items():
                    row = compute_row(st, now)
                    if row:
                        buf.append((now, s) + row[1:])
            if time.time() - last_sql >= 5:
                last_sql = time.time()
                if buf:
                    # в горячую базу — только 21 признак (движок),
                    # сырьё уровней живёт только в архиве-паркете
                    conn_sql.executemany(
                        f"INSERT OR REPLACE INTO feat ({FEATURE_ROW}) "
                        f"VALUES ({','.join('?' * SQL_N)})",
                        [r[:SQL_N] for r in buf])
                    hot_cut = now - HOT_HOURS * 3600_000
                    conn_sql.execute("delete from feat where ts < ?",
                                     (hot_cut,))
                    conn_sql.commit()
            if time.time() - last_arch >= 60:
                last_arch = time.time()
                if buf:
                    arch.add(buf)
                    buf = []
                arch.flush()
                arch.prune()
            # сборка мусора ВНУТРИ писателя (та же связь — не спорит
            # сама с собой). Внешний VACUUM убил рекордер 30.09.
            now_dt = datetime.now(timezone.utc)
            if (now_dt.hour == 4 and now_dt.minute >= 5
                    and last_vacuum_day != now_dt.date()):
                last_vacuum_day = now_dt.date()
                try:
                    # сигналы старше 7 дней не нужны (журнал сделок —
                    # отдельная таблица, не трогаем)
                    cut = int((time.time() - 7 * 86400) * 1000)
                    n = conn_sql.execute(
                        "delete from signals where ts < ?", (cut,)).rowcount
                    conn_sql.execute("VACUUM")
                    conn_sql.commit()
                    log.info("уборка: удалено %d старых сигналов, VACUUM "
                             "выполнен", n)
                except Exception as e:
                    log.warning("VACUUM отложен: %s", e)
        except sqlite3.OperationalError as e:
            log.warning("SQL занят (%s) — повторю через цикл", str(e)[:60])
        except Exception:
            log.exception("writer_task")
        if now - last_pulse >= 60_000:
            last_pulse = now
            rate = sum(st.msg_n for st in STATES.values())
            for st in STATES.values():
                st.msg_n = 0
            log.info("пульс: %d сообщений/мин, строк в буфере %d",
                     rate, len(buf))


async def run(universe: list[str]) -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn_sql = sqlite3.connect(DB, timeout=60)
    conn_sql.executescript(SCHEMA)
    # миграция старой базы (16 колонок): добавляем 5 лестничных; если
    # уже есть — не трогаем
    for col in ("imb1", "slope_b", "slope_a", "wall_b", "wall_a"):
        try:
            conn_sql.execute(f"ALTER TABLE feat ADD COLUMN {col} REAL")
        except sqlite3.OperationalError as e:
            if "duplicate" not in str(e).lower():
                log.warning("миграция %s: %s", col, e)
    conn_sql.commit()
    conn_sql.execute("PRAGMA journal_mode=WAL")
    conn_sql.execute("PRAGMA busy_timeout=60000")
    global UNIVERSE
    UNIVERSE = list(universe)
    for s in UNIVERSE:
        STATES.setdefault(s, SymState())
    arch = Archive()

    client = DerivativesTradingUsdsFutures(
        config_ws_streams=ConfigurationWebSocketStreams(
            stream_url=DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
            mode=WebsocketMode.POOL, pool_size=4,
            reconnect_attempts=10, reconnect_delay=5000))
    conn = await client.websocket_streams.create_connection()
    for sym in UNIVERSE:
        ds = await conn.partial_book_depth_streams(
            symbol=sym.lower(),
            levels=PartialBookDepthStreamsLevelsEnum["LEVELS_20"].value)
        ds.on("message", make_depth_handler(sym))
        at = await conn.aggregate_trade_streams(symbol=sym.lower())
        at.on("message", make_trade_handler(sym))
    log.info("SDK-стримы подписаны: %d символов x (depth20@500ms + aggTrade)",
             len(UNIVERSE))
    await asyncio.create_task(writer_task(conn_sql, arch))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
