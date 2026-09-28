"""Загрузка 30м истории ВСЕХ живых USDT-перпетуалов (asyncio gather).

Урок бана: параллелизм БЕЗ темпа = бан. Здесь наоборот: до 8 задач в
полёте (скрывают сетевые задержки), но старт каждого запроса зажат
общим замком — не быстрее ~3.2 запроса/сек = ~1900 весовых единиц/мин
при лимите биржи 2400. Кэш: data_cache_30m/<SYM>_30m.csv, докачивается
с места остановки.
Запуск в фоне: nohup python3 ops/download_panel_30m.py &
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import aiohttp

OUT = Path("/home/iek/PycharmProjects/USDC/data_cache_30m")
PACE_S = 0.52
MAX_INFLIGHT = 16
YEARS_BACK = 2
BASE = "https://fapi.binance.com"

_next_start = [0.0]
_pace_lock = asyncio.Lock()


async def paced_get(session, url, params):
    """Запрос с глобальным темпом + ретраи на 429/418."""
    async with _pace_lock:
        now = time.monotonic()
        wait = max(0.0, _next_start[0] - now)
        _next_start[0] = max(now, _next_start[0]) + PACE_S
    if wait > 0:
        await asyncio.sleep(wait)
    for attempt in range(6):
        try:
            async with session.get(url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=25)) as r:
                if r.status in (418, 429):
                    pause = 90 if r.status == 418 else 30
                    print(f"лимит ({r.status}), пауза {pause}с", flush=True)
                    await asyncio.sleep(pause)
                    continue
                r.raise_for_status()
                return await r.json()
        except Exception as e:
            print("retry:", e, flush=True)
            await asyncio.sleep(3)
    return None


async def get_symbol(session, sem, sym, start_ms, end_ms, i, total) -> None:
    out = OUT / f"{sym}_30m.csv"
    if out.exists():
        return
    async with sem:
        rows = []
        cur = start_ms
        while cur < end_ms:
            batch = await paced_get(session, f"{BASE}/fapi/v1/klines", {
                "symbol": sym, "interval": "30m", "startTime": cur,
                "limit": 1500})
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1500:
                break
            cur = int(batch[-1][0]) + 1
            await asyncio.sleep(0)
        if not rows:
            print(f"{i}/{total} {sym}: пусто", flush=True)
            return
        lines = ["open_time,open,high,low,close,volume,close_time,"
                 "quote_volume,n,taker_buy_base,taker_buy_quote,ignore"]
        for r in rows:
            lines.append(",".join(str(x) for x in r))
        tmp = out.with_suffix(".tmp")
        tmp.write_text("\n".join(lines))
        tmp.rename(out)
        print(f"{i}/{total} {sym}: {len(rows)} баров", flush=True)


async def main_async() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        info = await paced_get(session, f"{BASE}/fapi/v1/exchangeInfo", None)
        syms = sorted(s["symbol"] for s in (info or {}).get("symbols", [])
                      if s.get("contractType") == "PERPETUAL"
                      and s.get("status") == "TRADING"
                      and s.get("symbol", "").endswith("USDT"))
        total = len(syms)
        print(f"перпетуалов USDT: {total}", flush=True)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - YEARS_BACK * 365 * 86400_000
        sem = asyncio.Semaphore(MAX_INFLIGHT)
        t0 = time.monotonic()
        tasks = [get_symbol(session, sem, sym, start_ms, end_ms, i, total)
                 for i, sym in enumerate(syms, 1)]
        await asyncio.gather(*tasks)
    print(f"ГОТОВО за {time.monotonic()-t0:.0f}с", flush=True)


if __name__ == "__main__":
    asyncio.run(main_async())
