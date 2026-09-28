"""Кегл-кернел: качает 30м панель ВСЕХ USDT-перпетуалов (2 года) с биржи
и сохраняет в output. Свежие IP Kaggle + честный темп (~1700 weight/min
при лимите 2400) — без 429-пауз."""
import asyncio, time, io, os
import aiohttp

OUT = "/kaggle/working"
PACE_S = 0.35
MAX_INFLIGHT = 16
YEARS_BACK = 2
BASE = "https://fapi.binance.com"

_next = [0.0]
_lock = asyncio.Lock()

async def paced_get(session, url, params):
    async with _lock:
        now = time.monotonic()
        wait = max(0.0, _next[0] - now)
        _next[0] = max(now, _next[0]) + PACE_S
    if wait > 0:
        await asyncio.sleep(wait)
    for _ in range(6):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=25)) as r:
                if r.status in (418, 429):
                    await asyncio.sleep(60 if r.status == 418 else 20)
                    continue
                r.raise_for_status()
                return await r.json()
        except Exception:
            await asyncio.sleep(3)
    return None

async def get_symbol(session, sem, sym, start_ms, end_ms, i, total):
    path = f"{OUT}/{sym}_30m.csv"
    if os.path.exists(path):
        return
    async with sem:
        rows = []
        cur = start_ms
        while cur < end_ms:
            batch = await paced_get(session, f"{BASE}/fapi/v1/klines",
                {"symbol": sym, "interval": "30m", "startTime": cur, "limit": 1500})
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
        buf = io.StringIO()
        buf.write("open_time,open,high,low,close,volume,close_time,quote_volume,n,taker_buy_base,taker_buy_quote,ignore\n")
        for r in rows:
            buf.write(",".join(str(x) for x in r) + "\n")
        with open(path, "w") as f:
            f.write(buf.getvalue())
        print(f"{i}/{total} {sym}: {len(rows)} баров", flush=True)

async def main():
    async with aiohttp.ClientSession() as s:
        info = await paced_get(s, f"{BASE}/fapi/v1/exchangeInfo", None)
        syms = sorted(x["symbol"] for x in (info or {}).get("symbols", [])
                      if x.get("contractType") == "PERPETUAL"
                      and x.get("status") == "TRADING"
                      and x.get("symbol", "").endswith("USDT"))
        total = len(syms)
        print(f"перпетуалов: {total}", flush=True)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - YEARS_BACK * 365 * 86400_000
        sem = asyncio.Semaphore(MAX_INFLIGHT)
        t0 = time.monotonic()
        await asyncio.gather(*[get_symbol(s, sem, sym, start_ms, end_ms, i, total)
                               for i, sym in enumerate(syms, 1)])
        print(f"ГОТОВО за {time.monotonic()-t0:.0f}с", flush=True)

asyncio.run(main())
