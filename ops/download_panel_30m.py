"""Загрузка 30м истории ВСЕХ живых USDT-перпетуалов с мейннета.

ПАЦИЕНТСКОЕ скачивание (урок бана по IP за 12 параллельных соединений):
- один запрос за раз, пауза 0.35с между запросами;
- глубина: 2 года (walk-forward хватит);
- кэш: data_cache_30m/<SYM>_30m.csv, докачивается с места остановки.
Запуск в фоне: nohup python3 ops/download_panel_30m.py &
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

OUT = Path("/home/iek/PycharmProjects/USDC/data_cache_30m")
PAUSE_S = 0.35
YEARS_BACK = 2
BASE = "https://fapi.binance.com"


def get(url, params=None):
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 418 or r.status_code == 429:
                wait = 90 if r.status_code == 418 else 30
                print(f"лимит ({r.status_code}), пауза {wait}с", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print("retry:", e, flush=True)
            time.sleep(5)
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    info = get(f"{BASE}/fapi/v1/exchangeInfo") or []
    syms = sorted(s["symbol"] for s in info.get("symbols", [])
                  if s.get("contractType") == "PERPETUAL"
                  and s.get("status") == "TRADING"
                  and s.get("symbol", "").endswith("USDT"))
    print(f"перпетуалов USDT: {len(syms)}", flush=True)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - YEARS_BACK * 365 * 86400_000
    for i, sym in enumerate(syms):
        out = OUT / f"{sym}_30m.csv"
        if out.exists():
            continue
        rows = []
        cur = start_ms
        while cur < end_ms:
            batch = get(f"{BASE}/fapi/v1/klines", {
                "symbol": sym, "interval": "30m", "startTime": cur,
                "limit": 1500})
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1500:
                break
            cur = int(batch[-1][0]) + 1
            time.sleep(PAUSE_S)
        if not rows:
            print(f"{i+1}/{len(syms)} {sym}: пусто", flush=True)
            continue
        lines = ["open_time,open,high,low,close,volume,close_time,"
                 "quote_volume,n,taker_buy_base,taker_buy_quote,ignore"]
        for r in rows:
            lines.append(",".join(str(x) for x in r))
        out.write_text("\n".join(lines))
        print(f"{i+1}/{len(syms)} {sym}: {len(rows)} баров", flush=True)
        time.sleep(PAUSE_S)
    print("ГОТОВО", flush=True)


if __name__ == "__main__":
    main()
