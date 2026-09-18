"""KPI пилота из журнала + income history (вызывается ежечасным мониторингом).
Запуск: PYTHONPATH=.:pylibs python3 bot/kpi.py
"""
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.config import BotConfig                    # noqa: E402
from bot.executor import Executor, unwrap           # noqa: E402


def main() -> None:
    db = sqlite3.connect(Path(__file__).resolve().parent.parent / "bot" / "journal.db")

    # 1) фактическая комиссия по ролям (бп нотионала)
    print("KPI тариф (по исполнениям журнала):")
    for role, name in [("E", "вход (мейкер)"), ("T", "тейк (мейкер)"), ("S", "стоп (тейкер)")]:
        rows = db.execute(
            "SELECT price, qty, commission FROM fills WHERE role=? AND CAST(price AS REAL)>0",
            (role,)).fetchall()
        notional = sum(float(p) * float(q) for p, q, _ in rows)
        fee = sum(float(c or 0) for _, _, c in rows)
        if notional:
            print(f"  {name:<16} {len(rows):>4} исп., ставка {fee/notional*100:.4f}%")

    # 2) протрузии (глубина прохода цены сквозь уровень) — по средам раздельно
    tp = {"demo": [], "mainnet": []}
    en = {"demo": [], "mainnet": []}
    for sym, payload in db.execute("SELECT symbol, payload FROM events WHERE kind='fill_protrusion'"):
        d = json.loads(payload)
        env = d.get("env", "demo")
        (tp if d["kind"] == "tp" else en).setdefault(env, []).append(d["depth_bp"])
    for env in ("demo", "mainnet"):
        for name, arr in [("тейков", tp[env]), ("входов", en[env])]:
            tag = f"[{env}] KPI протрузия {name}"
            if arr:
                arr.sort()
                print(f"{tag}: n={len(arr)}, медиана {arr[len(arr)//2]:.1f} бп, "
                      f"90-й перц {arr[int(len(arr)*0.9)]:.1f} бп")
            else:
                print(f"{tag}: замеров пока нет")
    n_fills = db.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    print(f"(источник: журнал, накопительно с 17.09 через рестарты; {n_fills} исполнений)")

    # 3) слиппедж стопов: триггер (событие) -> цена исполнения (fill role S)
    slips = []
    last_trigger = {}
    for ts, kind, sym, payload in db.execute(
            "SELECT ts, kind, symbol, payload FROM events "
            "WHERE kind IN ('stop_placed','fill_stop') ORDER BY ts"):
        if kind == "stop_placed":
            try:
                last_trigger[sym] = float(json.loads(payload)["price"])
            except Exception:
                pass
        else:
            trig = last_trigger.get(sym)
            if trig:
                try:
                    fill = float(json.loads(payload).get("price", 0) or 0)
                    if fill:
                        slips.append((fill/trig - 1) * 10000)
                except Exception:
                    pass
    if slips:
        slips.sort()
        print(f"KPI слиппедж стопов: n={len(slips)}, медиана {slips[len(slips)//2]:+.1f} бп, "
              f"худший {slips[-1]:+.1f} бп")
    else:
        print("KPI слиппедж стопов: замеров пока нет")

    # 4) авторитетный тариф с биржи (income history, 24ч, с пагинацией)
    try:
        cfg = BotConfig.from_env(Path(__file__).resolve().parent.parent)
        ex = Executor(cfg, {})
        start = int(time.time() * 1000) - 24 * 3600 * 1000
        tot, n, complete = 0.0, 0, True
        cursor = start
        while True:
            comm = unwrap(ex.client.rest_api.get_income_history(
                income_type="COMMISSION", start_time=cursor, limit=1000).data())
            rows = getattr(comm, "root", None) or comm
            recs = [r.model_dump(by_alias=True) for r in rows]
            n += len(recs)
            for d in recs:
                tot += float(d.get("income", 0) or 0)
            if len(recs) < 1000:
                break
            cursor = max(int(d.get("time", 0)) for d in recs) + 1   # страница полна — дальше
            complete = False
            if time.time() * 1000 - cursor < 60_000:
                break
        tag = "полные данные" if complete else "ВНИМАНИЕ: выгрузка упёрлась в лимит, сумма неполная"
        print(f"KPI биржа-24ч: комиссий {tot:+.2f} USDC за {n} записей ({tag}) — "
              f"{'промо не действует' if tot else 'комиссий нет'}")
    except Exception as e:
        print(f"KPI биржа-24ч: недоступно ({e})")


if __name__ == "__main__":
    main()
