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

    # 1) КАНОНИЧЕСКАЯ таблица комиссий: вход-мейкер / вход-тейкер / тейк / стоп
    import numpy as np
    print("KPI тариф — каноническая таблица (окно: накопительно с 17.09; источник: fills):")
    rows = db.execute(
        "SELECT role, symbol, price, qty, commission FROM fills WHERE CAST(price AS REAL)>0 "
        "AND CAST(commission AS REAL)>0").fetchall()
    recs = []
    for role, sym, p, q, c in rows:
        p, q, c = float(p), float(q), float(c)
        recs.append(dict(role=role, sym=sym, n=p*q, rate=c/(p*q)*10000))
    def block(name, rs):
        if not rs:
            print(f"  {name:<14} нет данных"); return
        rates = np.array([r["rate"] for r in rs])
        notional = sum(r["n"] for r in rs)
        fees = sum(r["rate"]/10000*r["n"] for r in rs)
        print(f"  {name:<14} {len(rs):>5} исп. | оборот {notional:>10,.0f} | "
              f"комиссии {fees:>7.2f} | ставка: медиана {np.median(rates):.2f} бп, "
              f"по объёму {fees/notional*10000:.2f} бп")
    e_recs = [r for r in recs if r["role"] == "E"]
    block("вход-мейкер", [r for r in e_recs if r["rate"] <= 3])
    block("вход-тейкер", [r for r in e_recs if r["rate"] > 4])
    block("вход-серые", [r for r in e_recs if 3 < r["rate"] <= 4])
    block("тейк-профит", [r for r in recs if r["role"] == "T"])
    block("стоп", [r for r in recs if r["role"] == "S"])

    # 2) доля тейкерских входов: по счёту, по ОБЪЁМУ, по квартилям волатильности
    e_all = [r for r in e_recs]
    if e_all:
        # фактическая комиссия входа на объём (без классификации) — основной KPI
        ef = sum(r["rate"]/10000*r["n"] for r in e_all)
        ev = sum(r["n"] for r in e_all)
        ebp = ef/ev*10000
        zone = ("зелёный" if ebp <= 0.5 else "жёлтый" if ebp <= 1.1 else "красный")
        print(f"KPI фактическая комиссия входа: {ebp:.2f} бп на объём "
              f"[mainnet-порог: <=0.5 зелёный, 0.5-1.1 жёлтый, >1.1 красный] "
              f"— сейчас {zone} (демо-калибровка)")
        tcount = len([r for r in e_all if r["rate"] > 4])
        print(f"  разрез: тейкерских по счёту {tcount}/{len(e_all)} "
              f"({tcount/len(e_all)*100:.0f}%), серых {len([r for r in e_all if 3 < r['rate'] <= 4])}")
    sig = db.execute("SELECT symbol, payload FROM events WHERE kind='signal_entry'").fetchall()
    sym_vol = {}
    for s, pl in sig:
        d = json.loads(pl)
        sym_vol.setdefault(s, []).append(float(d.get("atr0") or 0) * 12)
    sym_vol = {s: float(np.median(a)) for s, a in sym_vol.items()}
    q1, q2, q3 = np.percentile(list(sym_vol.values()), [25, 50, 75])
    buckets = {}
    for r in recs:
        if r["role"] != "E" or r["sym"] not in sym_vol:
            continue
        v = sym_vol[r["sym"]]
        qb = ("Q1 спокойные" if v <= q1 else "Q2" if v <= q2 else
              "Q3" if v <= q3 else "Q4 дикие")
        buckets.setdefault(qb, []).append(r)
    for qb in ["Q1 спокойные", "Q2", "Q3", "Q4 дикие"]:
        rs = buckets.get(qb, [])
        if not rs:
            continue
        tv = sum(r["n"] for r in rs if r["rate"] > 4)
        av = sum(r["n"] for r in rs)
        print(f"  {qb:<14} {len(rs):>4} входов, тейкеры в ОБЪЁМЕ {tv/av*100:4.0f}%")

    # 2) протрузии (глубина прохода цены сквозь уровень) — по средам раздельно
    # дедупликация: частичные заполнения одной заявки в одну минуту дают
    # один бар -> одинаковые замеры; считаем уникальные (символ, минута, глубина)
    tp = {"demo": [], "mainnet": []}
    en = {"demo": [], "mainnet": []}
    seen = set(); dup = 0
    for sym, ts, payload in db.execute("SELECT symbol, ts, payload FROM events WHERE kind='fill_protrusion'"):
        d = json.loads(payload)
        env = d.get("env", "demo")
        key = (sym, int(float(ts) // 60), round(d["depth_bp"], 1))
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        (tp if d["kind"] == "tp" else en).setdefault(env, []).append(d["depth_bp"])
    for env in ("demo", "mainnet"):
        for name, arr in [("тейков", tp[env]), ("входов", en[env])]:
            tag = f"[{env}] KPI протрузия {name}"
            if arr:
                arr.sort()
                print(f"{tag}: n={len(arr)} (дедуплицировано {dup} частичных дубликатов), "
                      f"медиана {arr[len(arr)//2]:.1f} бп, "
                      f"90-й перц {arr[int(len(arr)*0.9)]:.1f} бп")
            else:
                print(f"{tag}: замеров пока нет")
    n_fills = db.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    print(f"(источник: журнал, накопительно с 17.09 через рестарты; {n_fills} исполнений)")

    # 3) доля TP с первого касания (мост линейка <-> модель)
    tp_all = tp["demo"] + tp["mainnet"]
    if tp_all:
        ft = sum(1 for d in tp_all if d <= 1)
        print(f"KPI доля TP с первого касания: {ft}/{len(tp_all)} = {ft/len(tp_all)*100:.0f}% "
              f"(модель предполагает 100% при протрузии порога)")
    # 4) фандинг за 24ч
    try:
        cfg = BotConfig.from_env(Path(__file__).resolve().parent.parent)
        ex = Executor(cfg, {})
        start = int(time.time() * 1000) - 24 * 3600 * 1000
        fund = unwrap(ex.client.rest_api.get_income_history(
            income_type="FUNDING_FEE", start_time=start, limit=1000).data())
        frows = getattr(fund, "root", None) or fund
        ftot = sum(float(r.model_dump(by_alias=True).get("income", 0) or 0) for r in frows)
        print(f"KPI фандинг-24ч: {ftot:+.2f} USDC")
    except Exception as e:
        print(f"KPI фандинг-24ч: недоступно ({e})")

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
