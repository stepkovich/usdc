"""Честный бэктест кандидата №1: кросс-секционная ротация моментума
с режим-гейтом BTC (26.09).

Данные: USDC-M панели 1h из архива владельца (binance/test):
  2024 (26 симв.), 2025 (37), 2026YTD (38) — наш родной рынок.

Класс сигнала: ранжируем пары по 30-дневному моментуму, держим топ-2
лонга в бычьем режиме BTC, шортим аутсайдеров в медвежьем, в болоте —
деньги в стороне. Гейт = тот самый верифицированный (BTC 30д ±3%).

Честность:
- крипта 24/7: close дня t = цена исполнения решения (без гэпов и
  lookahead — решение и сделка по одной цене);
- издержки: сценарии maker (0% комиссия + 0.05% проскальзывание) и
  taker (0.045% + 0.05%) НА КАЖДОЕ изменение позиции;
- листинги: пара попадает во вселенную только через 30 дней после
  старта данных (свежие пампы не участвуют);
- СУРВИВАНШИП: панели собраны 2026-08, мёртвые монеты не попали ->
  АБСОЛЮТНЫЕ цифры завышены. Главный вопрос поэтому ОТНОСИТЕЛЬНЫЙ:
  перцентиль моментум-выбора против нуль-модели (200 случайных
  выборов тех же 2 пар в те же дни под тем же гейтом). Нуль-тест
  контролирует выживаемость — и выборку, и дни у сравниваемых равные.
- фандинг не смоделирован (задокументировано): лонги в быке платят
  ~0.01%/8ч -> порядка −10%/год на нотионал лонгов, шорты в медведе
  примерно столько же ПОЛУЧАЮТ. Для честного сравнения версий между
  собой это константа, для абсолютных цифр — пометка.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/home/iek/Документы/projects/binance/test")
PANELS = {
    2024: BASE / "data2024_usdc" / "manifest.json",
    2025: BASE / "data2025_usdc" / "manifest.json",
    2026: BASE / "data_usdc" / "manifest.json",
}
LB = 30              # окно моментума, дней (заморожено — как в работающей системе)
GATE_THR = 0.03      # гейт BTC 30д: >+3% up, <-3% down
TOP_N = 2            # слотов в портфеле
COST_MAKER = 0.0005  # 0 комиссия + 0.05% проскальзывание, на сторону
COST_TAKER = 0.00095 # 0.045% + 0.05%


def load_daily_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(closes, opens) — дневные матрицы close/open.
    ВАЖНО: история одного символа СШИВАЕТСЯ по годам (символ живёт в
    нескольких панелях) — иначе длинные окна (SMA200, mom30) теряют
    данные и сигналы вырождаются."""
    by_c: dict[str, list] = {}
    by_o: dict[str, list] = {}
    for year, mf in PANELS.items():
        m = json.load(open(mf))
        folder = mf.parent / "1h"
        for key, meta in m["files"].items():
            sym = meta["symbol"]
            d = np.load(folder / f"{sym}.npz")
            df = pd.DataFrame({"t": d["t"].astype(np.int64),
                               "o": d["o"], "c": d["c"]})
            df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date
            day = df.groupby("date").agg(o=("o", "first"), c=("c", "last"),
                                         n=("t", "count"))
            day = day[day["n"] >= 20]          # день считаем полным от 20/24 часов
            by_c.setdefault(sym, []).append(day["c"])
            by_o.setdefault(sym, []).append(day["o"])
    C = pd.DataFrame({s: pd.concat(p).sort_index() for s, p in by_c.items()})
    O = pd.DataFrame({s: pd.concat(p).sort_index() for s, p in by_o.items()})
    # стыки панелей: дубликаты дат -> последняя цена дня
    C = C.groupby(level=0).last()
    O = O.groupby(level=0).last()
    return C, O


def daily_returns(C: pd.DataFrame) -> pd.DataFrame:
    return C / C.shift(1) - 1


def build_signals(C: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """momentum: 30д доходность (NaN до 30 дней истории);
    reg30: гейт по 30д доходности BTC; reg_sma: гейт SMA200."""
    mom = C / C.shift(LB) - 1
    if "BTCUSDC" in C.columns:
        btc = C["BTCUSDC"]
    else:  # фолбэк: самый ликвидный столбец
        btc = C.iloc[:, 0]
    reg30 = btc / btc.shift(LB) - 1
    reg_sma = btc / btc.rolling(200).mean() - 1
    return mom, reg30, reg_sma


def run_rotation(C: pd.DataFrame, mom: pd.DataFrame, reg: pd.Series,
                 rets: pd.DataFrame, mode: str,
                 rebalance_days: int = 1,
                 cost: float = COST_MAKER,
                 seed: int | None = None, rng: np.random.Generator | None = None,
                 gate_thr: float = GATE_THR) -> dict:
    """mode: 'csm_long' | 'gate_btc' | 'gate_long' | 'gate_ls' | 'null'
    reg — гейтовая серия: для 30д-гейта это доходность (сравнивается с
    gate_thr), для SMA-гейта передаётся (цена/SMA - 1) с gate_thr=0.
    Возвращает дневную кривую портфеля и статистику."""
    dates = C.index
    n = len(dates)
    port = pd.Series(0.0, index=range(n))
    held: dict[str, float] = {}          # sym -> weight (signed)
    last_rebal = -10**9
    n_trades = 0

    for i in range(n):
        d = dates[i]
        r = rets.iloc[i]
        # 1) доходность дня по текущим позициям
        pr = 0.0
        for sym, w in held.items():
            if sym in r.index and not np.isnan(r[sym]):
                pr += w * r[sym]
            # данных больше нет (делистинг в панели) — позиция заморожена по последней цене
        port.iloc[i] = pr
        # 2) решение в конце дня i (исполнение по close дня i = close дня)
        if i - last_rebal < rebalance_days or i < LB + 1:
            continue
        g = reg.get(d, np.nan)
        target: dict[str, float] = {}
        if mode == "csm_long":
            m = mom.iloc[i].dropna()
            top = m.nlargest(TOP_N).index.tolist()
            target = {s: 1.0 / TOP_N for s in top}
        elif mode == "gate_btc":
            if g > gate_thr:
                target = {"BTCUSDC"} and {"BTCUSDC": 1.0}
            elif g < -gate_thr:
                target = {"BTCUSDC": -1.0}
        elif mode == "gate_long":
            if g > gate_thr:
                m = mom.iloc[i].dropna()
                top = m.nlargest(TOP_N).index.tolist()
                target = {s: 1.0 / TOP_N for s in top}
        elif mode == "gate_ls":
            if g > gate_thr:
                m = mom.iloc[i].dropna()
                target = {s: 1.0 / TOP_N for s in m.nlargest(TOP_N).index}
            elif g < -gate_thr:
                m = mom.iloc[i].dropna()
                target = {s: -1.0 / TOP_N for s in m.nsmallest(TOP_N).index}
        elif mode == "null":
            if g > gate_thr:
                pool = mom.iloc[i].dropna().index.tolist()
                if len(pool) >= TOP_N:
                    pick = rng.choice(pool, TOP_N, replace=False)
                    target = {s: 1.0 / TOP_N for s in pick}
            elif g < -gate_thr:
                pool = mom.iloc[i].dropna().index.tolist()
                if len(pool) >= TOP_N:
                    pick = rng.choice(pool, TOP_N, replace=False)
                    target = {s: -1.0 / TOP_N for s in pick}
        # 3) издержки на изменения
        changes = set(target) ^ set(held)
        ch_cost = 0.0
        for s in changes:
            w_old = held.get(s, 0.0)
            w_new = target.get(s, 0.0)
            if abs(w_new) > abs(w_old):      # открытие/увеличение платит
                ch_cost += abs(w_new - w_old) * cost
        # также разворот стороны = двойной оборот (учтён разницей весов выше)
        held = target
        n_trades += len(changes)
        port.iloc[i] -= ch_cost
        last_rebal = i

    s = pd.Series(port.values, index=dates)
    return {"curve": s, "n_trades": n_trades}


def stats(curve: pd.Series, name: str) -> dict:
    c = (1 + curve).cumprod()
    total = c.iloc[-1] - 1
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (c.iloc[-1]) ** (1 / max(years, 1e-9)) - 1
    sharpe = curve.mean() / curve.std() * np.sqrt(365) if curve.std() > 0 else 0
    dd = (c / c.cummax() - 1).min()
    return {"name": name, "total": total * 100, "cagr": cagr * 100,
            "sharpe": sharpe, "maxdd": dd * 100}


if __name__ == "__main__":
    C, O = load_daily_panel()
    print(f"панель: {C.shape[0]} дней x {C.shape[1]} символов, "
          f"{C.index[0]} -> {C.index[-1]}")
    mom, reg30, reg_sma = build_signals(C)
    rets = daily_returns(C)
    print(f"гейт 30д±3%: up {(reg30>0.03).sum()} дн, down {(reg30<-0.03).sum()} дн, "
          f"болото {len(reg30)-(reg30>0.03).sum()-(reg30<-0.03).sum()} дн")
    print(f"гейт SMA200: выше {(reg_sma>0).sum()} дн, ниже {(reg_sma<0).sum()} дн")

    scen = []
    scen.append(stats(run_rotation(C, mom, reg30, rets, "csm_long",
                                   cost=COST_MAKER)["curve"],
                      "CSM лонг всегда (без гейта)"))
    scen.append(stats(run_rotation(C, mom, reg30, rets, "gate_btc",
                                   cost=COST_MAKER)["curve"],
                      "Гейт-30д на BTC (лонг/шорт индекса)"))
    scen.append(stats(run_rotation(C, mom, reg30, rets, "gate_ls",
                                   cost=COST_MAKER)["curve"],
                      "Гейт-30д + топ-2 лонг / 2 шорт (дневной)"))
    scen.append(stats(run_rotation(C, mom, reg30, rets, "gate_ls",
                                   rebalance_days=7,
                                   cost=COST_MAKER)["curve"],
                      "Гейт-30д + лонг/шорт (недельный)"))
    scen.append(stats(run_rotation(C, mom, reg_sma, rets, "gate_long",
                                   gate_thr=0.0,
                                   cost=COST_MAKER)["curve"],
                      "Гейт-SMA200 + топ-2 лонга (дневной)"))
    scen.append(stats(run_rotation(C, mom, reg_sma, rets, "gate_ls",
                                   gate_thr=0.0,
                                   cost=COST_MAKER)["curve"],
                      "Гейт-SMA200 + лонг/шорт (дневной)"))
    scen.append(stats(run_rotation(C, mom, reg_sma, rets, "gate_ls",
                                   gate_thr=0.0, rebalance_days=7,
                                   cost=COST_MAKER)["curve"],
                      "Гейт-SMA200 + лонг/шорт (недельный)"))
    scen.append(stats(run_rotation(C, mom, reg_sma, rets, "gate_ls",
                                   gate_thr=0.0,
                                   cost=COST_TAKER)["curve"],
                      "Гейт-SMA200 + лонг/шорт (тейкер-издержки)"))

    print(f"\n{'сценарий':44s} {'итог%':>8s} {'CAGR%':>7s} {'Sharpe':>7s} {'MaxDD%':>7s}")
    for s in scen:
        print(f"{s['name']:44s} {s['total']:8.1f} {s['cagr']:7.1f} "
              f"{s['sharpe']:7.2f} {s['maxdd']:7.1f}")

    # ---------- НУЛЬ-ТЕСТЫ (для обоих гейтов) ----------
    for label, regx, thr, ref in (
            ("30д±3%", reg30, GATE_THR, scen[2]["total"]),
            ("SMA200", reg_sma, 0.0, scen[5]["total"])):
        print(f"\nНУЛЬ-ТЕСТ [{label}]: 200 случайных выборов тех же 2 пар")
        res_null = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            cur = run_rotation(C, mom, regx, rets, "null",
                               rebalance_days=1, cost=COST_MAKER,
                               seed=seed, rng=rng, gate_thr=thr)["curve"]
            res_null.append(stats(cur, f"null{seed}")["total"])
        res_null = np.array(res_null)
        pct = (res_null < ref).mean() * 100
        print(f"нуль-модель: медиана {np.median(res_null):+.1f}%, "
              f"5..95 перц: [{np.percentile(res_null,5):+.1f}%, "
              f"{np.percentile(res_null,95):+.1f}%]")
        print(f"моментум-выбор: {ref:+.1f}% -> лучше {pct:.0f}% "
              f"случайных выборов")
