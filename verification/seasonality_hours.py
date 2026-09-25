"""Кандидат #2: сезонность часа суток (26.09).

Данные: edge_lab 15m, 10 ликвидных пар 2020-2026 (~2.3M баров).

Вопрос: есть ли устойчивый дрейф по часам суток (UTC), который
переживает разрез по времени? Метод:
- дрейф часа = средняя (close часа / open часа - 1) по всем дням и
  парам (4х15м бара на час);
- устойчивость: корреляция профиля ПЕРВОЙ половины (2020-2023) со
  ВТОРОЙ (2024-2026), согласованность знака по парам; бутстрэп-CI
  (ресемплинг дней, 500 итераций);
- цена вопроса: сравнение дрейфа с круговыми издержками (мейкер
  вход+выход ~5 бп, тейкер ~19 бп).
Никакой подгонки: смотрим ВЕСЬ профиль, а не лучший час.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/home/iek/PycharmProjects/edge_lab/data_cache/ml_klines_15m")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "FILUSDT",
           "NEARUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT", "ZECUSDT"]


def load_hourly() -> pd.DataFrame:
    """Часовые бары: open первого 15м, close последнего 15м часа."""
    frames = []
    for sym in SYMBOLS:
        df = pd.read_csv(DATA / f"{sym}_15m.csv",
                         usecols=["open_time", "open", "close"])
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["sym"] = sym
        frames.append(df)
    df = pd.concat(frames)
    df["hour_start"] = df["ts"].dt.floor("h")
    g = df.groupby(["sym", "hour_start"]).agg(
        o=("open", "first"), c=("close", "last"))
    g = g.reset_index()
    g["ret"] = g["c"] / g["o"] - 1
    g["hour"] = g["hour_start"].dt.hour
    g["year"] = g["hour_start"].dt.year
    return g


def profile(g: pd.DataFrame, value: str = "ret") -> pd.Series:
    return g.groupby("hour")[value].mean() * 10000       # бп


def bootstrap_ci(g: pd.DataFrame, hours, n_iter: int = 500):
    """CI среднего дрейфа часа: ресемплинг ДНЕЙ (устойчивость к кластерам)."""
    g2 = g.copy()
    g2["day"] = g2["hour_start"].dt.date
    rng = np.random.default_rng(7)
    out = {}
    for h in hours:
        sub = g2[g2["hour"] == h]
        days = sub["day"].unique()
        by_day = sub.groupby("day")["ret"].mean().values * 10000
        if len(by_day) < 50:
            out[h] = (np.nan, np.nan)
            continue
        means = [by_day[rng.integers(0, len(by_day), len(by_day))].mean()
                 for _ in range(n_iter)]
        out[h] = (np.percentile(means, 2.5), np.percentile(means, 97.5))
    return out


if __name__ == "__main__":
    g = load_hourly()
    print(f"часовых наблюдений: {len(g)} ({g['sym'].nunique()} пар, "
          f"{g['year'].min()}-{g['year'].max()})")

    # --- полный профиль ---
    prof = profile(g)
    print("\nДрейф по часам UTC (бп/час), все пары 2020-2026:")
    for h in range(24):
        bar = "#" * max(0, int(abs(prof[h]) * 2)) * (1 if prof[h] >= 0 else -1)
        print(f"  {h:02d}:00 {prof[h]:+7.2f}  {bar}")

    # --- устойчивость: первая половина vs вторая ---
    g1 = g[g["year"] <= 2023]
    g2 = g[g["year"] >= 2024]
    p1, p2 = profile(g1), profile(g2)
    corr = np.corrcoef(p1.values, p2.values)[0, 1]
    same_sign = int((np.sign(p1) == np.sign(p2)).sum())
    print(f"\nкорреляция профилей 2020-2023 vs 2024-2026: {corr:+.2f}; "
          f"совпадение знака часа: {same_sign}/24")

    # --- согласованность по парам (полный период): топ/худший час ---
    sym_prof = g.groupby(["sym", "hour"])["ret"].mean().unstack() * 10000
    best_hour = prof.idxmax()
    worst_hour = prof.idxmin()
    b = sym_prof[best_hour]
    w = sym_prof[worst_hour]
    print(f"\nлучший час {best_hour:02d}:00 {prof[best_hour]:+.2f} бп: "
          f"плюс у {(b>0).sum()}/{len(b)} пар (медиана {b.median():+.2f})")
    print(f"худший час {worst_hour:02d}:00 {prof[worst_hour]:+.2f} бп: "
          f"минус у {(w<0).sum()}/{len(w)} пар (медиана {w.median():+.2f})")

    # --- бутстрэп-CI для крайних часов ---
    ci = bootstrap_ci(g, [best_hour, worst_hour])
    for h, (lo, hi) in ci.items():
        print(f"CI 95% час {h:02d}:00: [{lo:+.2f}, {hi:+.2f}] бп "
              f"({'значим' if lo>0 or hi<0 else 'НЕ значим: 0 внутри'})")

    # --- цена вопроса ---
    print(f"\nиздержки круга: мейкер~5 бп, тейкер~19 бп; "
          f"лучший час даёт {prof[best_hour]:+.2f} бп/час дрейфа")
    print("вывод: фильтр по часу имеет смысл, только если дрейф > издержек "
          "и устойчив в обеих половинах")
