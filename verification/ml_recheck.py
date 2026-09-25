"""Кандидат #3: перепроверка ML-двигателя edge_lab нашей честной линейкой.

Их движок (research/run_ml_honest.py): 30м бары, LightGBM, walk-forward
переобучение каждые 20 дней, лонг при proba>0.55, выход через 10 баров
(5ч), издержки 0.2% на круг. Заявлено: 55 562 сделки, WR 50.5%,
суммарный PnL +9395%.

Проверяем четыре вещи:
1. РЕКОНЦИЛЯЦИЯ: воспроизводятся ли их цифры из их же CSV сделок.
2. КАПИТАЛ-НОРМИРОВАНИЕ: +9395% = сумма сделок при ПОЛНОМ нотионале
   каждой, при 20 одновременных позициях. Честный портфель = 1/20 на
   сделку -> компаунд.
3. МОНОТОННОСТЬ: сделки с большей уверенностью модели (proba) должны
   зарабатывать больше. Если нет - сигнал фиктивен.
4. НУЛЬ-ТЕСТ: случайные лонги ТОЙ ЖЕ ПЛОТНОСТИ (по годам и символам
   совпадает число входов, cooldown 10 баров) с теми же издержками и
   тем же горизонтом. Если ML не лучше случайных входов - весь плюс
   это дрейф рынка, а не сигнал.
Сурвайвормешип вселенной (топ-20 по объёму на 2026) общий для ML и
нуля -> относительное сравнение честное.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRADES = Path("/home/iek/PycharmProjects/edge_lab/reports/ml_honest_trades.csv")
BARS = Path("/home/iek/PycharmProjects/edge_lab/data_cache/ml_klines")
HORIZON = 10          # баров 30м
COOLDOWN = 10
SLIP = 0.0005
FEE2 = 0.001          # 2 x taker 0.05%
N_SLOTS = 20          # топ-20 символов = максимум одновременных позиций


def same_pnl(close_entry: float, close_exit: float) -> float:
    entry = close_entry * (1 + SLIP)
    exit_p = close_exit * (1 - SLIP)
    return exit_p / entry - 1 - FEE2


def main() -> None:
    t = pd.read_csv(TRADES)
    t["time"] = t["time"].astype(np.int64)

    # ---------- 1. реконциляция ----------
    n = len(t)
    wr = t["won"].mean() * 100
    total = t["pnl"].sum() * 100
    avg = t["pnl"].mean() * 100
    print("=== 1. РЕКОНЦИЛЯЦИЯ их CSV ===")
    print(f"сделок {n}, WR {wr:.1f}%, сумма {total:+.0f}% (их учёт: полный "
          f"нотионал на каждую), средняя {avg:+.4f}%")

    t["year"] = pd.to_datetime(t["time"], unit="ms").dt.year
    print("\nпо годам (их учёт):")
    print(t.groupby("year")["pnl"].agg(["count", "sum"]).assign(
        sum_pct=lambda d: d["sum"] * 100).round(1).to_string())

    # ---------- 2. капитал-нормирование ----------
    t["exit_time"] = t["time"] + HORIZON * 1800_000
    port = t.groupby("exit_time")["pnl"].sum() / N_SLOTS
    port = port.sort_index()
    eq = (1 + port).cumprod()
    yrs = (t["exit_time"].max() - t["time"].min()) / (365.25 * 86400_000)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    print("\n=== 2. КАПИТАЛ-НОРМИРОВАНИЕ (1/20 на сделку) ===")
    print(f"компаунд за {yrs:.1f} лет: x{eq.iloc[-1]:.2f} "
          f"(CAGR {cagr*100:+.1f}%) против их заголовка {total:+.0f}%")

    # ---------- 3. монотонность по proba ----------
    print("\n=== 3. МОНОТОННОСТЬ по уверенности модели ===")
    t["proba_bin"] = pd.qcut(t["proba"], 5, labels=False, duplicates="drop")
    mono = t.groupby("proba_bin")["pnl"].agg(["count", "mean"])
    for b, r in mono.iterrows():
        print(f"  квинтиль proba {b}: {int(r['count'])} сделок, "
              f"ср. {r['mean']*100:+.4f}%")

    # ---------- 4. нуль-тест ----------
    print("\n=== 4. НУЛЬ-ТЕСТ: случайные лонги той же плотности ===")
    syms = sorted(t["symbol"].unique())
    counts = t.groupby(["symbol", "year"]).size().to_dict()
    bars_cache = {}
    for sym in syms:
        f = BARS / f"{sym}_30m.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=["open_time", "close"])
        df = df.dropna().reset_index(drop=True)
        if len(df) < HORIZON + 50:
            continue
        df["year"] = pd.to_datetime(df["open_time"], unit="ms").dt.year
        bars_cache[sym] = df
    print(f"баров загружено для {len(bars_cache)}/{len(syms)} символов")

    def random_once(rng: np.random.Generator) -> tuple[float, float]:
        """(sum pnl, avg pnl) случайного портфеля той же плотности."""
        all_pnl = []
        for (sym, year), k in counts.items():
            df = bars_cache.get(sym)
            if df is None:
                continue
            sub = df[df["year"] == year]
            idx = np.arange(len(sub) - HORIZON)
            if len(idx) < 1:
                continue
            order = rng.permutation(idx)
            picked, last = [], -10**9
            for i in order:
                if i - last >= COOLDOWN:
                    picked.append(i)
                    last = i
                if len(picked) >= k:
                    break
            c = sub["close"].values
            for i in picked:
                all_pnl.append(same_pnl(c[i], c[i + HORIZON]))
        a = np.array(all_pnl)
        return a.sum(), a.mean()

    rng = np.random.default_rng(42)
    null_sum, null_avg = [], []
    for it in range(200):
        s, a = random_once(rng)
        null_sum.append(s)
        null_avg.append(a)
    null_sum = np.array(null_sum) * 100
    null_avg = np.array(null_avg) * 100
    ml_avg = avg
    ml_sum = total
    pct_avg = (null_avg < ml_avg).mean() * 100
    pct_sum = (null_sum < ml_sum).mean() * 100
    print(f"нуль: средняя сделка медиана {np.median(null_avg):+.4f}%, "
          f"5..95 перц [{np.percentile(null_avg,5):+.4f}%, "
          f"{np.percentile(null_avg,95):+.4f}%]")
    print(f"ML:  средняя сделка {ml_avg:+.4f}% -> лучше {pct_avg:.0f}% "
          f"случайных портфелей")
    print(f"нуль: сумма медиана {np.median(null_sum):+.0f}%; "
          f"ML сумма {ml_sum:+.0f}% -> лучше {pct_sum:.0f}%")


if __name__ == "__main__":
    main()
