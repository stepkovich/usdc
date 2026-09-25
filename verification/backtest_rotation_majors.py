"""Честный бэктест ротации моментума на ВСЕЛЕННОЙ МЕЙДЖОРОВ (спот 1d,
кэш zcodetrading — та же вселенная, где живая демо-система владельца
показывает плюс). Использует движок backtest_rotation (те же конвенции:
решение по close дня = цена исполнения, издержки на смену позиций,
нуль-перестановочный тест).

Кросс-чек контрпримера: панели USDC-M альтов 2024-2026 — кровопускание
(случайный выбор −85%), моментум относительно лучше (94-98 перцентиль),
абсолютно убыточно. Здесь: мейджоры 2023-2026, 20 крупнейших.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_rotation import (run_rotation, stats, build_signals,
                               daily_returns, COST_MAKER, COST_TAKER,
                               GATE_THR)

CACHE = Path("/home/iek/PycharmProjects/zcodetrading/data/cache")


def load_majors() -> pd.DataFrame:
    cols = {}
    for f in sorted(CACHE.glob("*_1d_v2.json")):
        sym = f.name.split("_")[0]
        d = json.load(open(f))
        rows = [(c["open_time"], float(c["open"]), float(c["close"]))
                for c in d["candles"]]
        s = pd.Series({pd.to_datetime(t, unit="ms", utc=True).date(): cl
                       for t, o, cl in rows})
        cols[sym] = s
    C = pd.DataFrame(cols).sort_index()
    C = C[~C.index.duplicated(keep="last")]
    return C


if __name__ == "__main__":
    import json
    C = load_majors()
    print(f"мейджоры: {C.shape[0]} дней x {C.shape[1]} символов, "
          f"{C.index[0]} -> {C.index[-1]}")
    mom, reg30, reg_sma = build_signals(C)
    rets = daily_returns(C)
    print(f"гейт 30д±3%: up {(reg30>0.03).sum()}, down {(reg30<-0.03).sum()}, "
          f"болото {(reg30.between(-0.03,0.03)).sum()}")
    print(f"гейт SMA200: выше {(reg_sma>0).sum()}, ниже {(reg_sma<0).sum()}")

    scen = []
    scen.append(stats(run_rotation(C, mom, reg30, rets, "csm_long")["curve"],
                      "CSM лонг всегда (без гейта)"))
    scen.append(stats(run_rotation(C, mom, reg30, rets, "gate_ls")["curve"],
                      "Гейт-30д + топ-2 лонг/2 шорт (дневной)"))
    scen.append(stats(run_rotation(C, mom, reg30, rets, "gate_ls",
                                   rebalance_days=7)["curve"],
                      "Гейт-30д + лонг/шорт (недельный)"))
    scen.append(stats(run_rotation(C, mom, reg_sma, rets, "gate_long",
                                   gate_thr=0.0)["curve"],
                      "Гейт-SMA200 + топ-2 лонга (дневной)"))
    scen.append(stats(run_rotation(C, mom, reg_sma, rets, "gate_ls",
                                   gate_thr=0.0)["curve"],
                      "Гейт-SMA200 + лонг/шорт (дневной)"))
    scen.append(stats(run_rotation(C, mom, reg_sma, rets, "gate_ls",
                                   gate_thr=0.0, rebalance_days=7)["curve"],
                      "Гейт-SMA200 + лонг/шорт (недельный)"))
    scen.append(stats(run_rotation(C, mom, reg_sma, rets, "gate_ls",
                                   gate_thr=0.0,
                                   cost=COST_TAKER)["curve"],
                      "Гейт-SMA200 + лонг/шорт (тейкер)"))

    print(f"\n{'сценарий':44s} {'итог%':>8s} {'CAGR%':>7s} {'Sharpe':>7s} {'MaxDD%':>7s}")
    for s in scen:
        print(f"{s['name']:44s} {s['total']:8.1f} {s['cagr']:7.1f} "
              f"{s['sharpe']:7.2f} {s['maxdd']:7.1f}")

    for label, regx, thr, ref in (
            ("30д±3%", reg30, GATE_THR, scen[1]["total"]),
            ("SMA200", reg_sma, 0.0, scen[4]["total"])):
        print(f"\nНУЛЬ-ТЕСТ [{label}]: 200 случайных выборов 2 пар")
        res = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            cur = run_rotation(C, mom, regx, rets, "null",
                               seed=seed, rng=rng, gate_thr=thr)["curve"]
            res.append(stats(cur, f"n{seed}")["total"])
        res = np.array(res)
        pct = (res < ref).mean() * 100
        print(f"нуль: медиана {np.median(res):+.1f}%, "
              f"5..95: [{np.percentile(res,5):+.1f}%, {np.percentile(res,95):+.1f}%]")
        print(f"моментум: {ref:+.1f}% -> лучше {pct:.0f}% случайных")
