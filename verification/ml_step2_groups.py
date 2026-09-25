"""ШАГ 2 плана ML: добавление групп признаков ПО ОДНОЙ (26.09).

ПРЕДРЕГИСТРАЦИЯ (записано до запуска):
- база = шаг 1 (воспроизведён бит-в-бит: 55 562 сделки, +0.1691%/сделку);
- тестируются ровно 4 группы, каждая в ОДНОЙ конфигурации (без перебора
  параметров внутри группы — окна натуральные 10/20/50/240):

  G-funding: ставка фандинга (последнее 8ч-событие), средние за 7д и
    30д, z-скор против 30д, флаг наличия данных. До 2024 данных нет ->
    fill 0 + флаг. Источники: binance/test/data/funding (2025-26) +
    funding_2024.
  G-flow: дисбаланс taker-потока за 10 и 50 баров, z-скор потока (20),
    отношение quote_volume к своей средней (20).
  G-cross: контекст соседей: относительная доходность к BTC за 10ч и
    30ч, beta к BTC (480 баров = 10 дней), корреляция к BTC (480).
  G-volreg: режим волатильности: перцентиль ATR в окне 200, волатильность
    волатильности (std20 за 50), размах/ATR.

- ПРАВИЛО ОСТАВЛЕНИЯ (записано заранее): средняя сделка >= +5% к базе
  (>=0.1776%) И WR >= 50.0% И плюс-лет 7/7.
- ФИНАЛ: все удержанные группы вместе — один прогон, без перебора.

Строки и фолды те же, что в базе (маска только по базовым фичам) ->
чистая абляция: разница результатов только от новых колонок.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ml_train_baseline import (
    load_klines, get_top_liquid, make_features, TARGET_HORIZON,
    COOLDOWN_BARS, TRAIN_MS, RETRAIN_MS, MIN_TRAIN_SAMPLES, PROB_THRESHOLD,
    TAKER_FEE, SLIPPAGE, TOP_N_SYMBOLS, N_ESTIMATORS, LEARNING_RATE,
    MAX_DEPTH, SEED,
)

warnings.filterwarnings("ignore")
log = logging.getLogger("ml_step2")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

FUNDING_DIRS = [
    Path("/home/iek/Документы/projects/binance/test/data/funding_2024"),
    Path("/home/iek/Документы/projects/binance/test/data/funding"),
]
BASELINE_AVG = 0.001691
BASELINE_WR = 50.5
GROUPS = ["G-funding", "G-flow", "G-cross", "G-volreg"]


def load_funding(sym: str) -> pd.Series | None:
    parts = []
    for d in FUNDING_DIRS:
        f = d / f"{sym}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        if "t" not in z.files or "rate" not in z.files:
            continue
        idx = pd.to_datetime(z["t"].astype(np.int64), unit="ms")
        parts.append(pd.Series(z["rate"].astype(float), index=idx))
    if not parts:
        return None
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def group_columns(df: pd.DataFrame, btc_close, sym: str, group: str) \
        -> pd.DataFrame:
    """Колонки ОДНОЙ группы; всё backward-looking; NaN только там, где
    истории реально нет (заполняются до обучения)."""
    close = df["close"]
    high, low, vol = df["high"], df["low"], df["volume"]
    tb = df["taker_buy_base"]
    rets = close.pct_change()
    out = pd.DataFrame(index=df.index)
    if group == "G-funding":
        fr = load_funding(sym)
        if fr is None:
            out["funding"] = 0.0
            out["funding_7d"] = 0.0
            out["funding_30d"] = 0.0
            out["funding_z"] = 0.0
            out["has_funding"] = 0.0
        else:
            last = fr.reindex(df.index, method="ffill")
            out["funding"] = last
            out["funding_7d"] = last.rolling(56, min_periods=8).mean()
            out["funding_30d"] = last.rolling(240, min_periods=8).mean()
            m = last.rolling(240, min_periods=8).mean()
            s = last.rolling(240, min_periods=8).std()
            out["funding_z"] = (last - m) / s.replace(0, np.nan)
            out["has_funding"] = last.notna().astype(float)
    elif group == "G-flow":
        ratio = tb / vol.replace(0, np.nan)
        out["flow_10"] = ratio.rolling(10, min_periods=5).mean().sub(0.5)
        out["flow_50"] = ratio.rolling(50, min_periods=10).mean().sub(0.5)
        m = ratio.rolling(20, min_periods=10).mean().sub(0.5)
        s = ratio.rolling(20, min_periods=10).std()
        out["flow_z"] = m / s.replace(0, np.nan)
        out["qvol_ratio"] = df["quote_volume"] / \
            df["quote_volume"].rolling(20, min_periods=10).mean()
    elif group == "G-cross":
        if btc_close is None or sym == "BTCUSDT":
            out["rel_ret_20"] = 0.0
            out["rel_ret_60"] = 0.0
            out["beta_20"] = 0.0
            out["corr_20"] = 0.0
        else:
            btc = btc_close.reindex(df.index).ffill()
            bret = btc.pct_change()
            out["rel_ret_20"] = rets.rolling(20).sum() - bret.rolling(20).sum()
            out["rel_ret_60"] = rets.rolling(60).sum() - bret.rolling(60).sum()
            cov = rets.rolling(480).cov(bret)
            var = bret.rolling(480).var()
            out["beta_20"] = cov / var.replace(0, np.nan)
            out["corr_20"] = rets.rolling(480, min_periods=100).corr(bret)
    elif group == "G-volreg":
        tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean() / close
        out["atr_pctile"] = atr.rolling(200, min_periods=50).apply(
            lambda a: (a[-1] >= a).mean(), raw=True)
        out["vol_of_vol"] = rets.rolling(20).std().rolling(50).std()
        out["range_ratio"] = (high - low) / tr.rolling(14).mean() \
            .replace(0, np.nan)
    else:
        raise ValueError(group)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_config(built: dict, group_names: list[str]) -> pd.DataFrame:
    """built: sym -> {X_base, df, cols_new:{group: DataFrame}}; маска и
    фолды идентичны базе."""
    data = {}
    for sym, b in built.items():
        df = b["df"]
        X = b["X_base"].copy()
        for g in group_names:
            for c in b["cols_new"][g].columns:
                X[c] = b["cols_new"][g][c]
        mask = b["mask"]
        close = df["close"].values
        data[sym] = {"X": X[mask].reset_index(drop=True),
                     "y": b["y"][mask],
                     "times": df["open_time"].values[mask],
                     "closes": close[mask]}
    all_times = np.concatenate([d["times"] for d in data.values()])
    gmin, gmax = all_times.min(), all_times.max()
    trades = []
    start = gmin + TRAIN_MS
    while start + RETRAIN_MS <= gmax:
        tr_end, te_end = start, min(start + RETRAIN_MS, gmax)
        Xs, ys = [], []
        for sym, d in data.items():
            m = (d["times"] >= tr_end - TRAIN_MS) & (d["times"] < tr_end)
            if m.sum() > 200:
                Xs.append(d["X"][m])
                ys.append(d["y"][m])
        if not Xs:
            start += RETRAIN_MS
            continue
        X_tr = pd.concat(Xs, ignore_index=True)
        y_tr = np.concatenate(ys)
        if len(X_tr) < MIN_TRAIN_SAMPLES:
            start += RETRAIN_MS
            continue
        model = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbosity=-1,
            class_weight="balanced")
        model.fit(X_tr, y_tr)
        cols = X_tr.columns.tolist()
        for sym, d in data.items():
            m = (d["times"] >= tr_end) & (d["times"] < te_end)
            if m.sum() < 50:
                continue
            X_te = d["X"][m].reindex(columns=cols, fill_value=0)
            proba = model.predict_proba(X_te)[:, 1]
            sig = np.where(proba > PROB_THRESHOLD)[0]
            kept, last = [], -COOLDOWN_BARS
            for i in sig:
                if i - last >= COOLDOWN_BARS:
                    kept.append(i)
                    last = i
            closes_te = d["closes"][m]
            times_te = d["times"][m]
            for i in kept:
                if i + TARGET_HORIZON >= len(closes_te):
                    continue
                entry = closes_te[i] * (1 + SLIPPAGE)
                exit_p = closes_te[i + TARGET_HORIZON] * (1 - SLIPPAGE)
                trades.append({"time": times_te[i], "symbol": sym,
                               "pnl": exit_p / entry - 1 - 2 * TAKER_FEE,
                               "proba": float(proba[i])})
        start += RETRAIN_MS
    return pd.DataFrame(trades)


def report(t: pd.DataFrame, name: str) -> dict:
    t = t.sort_values("time").reset_index(drop=True)
    t["year"] = pd.to_datetime(t["time"], unit="ms").dt.year
    yr = t.groupby("year")["pnl"].sum()
    t["pb"] = pd.qcut(t["proba"], 5, labels=False, duplicates="drop")
    mono = t.groupby("pb")["pnl"].mean()
    r = {"n": len(t), "wr": (t["pnl"] > 0).mean() * 100,
         "avg": t["pnl"].mean() * 100, "sum": t["pnl"].sum() * 100,
         "pos_years": int((yr > 0).sum()),
         "mono_spread": (mono.max() - mono.min()) * 100}
    print(f"{name:26s} сделок {r['n']:6d} WR {r['wr']:5.1f}% "
          f"средняя {r['avg']:+.4f}% сумма {r['sum']:+8.0f}% "
          f"плюс-лет {r['pos_years']}/7 квинт-разброс {r['mono_spread']:.3f}%")
    return r


if __name__ == "__main__":
    caches = load_klines()
    top = get_top_liquid(caches, TOP_N_SYMBOLS)
    btc_close = caches.get("BTCUSDT", pd.DataFrame())["close"]
    log.info("готовлю признаки (%d символов)...", len(top))
    built = {}
    for sym in top:
        df = caches[sym]
        X_base = make_features(df, btc_close, sym) \
            .replace([np.inf, -np.inf], np.nan)
        mask = X_base.notna().all(axis=1).values
        close = df["close"].values
        fwd = np.full(len(df), np.nan)
        fwd[:-TARGET_HORIZON] = close[TARGET_HORIZON:] / close[:-TARGET_HORIZON] - 1
        y = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
        cols_new = {g: group_columns(df, btc_close, sym, g) for g in GROUPS}
        built[sym] = {"df": df, "X_base": X_base, "mask": mask, "y": y,
                      "cols_new": cols_new}
    log.info("признаки готовы; прогоны: база + 4 группы + финал")

    print(f"\n{'конфиг':26s}          (правило: ср >= +5% к базе, "
          f"WR >= 50.0, плюс-лет 7/7)")
    res = {}
    t = run_config(built, [])
    res["база"] = report(t, "база (шаг 1)")
    for g in GROUPS:
        t = run_config(built, [g])
        res[g] = report(t, g)
        keep = (res[g]["avg"] >= BASELINE_AVG * 100 * 1.05
                and res[g]["wr"] >= BASELINE_WR - 0.5
                and res[g]["pos_years"] >= 7)
        print(f"   -> {'ОСТАВЛЯЕМ' if keep else 'выбрасываем'}\n")
    kept = [g for g in GROUPS
            if (res[g]["avg"] >= BASELINE_AVG * 100 * 1.05
                and res[g]["wr"] >= BASELINE_WR - 0.5
                and res[g]["pos_years"] >= 7)]
    print(f"\nудержанные группы: {kept or 'НИ ОДНОЙ'}")
    if kept:
        t = run_config(built, kept)
        r = report(t, "ФИНАЛ: " + "+".join(kept))
        t.to_csv(Path(__file__).parent / "ml_step2_final_trades.csv",
                 index=False)
        print(f"сделки -> ml_step2_final_trades.csv")
