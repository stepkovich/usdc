"""ШАГ 1 плана ML: воспроизведение базовой модели СВОИМИ РУКАМИ (26.09).

ПРЕДРЕГИСТРАЦИЯ (параметры записаны и закоммичены ДО первого запуска,
никаких правок после просмотра результатов — иначе снова подгонка):

- данные: edge_lab ml_klines, 30м бары, топ-20 по среднему
  quote_volume за последние 30 дней (метод 1:1 из run_ml_honest.py);
- признаки: ИХ набор 1:1 (ретёрны 1/4/10/20, ATR, std, объём,
  taker_buy, clv, zscore, RSI, skew, SMA-кросс, BTC-контекст,
  h4_momentum, hour, dow);
- таргет: close через 10 баров выше? (бинарный);
- walk-forward: train 60 дней, переобучение каждые 20 дней, минимум
  1500 строк в train;
- сигнал: proba > 0.55, cooldown 10 баров после сигнала, одна позиция
  на символ;
- исполнение: вход close сигнального бара + 0.05% slip, выход close
  через 10 баров - 0.05% slip, комиссия тейкер 0.05% x2;
- модель: LightGBM 200 деревьев, depth 5, lr 0.05, seed 42.

Критерии успеха шага (записаны до запуска): средний PnL/сделку в
коридоре +0.10..+0.25%, плюс в большинстве лет, монотонность по
уверенности. Если не сошлось — СНАЧАЛА ищем ошибку воспроизведения,
никаких изменений параметров.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ml_baseline")

DATA = Path("/home/iek/PycharmProjects/edge_lab/data_cache/ml_klines")
OUT = Path(__file__).parent / "ml_baseline_trades.csv"

TARGET_HORIZON = 10
COOLDOWN_BARS = 10
TRAIN_MS = 60 * 86400_000
RETRAIN_MS = 20 * 86400_000
MIN_TRAIN_SAMPLES = 1500
PROB_THRESHOLD = 0.55
TAKER_FEE = 0.0005
SLIPPAGE = 0.0005
TOP_N_SYMBOLS = 20
N_ESTIMATORS = 200
LEARNING_RATE = 0.05
MAX_DEPTH = 5
SEED = 42


def load_klines() -> dict:
    result = {}
    for path in sorted(DATA.glob("*_30m.csv")):
        sym = path.stem.replace("_30m", "")
        df = pd.read_csv(path)
        if len(df) < 2000:
            continue
        for c in ["open", "high", "low", "close", "volume",
                  "taker_buy_base", "quote_volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close", "volume"]).reset_index(drop=True)
        df.index = pd.to_datetime(df["open_time"], unit="ms")
        result[sym] = df
    return result


def get_top_liquid(caches: dict, top_n: int) -> list:
    volumes = {}
    for sym, df in caches.items():
        recent = df.tail(1440)
        volumes[sym] = recent["quote_volume"].mean() if len(recent) else 0
    return [s for s, _ in sorted(volumes.items(), key=lambda x: -x[1])[:top_n]]


def make_features(df: pd.DataFrame, btc_close, symbol: str) -> pd.DataFrame:
    """1:1 признаки из run_ml_honest.py."""
    close = df["close"]
    high, low, vol = df["high"], df["low"], df["volume"]
    tb = df["taker_buy_base"]
    rets = close.pct_change()
    out = pd.DataFrame(index=df.index)
    for h in [1, 4, 10, 20]:
        out[f"ret_{h}"] = close.pct_change(h)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["atr_pct"] = tr.rolling(14).mean() / close
    out["std_20"] = rets.rolling(20).std()
    out["vol_ratio"] = vol / vol.rolling(20).mean().replace(0, np.nan)
    out["taker_buy"] = (tb / vol.replace(0, np.nan)).rolling(
        20, min_periods=10).mean()
    rng = (high - low).replace(0, np.nan)
    out["clv"] = ((close - low) / rng - 0.5).rolling(20, min_periods=10).mean()
    sma = close.rolling(200, min_periods=50).mean()
    std = close.rolling(200, min_periods=50).std()
    out["zscore"] = ((close - sma) / std).replace([np.inf, -np.inf], np.nan)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["skew"] = rets.rolling(20, min_periods=10).skew()
    sma50 = close.rolling(50).mean()
    sma20 = close.rolling(20).mean()
    out["sma_cross"] = ((close > sma50).astype(float)
                        - (close > sma20).astype(float))
    if btc_close is not None and symbol != "BTCUSDT":
        btc = btc_close.reindex(df.index).ffill()
        out["btc_ret_10"] = btc.pct_change(10)
        out["btc_ret_20"] = btc.pct_change(20)
        out["btc_corr"] = close.rolling(50, min_periods=20).corr(btc)
    h4 = close.resample("4h").last().dropna()
    out["h4_momentum"] = h4.pct_change(6).reindex(df.index, method="ffill")
    ts = pd.to_datetime(df["open_time"], unit="ms")
    out["hour"] = ts.dt.hour.values
    out["dow"] = ts.dt.dayofweek.values
    return out


def main() -> None:
    caches = load_klines()
    top = get_top_liquid(caches, TOP_N_SYMBOLS)
    log.info("топ-%d: %s", TOP_N_SYMBOLS, top[:5])
    btc_close = caches.get("BTCUSDT", pd.DataFrame())["close"]

    data = {}
    for sym in top:
        df = caches[sym]
        X = make_features(df, btc_close, sym).replace([np.inf, -np.inf], np.nan)
        close = df["close"].values
        fwd = np.full(len(df), np.nan)
        fwd[:-TARGET_HORIZON] = close[TARGET_HORIZON:] / close[:-TARGET_HORIZON] - 1
        y = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
        mask = (X.notna().all(axis=1) & ~np.isnan(y)).values
        if mask.sum() < MIN_TRAIN_SAMPLES:
            log.warning("%s: %d строк < минимума", sym, mask.sum())
            continue
        data[sym] = {"X": X[mask], "y": y[mask],
                     "times": df["open_time"].values[mask],
                     "closes": close[mask]}
    log.info("символов с данными: %d", len(data))

    all_times = np.concatenate([d["times"] for d in data.values()])
    gmin, gmax = all_times.min(), all_times.max()
    log.info("период: %s .. %s", pd.Timestamp(gmin, unit="ms"),
             pd.Timestamp(gmax, unit="ms"))

    trades = []
    folds = 0
    start = gmin + TRAIN_MS
    while start + RETRAIN_MS <= gmax:
        folds += 1
        tr_end = start
        te_end = min(start + RETRAIN_MS, gmax)
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
            times_te = d["times"][m]
            closes_te = d["closes"][m]
            proba = model.predict_proba(X_te)[:, 1]
            sig = np.where(proba > PROB_THRESHOLD)[0]
            kept, last = [], -COOLDOWN_BARS
            for i in sig:
                if i - last >= COOLDOWN_BARS:
                    kept.append(i)
                    last = i
            for i in kept:
                if i + TARGET_HORIZON >= len(closes_te):
                    continue
                entry = closes_te[i] * (1 + SLIPPAGE)
                exit_p = closes_te[i + TARGET_HORIZON] * (1 - SLIPPAGE)
                pnl = exit_p / entry - 1 - 2 * TAKER_FEE
                trades.append({"time": times_te[i], "symbol": sym,
                               "pnl": pnl, "won": pnl > 0,
                               "proba": float(proba[i])})
        if folds % 10 == 0:
            log.info("фолд %d, сделок %d", folds, len(trades))
        start += RETRAIN_MS

    t = pd.DataFrame(trades).sort_values("time").reset_index(drop=True)
    t["month"] = pd.to_datetime(t["time"], unit="ms").dt.to_period("M")
    n = len(t)
    wr = t["won"].mean() * 100
    print("\n=== БАЗА (своя тренировка, seed 42) ===")
    print(f"фолдов {folds}, сделок {n}, WR {wr:.1f}%, "
          f"сумма {t['pnl'].sum()*100:+.0f}%, средняя {t['pnl'].mean()*100:+.4f}%")
    t["year"] = pd.to_datetime(t["time"], unit="ms").dt.year
    print(t.groupby("year")["pnl"].agg(["count", "sum"]).assign(
        sum_pct=lambda d: d["sum"] * 100).round(1).to_string())
    print("\nмонотонность по proba (квинтили):")
    t["pb"] = pd.qcut(t["proba"], 5, labels=False, duplicates="drop")
    print(t.groupby("pb")["pnl"].agg(["count", "mean"]).assign(
        mean_pct=lambda d: d["mean"] * 100).round(4).to_string())
    port = t.groupby(t["time"] + TARGET_HORIZON * 1800_000)["pnl"].sum() / TOP_N_SYMBOLS
    eq = (1 + port.sort_index()).cumprod()
    yrs = (t["time"].max() - t["time"].min()) / (365.25 * 86400_000)
    print(f"\nкапитал-нормировано (1/{TOP_N_SYMBOLS} на сделку): "
          f"x{eq.iloc[-1]:.2f} за {yrs:.1f} лет")
    t.drop(columns=["month"]).to_csv(OUT, index=False)
    print(f"сделки -> {OUT}")


if __name__ == "__main__":
    main()
