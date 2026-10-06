"""ФИНАЛЬНАЯ МОДЕЛЬ «БУРЯ» ДЛЯ ЖИВОГО ДВИЖКА (предрегистрация 06.10).

Решение владельца: ставим ВМЕСТО текущего аналитика, лонги И шорты
(владелец верит в шорты — оставляем обе стороны, судит демо).

Обучение: та же панель 30м, те же 24 признака, МЕТКА-буря: y=1 если
fwd > 0.5 ATR(14), y=0 если fwd < -0.5 ATR, болото NaN. Обучение на
ВСЕЙ истории (как у боевой модели: чем больше — тем лучше дереву).
Порог в бою: лонг p>0.65, шорт p<0.35.

Соответствие боевому движку ml5h: 24 признака теми же формулами
(живой features.py: btc-признаки есть, sym_id категориальный),
Артефакты: ml5h.txt (модель) + ml5h_meta.json (gate 0.65, hold 10,
bar 30m, label storm_05atr, обе стороны). Деплой через kg_sync-обвязку
вручную (это НЕ плановое переобучение панели — обкатка нового вопроса).
"""
import glob, json, time, zipfile, io, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
H = 10
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
print("монет:", len(members), flush=True)
btc_pre = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
btc_pre["open_time"] = btc_pre["open_time"].astype(np.int64)
btc_pre.index = pd.to_datetime(btc_pre["open_time"], unit="ms")
btc = btc_pre["close"].astype("float32")

def make_feats(df, btc_close, sym):
    # формулы 1:1 с живым ml5h/features.py (паритет боевых признаков)
    close = df["close"]; high, low = df["high"], df["low"]
    rets = close.pct_change()
    X = pd.DataFrame(index=df.index)
    for h in (1, 4, 10, 20):
        X[f"ret_{h}"] = close.pct_change(h).astype("float32")
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    X["atr_pct"] = (tr.rolling(14).mean() / close).astype("float32")
    X["std_20"] = rets.rolling(20).std().astype("float32")
    X["vol_ratio"] = (df["volume"] / df["volume"].rolling(20).mean()
                      .replace(0, np.nan)).astype("float32")
    tb = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    X["taker_buy"] = tb.rolling(20, min_periods=10).mean().astype("float32")
    rng = (high - low).replace(0, np.nan)
    X["clv"] = ((close - low) / rng - 0.5).rolling(20, min_periods=10) \
        .mean().astype("float32")
    sma = close.rolling(200, min_periods=50).mean()
    std = close.rolling(200, min_periods=50).std()
    X["zscore"] = ((close - sma) / std).replace([np.inf, -np.inf], np.nan) \
        .astype("float32")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    X["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) \
        .astype("float32")
    X["skew"] = rets.rolling(20, min_periods=10).skew().astype("float32")
    sma50, sma20 = close.rolling(50).mean(), close.rolling(20).mean()
    X["sma_cross"] = ((close > sma50).astype("float32")
                      - (close > sma20).astype("float32"))
    if sym != "BTCUSDT" and btc_close is not None:
        b = btc_close.reindex(df.index).ffill()
        X["btc_ret_10"] = b.pct_change(10).astype("float32")
        X["btc_ret_20"] = b.pct_change(20).astype("float32")
        bret = b.pct_change()
        X["btc_corr"] = rets.rolling(50, min_periods=20).corr(bret) \
            .astype("float32")
        X["rel_ret_20"] = (rets.rolling(20).sum()
                           - bret.rolling(20).sum()).astype("float32")
        X["rel_ret_60"] = (rets.rolling(60).sum()
                           - bret.rolling(60).sum()).astype("float32")
        cov = rets.rolling(480).cov(bret); var = bret.rolling(480).var()
        X["beta_20"] = (cov / var.replace(0, np.nan)).astype("float32")
        X["corr_20"] = rets.rolling(480, min_periods=100).corr(bret) \
            .astype("float32")
    X["h4_momentum"] = close.pct_change(48).astype("float32")
    X["hour"] = df.index.hour.astype("float32")
    X["dow"] = df.index.dayofweek.astype("float32")
    X["sym_id"] = np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)
    return X, tr

frames = []
n_y1 = n_y0 = 0
for i, name in enumerate(members, 1):
    sym = name.replace("_30m.csv", "")
    df = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["open_time", "high", "low", "close",
                              "volume", "quote_volume", "taker_buy_base"])
    for c in df.columns:
        if c != "open_time":
            df[c] = df[c].astype("float32")
    df["open_time"] = df["open_time"].astype(np.int64)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    X, tr = make_feats(df, btc, sym)
    close = df["close"].values
    atr = (tr.rolling(14).mean() / close).values.astype("float64")
    fut = np.full(len(df), np.nan, dtype="float32")
    fut[:-H] = close[H:]
    fwd = (fut / close - 1).astype("float64")
    thr = 0.5 * np.nan_to_num(atr, nan=np.nan)
    y = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                 np.where(fwd > thr, 1.0,
                          np.where(fwd < -thr, 0.0, np.nan)))
    X["y"] = y.astype("float32")
    X["ot"] = df["open_time"].values.astype(np.int64)
    X = X.replace([np.inf, -np.inf], np.nan)
    g = X.dropna(subset=[c for c in X.columns if c != "y"])
    n_y1 += int((g["y"] == 1).sum())
    n_y0 += int((g["y"] == 0).sum())
    frames.append(g)
    if i % 150 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
data = data.dropna(subset=["y"]).reset_index(drop=True)
print(f"строк {len(data)}, класс 1 (буря вверх) {n_y1}, "
      f"класс 0 (буря вниз) {n_y0}, болото отброшено", flush=True)

FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
m = lgb.LGBMClassifier(**PARAMS)
m.fit(data[FEATS], data["y"], categorical_feature=["sym_id"])
m.booster_.save_model("/kaggle/working/ml5h.txt")
syms = sorted({n.replace("_30m.csv", "") for n in members})
json.dump({"trained_at": __import__("datetime").datetime.now(
    __import__("datetime").timezone.utc).isoformat(),
    "features": FEATS, "symbols": syms, "gate": 0.65,
    "hold_bars": H, "bar_minutes": 30,
    "label": "storm_05atr_both_sides",
    "train_rows": int(len(data)), "universe": "wide-522",
    "verdict": True, "note": "боевая модель-буря: лонг p>0.65 / "
    "шорт p<0.35 (решение владельца 06.10, замена аналитика)"},
    open("/kaggle/working/ml5h_meta.json", "w"), indent=1)
imp = sorted(zip(FEATS, m.booster_.feature_importance("gain")),
             key=lambda x: -x[1])[:10]
print("топ-важности:", [(f, round(v)) for f, v in imp], flush=True)
print("ФИНАЛЬНАЯ МОДЕЛЬ-БУРЯ СОХРАНЕНА", flush=True)
