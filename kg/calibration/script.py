"""КАЛИБРОВКА ВЕРОЯТНОСТЕЙ БУРИ (идея владельца 07.11, очередь 3).
Вопрос: что на самом деле означает уверенность модели «0.72»?
Протокол: 8 честных фолдов (обучение строго до своего квартала, та же
модель/признаки/метка, что в бою и в матрице). Для каждого экзамен-
квартала собираем предсказания, сортируем по уверенности, режем на
корзины по 0.05 и меряем РЕАЛЬНУЮ частоту роста. Кривая «сказал ->
реально» = калибровка. Дополнительно: изотоническая поправка (карта
сказано->истина) на 7 фолдах, проверенная на 8-м (hold-out).
Результат: таблица корзин + данные для переводной таблицы в бою.
"""
import io, json, os, time, zipfile, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
GATE = 0.65
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
SYM_IDX = FEATS.index("sym_id")


def log(m):
    print(f"{m} | {time.time()-t0:.0f}с", flush=True)


def make_feats(df, btc_close, sym):
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


def find_zip(name):
    for root, _d, fs in os.walk("/kaggle/input"):
        if name in fs:
            return os.path.join(root, name)
    return None


def build_arrays(zpath):
    zf = zipfile.ZipFile(zpath)
    members = sorted(n for n in zf.namelist() if n.endswith("_30m.csv"))
    log(f"панель: {len(members)} монет")
    btc = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
    btc["open_time"] = btc["open_time"].astype(np.int64)
    btc.index = pd.to_datetime(btc["open_time"], unit="ms")
    btc = btc["close"].astype("float32")
    Fs, ys, ots = [], [], []
    for i, name in enumerate(members, 1):
        sym = name.replace("_30m.csv", "")
        df = pd.read_csv(io.BytesIO(zf.read(name)),
                         usecols=["open_time", "high", "low", "close",
                                  "volume", "quote_volume",
                                  "taker_buy_base"])
        for c in df.columns:
            if c != "open_time":
                df[c] = df[c].astype("float32")
        df["open_time"] = df["open_time"].astype(np.int64)
        df.index = pd.to_datetime(df["open_time"], unit="ms")
        X, tr = make_feats(df, btc, sym)
        close = df["close"].values
        atr = (tr.rolling(14).mean() / close).values.astype("float64")
        fut = np.full(len(df), np.nan, dtype="float32")
        fut[:-10] = close[10:]
        fwd = (fut / close - 1).astype("float64")
        thr = 0.5 * np.nan_to_num(atr, nan=np.nan)
        y = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                     np.where(fwd > thr, 1.0,
                              np.where(fwd < -thr, 0.0, np.nan)))
        X["y"] = y.astype("float32")
        X = X.replace([np.inf, -np.inf], np.nan)
        fcols = [c for c in FEATS if c in X.columns]
        keep = X[fcols + ["y"]].notna().all(axis=1).values
        ot_i = df["open_time"].values.astype(np.int64)
        if keep.any():
            Fs.append(X.loc[keep, fcols]
                      .reindex(columns=FEATS).to_numpy(dtype="float32"))
            ys.append(y[keep].astype("float32"))
            ots.append(ot_i[keep])
        del X, df
        if i % 150 == 0:
            log(f"{i}/{len(members)}")
    F = np.concatenate(Fs); del Fs
    return {"F": F, "y": np.concatenate(ys), "ot": np.concatenate(ots)}


zp = find_zip("panel.zip")
assert zp, "panel.zip не найден"
A = build_arrays(zp)
_dti = pd.DatetimeIndex(pd.to_datetime(A["ot"], unit="ms"))  # ms явно
A["qc"] = ((_dti.year - 2020) * 4 + _dti.quarter).astype(np.int16)
del _dti
codes = sorted(int(c) for c in np.unique(A["qc"]))
qq_all = [f"{2020 + c // 4}Q{c % 4 + 1}" for c in codes]
_ix = np.linspace(0, len(qq_all) - 1, 8).round().astype(int)
TEST_Q = [qq_all[i] for i in _ix]
log(f"экзамен: {TEST_Q}")


def qcode_from_str(qq):
    y, q = qq.split("Q")
    return (int(y) - 2020) * 4 + int(q)


BUCKETS = np.arange(0.0, 1.0001, 0.05)
curves = {}      # qq -> список (p, y)
for qq in TEST_Q:
    code = qcode_from_str(qq)
    mtr = np.flatnonzero(A["qc"] < code)
    mte = np.flatnonzero(A["qc"] == code)
    if len(mte) < 5000 or len(mtr) < 200_000:
        log(f"{qq}: пропущен")
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(A["F"][mtr], A["y"][mtr], categorical_feature=[SYM_IDX])
    p = m.predict_proba(A["F"][mte])[:, 1]
    y = A["y"][mte]
    curves[qq] = (p, y)
    log(f"{qq}: баров {len(mte)}")
    del m

# === КРИВАЯ КАЛИБРОВКИ по корзинам (все фолды вместе) ===
allP = np.concatenate([c[0] for c in curves.values()])
allY = np.concatenate([c[1] for c in curves.values()])
print("\nКОРЗИНА | предсказаний | РЕАЛЬНАЯ частота роста | калибровка", flush=True)
rows = []
for lo in BUCKETS[:-1]:
    hi = lo + 0.05
    m = (allP >= lo) & (allP < hi)
    if m.sum() < 200:
        continue
    real = allY[m].mean()
    rows.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2),
                 "n": int(m.sum()),
                 "real": round(float(real), 4),
                 "mid": round(float(lo + 0.025), 3)})
    print(f"{lo:.2f}-{hi:.2f} | {int(m.sum()):8d} | {real*100:5.1f}%",
          flush=True)

# === ИЗОТОНИЧЕСКАЯ ПОПРАВКА: учимся на 7 фолдах, проверяем на 8-м ===
from sklearn.isotonic import IsotonicRegression
train_q = TEST_Q[:-1]
hold_q = TEST_Q[-1]
tp = np.concatenate([curves[q][0] for q in train_q if q in curves])
ty = np.concatenate([curves[q][1] for q in train_q if q in curves])
iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
iso.fit(tp, ty)
hp, hy = curves.get(hold_q, (np.array([]), np.array([])))
if len(hp):
    corrected = iso.predict(hp)
    print(f"\nПРОВЕРКА ПОПРАВКИ на невиданном фолде {hold_q}:", flush=True)
    print("корзина | сказано(сырое) | после поправки | реально", flush=True)
    check = []
    for lo in BUCKETS[:-1]:
        hi = lo + 0.05
        m = (hp >= lo) & (hp < hi)
        if m.sum() < 100:
            continue
        raw = hp[m].mean(); corr = corrected[m].mean(); real = hy[m].mean()
        check.append({"raw": round(float(raw), 3),
                      "corr": round(float(corr), 3),
                      "real": round(float(real), 3), "n": int(m.sum())})
        print(f"{lo:.2f}-{hi:.2f} | {raw*100:5.1f}% | "
              f"{corr*100:5.1f}% | {real*100:5.1f}%", flush=True)
else:
    check = []
    log(f"{hold_q}: нет данных")

with open("/kaggle/working/calibration_report.json", "w") as f:
    json.dump({"buckets": rows, "holdout_check": check,
               "holdout_quarter": hold_q,
               "iso_x": iso.X_thresholds_.tolist(),
               "iso_y": iso.y_thresholds_.tolist(),
               "test_quarters": TEST_Q}, f, indent=1, default=float)
log("КАЛИБРОВКА ГОТОВА")
