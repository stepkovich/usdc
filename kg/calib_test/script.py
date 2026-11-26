"""ТЕСТ ПОПРАВКИ КАЛИБРОВКИ (07.11): переносится ли карта «сказано->истина» между кварталами, и даёт ли гейт по настоящим процентам
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


# === НАСТОЯЩИЙ ТЕСТ: leave-one-fold-out ===
folds_ok = sorted(curves.keys())
from sklearn.isotonic import IsotonicRegression
rows = []
for k in folds_ok:
    other = [q for q in folds_ok if q != k]
    tp = np.concatenate([curves[q][0] for q in other])
    ty = np.concatenate([curves[q][1] for q in other])
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(tp, ty)
    hp, hy = curves[k]
    cp = iso.predict(hp)
    b_raw = float(np.mean((hp - hy) ** 2))
    b_corr = float(np.mean((cp - hy) ** 2))
    # отбор: сырой гейт 0.65 против честного гейта 60% после поправки
    m_raw = hp >= 0.65
    m_c60 = cp >= 0.60
    m_c65 = cp >= 0.65
    rows.append({
        "fold": k,
        "brier_raw": round(b_raw, 5), "brier_corr": round(b_corr, 5),
        "raw_n": int(m_raw.sum()),
        "raw_win": round(float(hy[m_raw].mean() * 100), 1) if m_raw.sum() else None,
        "c60_n": int(m_c60.sum()),
        "c60_win": round(float(hy[m_c60].mean() * 100), 1) if m_c60.sum() else None,
        "c65_n": int(m_c65.sum()),
        "c65_win": round(float(hy[m_c65].mean() * 100), 1) if m_c65.sum() else None,
    })
    print(f"фолд {k}: Брайер сырой {b_raw:.4f} vs поправленный {b_corr:.4f} | "
          f"сырой гейт {int(m_raw.sum())} шт win {rows[-1]['raw_win']}% | "
          f"честный 60% {int(m_c60.sum())} шт win {rows[-1]['c60_win']}%", flush=True)

n_br = sum(1 for r in rows if r["brier_corr"] < r["brier_raw"])
print(f"\nпоправка улучшила Брайер в {n_br}/{len(rows)} фолдах", flush=True)
import statistics as st
avg_raw_gate = st.mean([r["raw_win"] for r in rows if r["raw_win"] is not None and r["raw_n"] >= 100])
avg_c60 = st.mean([r["c60_win"] for r in rows if r["c60_win"] is not None and r["c60_n"] >= 100])
print(f"средний win: сырой гейт 0.65 = {avg_raw_gate:.1f}% | честный 60% = {avg_c60:.1f}%", flush=True)

with open("/kaggle/working/calib_test_report.json", "w") as f:
    json.dump({"folds": rows, "n_brier_improved": n_br,
               "avg_raw65_win": round(avg_raw_gate, 2),
               "avg_c60_win": round(avg_c60, 2)}, f, indent=1)
log("ТЕСТ ПОПРАВКИ ГОТОВ")
