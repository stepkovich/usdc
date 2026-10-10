"""СЫРЫЕ CLOSE vs 24 ПРИЗНАКА (предрегистрация 09.10).

УПРОЩЁННАЯ ПРОБА: только close-цены последних 48 баров (z-норма),
без OHLCV, без формул — «может дерево само увидит паттерн в сырой
цене». Если и это не сработает — свечи пусты окончательно.

A = 24 признака (реплика боевой Бури), B = 48 z-normed closes.
Walk-forward 8 кварталов, gate 0.65, обе стороны, close[t+10],
taker 0.2% круг.
ВЕРДИКТ: B жив, если avg(B) − avg(A) >= +3 бп И n(B) >= 1000.
"""
import glob, io, json, time, zipfile, zlib
import glob, json, time, zipfile, zlib
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000
FEE_ROUND = 0.0020
WIN = 48
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
    X["zscore"] = ((close - sma) / std).replace([np.inf, -np.inf],
                                                np.nan).astype("float32")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    X["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) \
        .astype("float32")
    X["skew"] = rets.rolling(20, min_periods=10).skew().astype("float32")
    sma50, sma20 = close.rolling(50).mean(), close.rolling(20).mean()
    X["sma_cross"] = ((close > sma50).astype("float32")
                      - (close > sma20).astype("float32"))
    for c in ("btc_ret_10", "btc_ret_20", "btc_corr", "rel_ret_20",
              "rel_ret_60", "beta_20", "corr_20"):
        X[c] = np.float32(0)
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

FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]

frames = []
raw_lookup = {}
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
    close = df["close"].values.astype("float64")
    atr_rel = (tr.rolling(14).mean() / close).values.astype("float64")
    fut = np.full(len(df), np.nan, dtype="float64")
    fut[:-H] = close[H:]
    fwd = fut / close - 1
    thr = 0.5 * np.nan_to_num(atr_rel, nan=np.nan)
    X["y"] = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                      np.where(fwd > thr, 1.0,
                               np.where(fwd < -thr, 0.0, np.nan)))
    X["fwd"] = fwd.astype("float32")
    X["ot"] = df["open_time"].values.astype(np.int64)
    X["sym"] = sym
    # сырой ряд: closes в float16
    cl16 = close.astype(np.float32)
    raw_lookup[sym] = {"close": cl16, "atr": atr_rel.astype(np.float32),
                       "fwd": fwd.astype(np.float32),
                       "ot": df["open_time"].values,
                       "nan_mask": np.isnan(fwd) | np.isnan(thr)}
    frames.append(X.dropna(subset=FEATS))
    if i % 100 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx_q = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx_q]
print(f"панель {len(data)}, фолды {test_q}", flush=True)

# ---------- walk-forward ----------
report = {"pre_reg": "B жив: avg(B)-avg(A)>=+3 бп, n>=1000",
          "folds": {}}
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 20000 or len(trd) < 500_000:
        continue
    trd_ok = trd.dropna(subset=FEATS + ["y"])
    ted_ok = ted.dropna(subset=FEATS + ["y"])
    # A: 24 признака
    mA = lgb.LGBMClassifier(**PARAMS)
    mA.fit(trd_ok[FEATS], trd_ok["y"], categorical_feature=["sym_id"])
    pA = mA.predict_proba(ted_ok[FEATS])[:, 1]
    fwd_te = ted_ok["fwd"].values.astype("float64")
    acc_a = float(((pA > 0.5) == (ted_ok["y"] > 0.5)).mean() * 100)
    q20a, q80a = np.quantile(pA, [0.2, 0.8])
    sk_a = float((fwd_te[pA >= q80a].mean()
                  - fwd_te[pA <= q20a].mean()) * 10000)
    # B: сырые closes (48) — из raw_lookup
    n_test = len(ted_ok)
    raw_X = np.full((n_test, WIN), np.nan, dtype="float32")
    raw_y = np.full(n_test, np.nan)
    raw_fwd = np.full(n_test, np.nan)
    ts_arr = ted_ok["ot"].values
    sym_arr = ted_ok["sym"].values
    for sym in data['sym'].unique():
        w = raw_lookup.get(sym)
        if w is None:
            continue
        mask = sym_arr == sym
        if not mask.any():
            continue
        ots = ted_ok["ot"].values[mask]
        for j, ot in enumerate(ots):
            pos = np.searchsorted(w["ot"], ot)
            if pos >= len(w["close"]) or w["ot"][pos] != ot:
                continue
            global_idx = np.where(sym_arr == sym)[0][j]
            start = pos - WIN
            if start < 0:
                continue
            seg = w["close"][start:pos].astype("float64")
            if len(seg) != WIN or np.isnan(seg).any():
                continue
            mu = seg.mean(); sd = seg.std() + 1e-8
            if len(seg) != WIN:
                print(f"ДЕБАГ: seg len {len(seg)} != {WIN}", flush=True)
                continue
            raw_X[global_idx] = ((seg - mu) / sd).astype("float32")
            raw_y[global_idx] = w["fwd"][pos]
            raw_fwd[global_idx] = w["fwd"][pos]
    valid = ~np.isnan(raw_X).any(axis=1)
    Xv = raw_X[valid]
    yv = raw_y[valid]
    fv = raw_fwd[valid]
    if len(Xv) < 1000:
        print(f"фолд {qq}: мало валидных сырых окон ({len(Xv)})",
              flush=True)
        continue
    mB = lgb.LGBMClassifier(**PARAMS)
    mB.fit(Xv, yv.astype("int"))
    pB = mB.predict_proba(Xv)[:, 1]
    acc_b = float(((pB > 0.5) == (yv > 0.5)).mean() * 100)
    q20b, q80b = np.quantile(pB, [0.2, 0.8])
    sk_b = float((fv[pB >= q80b].mean() - fv[pB <= q20b].mean()) * 10000)
    q20a, q80a = np.quantile(pA[:len(yv)], [0.2, 0.8])
    sk_a = float((fv[pA[:len(yv)] >= q80a].mean()
                  - fv[pA[:len(yv)] <= q20a].mean()) * 10000)
    acc_a_f = float(((pA[:len(yv)] > 0.5) == (yv > 0.5)).mean() * 100)
    report["folds"][qq] = {"acc_A": round(acc_a_f, 2),
                           "acc_B": round(acc_b, 2),
                           "sk_A": round(sk_a, 2), "sk_B": round(sk_b, 2)}
    print(f"фолд {qq}: A acc {acc_a_f:.2f}% sk {sk_a:+.1f} | "
          f"B acc {acc_b:.2f}% sk {sk_b:+.1f} | {time.time()-t0:.0f}с",
          flush=True)

json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("СЫРЫЕ CLOSE завершена", flush=True)
