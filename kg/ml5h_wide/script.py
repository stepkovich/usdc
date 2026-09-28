"""Кегл-кернел: ШИРОКАЯ ML5H (все монеты панели + метка монеты).
Вход: output кернела usdc-panel-downloader (527 CSV 30м, 2 года).
Walk-forward 8 кварталов + нуль-тест; критерии деплоя жёсткие и
записаны заранее: средняя >= +0.10%, нуль-перцентиль >= 95,
плюсовых кварталов >= 5 из 8. verdict -> ml5h.txt + meta."""
import glob, json, zlib
import numpy as np, pandas as pd, lightgbm as lgb

H = 10          # горизонт 10 баров 30м = 5 часов
GATE = 0.55
COOLDOWN_MS = 10 * 1800_000
SLIP, FEE2 = 0.0005, 0.001
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")

def sym_id(sym): return np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)

import os, zipfile, io
zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
print("panel.zip найден:", zips[:2], flush=True)
if not zips:
    for root, dirs, fs in os.walk("/kaggle/input"):
        print(" ", root, len(fs), "файлов", flush=True)
    json.dump({"verdict": False, "error": "panel.zip not found",
               "input_tree": str(os.listdir("/kaggle/input"))},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
print("монет в панели:", len(members), flush=True)
if len(members) < 500:
    json.dump({"verdict": False, "error": f"panel members {len(members)}"},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
btc = None
parts = []
for i, name in enumerate(members, 1):
    sym = name.replace("_30m.csv", "")
    df = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["open_time", "high", "low", "close",
                              "volume", "quote_volume", "taker_buy_base"])
    for c in df.columns:
        df[c] = df[c].astype("float32")
    df["open_time"] = df["open_time"].astype(np.int64)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    close = df["close"]
    rets = close.pct_change()
    X = pd.DataFrame(index=df.index)
    for h in (1, 4, 10, 20):
        X[f"ret_{h}"] = close.pct_change(h).astype("float32")
    tr = pd.concat([df["high"] - df["low"], (df["high"] - close.shift(1)).abs(),
                    (df["low"] - close.shift(1)).abs()], axis=1).max(axis=1)
    X["atr_pct"] = (tr.rolling(14).mean() / close).astype("float32")
    X["std_20"] = rets.rolling(20).std().astype("float32")
    X["vol_ratio"] = (df["volume"] / df["volume"].rolling(20).mean()
                      .replace(0, np.nan)).astype("float32")
    tb_ratio = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    X["taker_buy"] = tb_ratio.rolling(20, min_periods=10).mean().astype("float32")
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    X["clv"] = ((close - df["low"]) / rng - 0.5).rolling(20, min_periods=10) \
        .mean().astype("float32")
    sma = close.rolling(200, min_periods=50).mean()
    std = close.rolling(200, min_periods=50).std()
    X["zscore"] = ((close - sma) / std).replace([np.inf, -np.inf], np.nan) \
        .astype("float32")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    X["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).astype("float32")
    X["skew"] = rets.rolling(20, min_periods=10).skew().astype("float32")
    sma50, sma20 = close.rolling(50).mean(), close.rolling(20).mean()
    X["sma_cross"] = ((close > sma50).astype("float32")
                      - (close > sma20).astype("float32"))
    if sym != "BTCUSDT" and btc is not None:
        b = btc.reindex(df.index).ffill()
        bret = b.pct_change()
        X["btc_ret_10"] = bret.rolling(10).sum().astype("float32") * 0 + b.pct_change(10).astype("float32")
        X["btc_ret_20"] = b.pct_change(20).astype("float32")
        X["btc_corr"] = rets.rolling(50, min_periods=20).corr(bret).astype("float32")
    h4 = close.resample("4h").last().dropna()
    X["h4_momentum"] = h4.pct_change(6).reindex(df.index, method="ffill") \
        .astype("float32")
    tsdt = df.index
    X["hour"] = tsdt.hour.astype("float32")
    X["dow"] = tsdt.dayofweek.astype("float32")
    # G-cross
    if sym != "BTCUSDT" and btc is not None:
        bret = btc.reindex(df.index).ffill().pct_change()
        X["rel_ret_20"] = (rets.rolling(20).sum() - bret.rolling(20).sum()).astype("float32")
        X["rel_ret_60"] = (rets.rolling(60).sum() - bret.rolling(60).sum()).astype("float32")
        cov = rets.rolling(480).cov(bret); var = bret.rolling(480).var()
        X["beta_20"] = (cov / var.replace(0, np.nan)).astype("float32")
        X["corr_20"] = rets.rolling(480, min_periods=100).corr(bret).astype("float32")
    else:
        for c in ("rel_ret_20", "rel_ret_60", "beta_20", "corr_20"):
            X[c] = np.float32(0)
    X["sym_id"] = sym_id(sym)
    fwd = np.full(len(df), np.nan, dtype="float32")
    fwd[:-H] = (close.values[H:] / close.values[:-H] - 1).astype("float32")
    X["y"] = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype("float32"))
    X["fwd"] = fwd
    X["close"] = close.values
    X["ot"] = df["open_time"].values
    X = X.replace([np.inf, -np.inf], np.nan)
    parts.append(X.dropna(subset=[c for c in X.columns if c not in ("fwd", "close")]))
    if i % 100 == 0:
        print(f"признаки {i}/{len(members)}", flush=True)
data = pd.concat(parts, ignore_index=True)
del parts
print("строк:", len(data), flush=True)
feats = [c for c in data.columns if c not in ("y", "fwd", "close", "ot")]
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
test_q = quarters[-8:]
print("тестовые кварталы:", test_q, flush=True)
all_trades, folds, nulls = [], [], []
for k, qq in enumerate(test_q):
    tr = data[data["quarter"] < qq]
    te = data[data["quarter"] == qq]
    if len(te) < 5000 or len(tr) < 200_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(tr[feats], tr["y"], categorical_feature=["sym_id"])
    p = m.predict_proba(te[feats])[:, 1]
    d = te.assign(p=p)
    d = d[d["p"] > GATE].sort_values(["sym_id", "ot"])
    kept, last = [], {}
    for row in d.itertuples():
        if row.ot - last.get(row.Index, -10**18) >= COOLDOWN_MS:
            kept.append(row); last[row.Index] = row.ot
    t = pd.DataFrame(kept)
    if len(t) == 0:
        folds.append({"quarter": qq, "trades": 0, "avg": None}); continue
    entry = t["close"].values * (1 + SLIP)
    exit_p = t["close"].values * (1 + t["fwd"].values) * (1 - SLIP)
    t["pnl"] = exit_p / entry - 1 - FEE2
    avg = float(t["pnl"].mean() * 100)
    folds.append({"quarter": qq, "trades": len(t), "avg": round(avg, 4)})
    print(f"фолд {qq}: {len(t)} сделок, {avg:+.4f}%", flush=True)
    if k >= 6 and len(t):
        rng = np.random.default_rng(7)
        nl = []
        for _ in range(50):
            idx = rng.integers(0, len(te), len(t))
            smp = te.iloc[idx]
            nl.extend((smp["close"].values * (1 + smp["fwd"].values)
                       * (1 - SLIP) / (smp["close"].values * (1 + SLIP))
                       - 1 - FEE2) * 100)
        if nl:
            pct = (np.array(nl) < avg).mean() * 100
            nulls.append(pct)
            print(f"  нуль: лучше {pct:.0f}%", flush=True)
    all_trades.append(t)
res = pd.concat(all_trades)
avg_all = float(res["pnl"].mean() * 100)
pos_q = sum(1 for f in folds if (f["avg"] or 0) > 0)
null_pct = int(np.mean(nulls)) if nulls else 0
verdict = bool(avg_all >= 0.10 and null_pct >= 95 and pos_q >= 5)
print(f"ИТОГ: {avg_all:+.4f}% | плюс-кварталов {pos_q}/{len(folds)} | "
      f"нуль {null_pct}% | ВЕРДИКТ {verdict}", flush=True)
json.dump({"avg": avg_all, "pos_quarters": pos_q, "null_pct": null_pct,
           "verdict": verdict, "folds": folds},
          open("/kaggle/working/report.json", "w"), indent=1)
if verdict:
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(data[feats], data["y"], categorical_feature=["sym_id"])
    m.booster_.save_model("/kaggle/working/ml5h.txt")
    syms = sorted({m.split("/")[-1].replace("_30m.csv", "") for m in members})
    json.dump({"trained_at": str(pd.Timestamp.utcnow()),
               "features": feats, "symbols": syms, "gate": GATE,
               "hold_bars": H, "bar_minutes": 30, "universe": "wide-kaggle",
               "train_rows": int(len(data))},
              open("/kaggle/working/ml5h_meta.json", "w"), indent=1)
    print("артефакт сохранён", flush=True)
