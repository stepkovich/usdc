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
    X["low"] = df["low"].astype("float32")
    X["fi"] = np.uint16(i)
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
feats = [c for c in data.columns if c not in ("y", "fwd", "close", "ot", "low", "fi")]
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
# ПРЕДРЕГИСТРАЦИЯ v3 (до просмотра результата): 8 кварталов, индексы
# np.linspace по ВСЕЙ истории 2021-2026 — медведь 2021-2022 включён.
import numpy as _np
idx = _np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
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
        # пауза ПО МОНЕТЕ (sym_id), не по индексу строки — иначе паузы нет
        if row.ot - last.get(row.sym_id, -10**18) >= COOLDOWN_MS:
            kept.append(row); last[row.sym_id] = row.ot
    t = pd.DataFrame(kept)
    if len(t) == 0:
        folds.append({"quarter": qq, "trades": 0, "avg": None}); continue
    entry = t["close"].values * (1 + SLIP)
    exit_p = t["close"].values * (1 + t["fwd"].values) * (1 - SLIP)
    t["pnl"] = exit_p / entry - 1 - FEE2
    avg = float(t["pnl"].mean() * 100)
    folds.append({"quarter": qq, "trades": len(t), "avg": round(avg, 4)})
    print(f"фолд {qq}: {len(t)} сделок, {avg:+.4f}%", flush=True)
    # нуль-тест по КАЖДОМУ тестовому кварталу (а не только последним двум —
    # это была недоделка реализации, сужавшая сравнение до бычьих окон)
    # MAE: худший нырок каждой сделки за холд (10 баров после входа)
    pools = {fi: g for fi, g in te.groupby("fi")}
    t = t.reset_index(drop=True)
    mae_vals = np.zeros(len(t))
    for fi, g in t.groupby("fi"):
        pool = pools.get(fi)
        if pool is None:
            continue
        ots = pool["ot"].values
        lows = pool["low"].values
        for idx, row in g.iterrows():
            j = int(np.searchsorted(ots, row["ot"]))
            seg = lows[j + 1: j + 1 + 10]
            if len(seg):
                mae_vals[idx] = seg.min() / (row["close"] * (1 + SLIP)) - 1
    t["mae"] = mae_vals
    all_trades.append(t)
res = pd.concat(all_trades)
mae_all = np.concatenate([t["mae"].values for t in all_trades if len(t)])
pq = lambda p: round(float(np.percentile(mae_all, p)) * 100, 2)
worst = round(float(mae_all.min()) * 100, 2)
print(f"МАЕ нырки: медиана {pq(50):+.2f}% | p90 {pq(90):+.2f}% | p99 {pq(99):+.2f}% | худший {worst:+.2f}%", flush=True)
deep = res[res["mae"] <= -0.15]
n_deep = len(deep)
deep_avg = round(float(deep["pnl"].mean() * 100), 4) if n_deep else 0.0
deep_pos = round(float((deep["pnl"] > 0).mean() * 100), 1) if n_deep else 0.0
print(f"нырявших глубже -15%: {n_deep} | их финал {deep_avg:+.3f}% | "
      f"из них в плюсе {deep_pos}%", flush=True)
def stop_sim(level):
    stopped = (res["mae"] <= -level).values
    pnl_stop = np.where(stopped,
                        (1 - level) * (1 - SLIP) - 1 - FEE2,
                        res["pnl"].values)
    return int(stopped.sum()), round(float(pnl_stop.mean() * 100), 4)
n10, a10 = stop_sim(0.10)
n15, a15 = stop_sim(0.15)
n20, a20 = stop_sim(0.20)
base_avg = round(float(res["pnl"].mean() * 100), 4)
print(f"стоп -10%: сработал бы {n10} раз, средняя стала бы {a10:+.4f}%", flush=True)
print(f"стоп -15%: сработал бы {n15} раз, средняя стала бы {a15:+.4f}%", flush=True)
print(f"стоп -20% (ликвидация): {n20} раз, средняя {a20:+.4f}%", flush=True)
json.dump({"mae_p50": pq(50), "mae_p90": pq(90), "mae_p99": pq(99),
           "mae_worst": worst, "n_trades": int(len(res)),
           "deep15_count": n_deep, "deep15_avg_final": deep_avg,
           "deep15_pos_pct": deep_pos,
           "stop10": {"fired": n10, "avg": a10},
           "stop15": {"fired": n15, "avg": a15},
           "stop20": {"fired": n20, "avg": a20},
           "base_avg": base_avg, "folds": folds},
          open("/kaggle/working/report.json", "w"), indent=1)
res[res["mae"] <= -0.10].to_csv("/kaggle/working/deep_trades.csv", index=False)
print("измерение завершено", flush=True)