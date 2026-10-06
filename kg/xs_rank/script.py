"""КРОСС-РЫНОЧНЫЕ ПРИЗНАКИ (предрегистрация 05.10): место монеты среди
соседей + широта рынка. Ось информации, которую мы не мерили ни разу.

A/B: база 24 признака против 27 (+xs_rank20, xs_rank60, breadth20).
Экзамен: 8 кварталов linspace, пороги 0.55/0.60/0.65 на общих
предсказаниях, тейкер 20бп круг. B ЖИВ: улучшает A на >= +0.03% И
avg >= +0.10% И плюс-кв >= 5/8. Нуль-тест на двух последних фолдах.
"""
import glob, json, time, zipfile, io, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
H = 10
GATES = [0.55, 0.60, 0.65]
COOLDOWN_MS = 10 * 1800_000
SLIP, FEE2 = 0.0005, 0.001
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
    X["ret_60"] = close.pct_change(60).astype("float32")
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
    return X

frames = []
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
    X = make_feats(df, btc, sym)
    close = df["close"].values
    fut = np.full(len(df), np.nan, dtype="float32")
    fut[:-H] = close[H:]
    X["y"] = np.where(np.isnan(fut), np.nan,
                      (fut / close > 1).astype("float32"))
    X["fwd"] = (fut / close - 1).astype("float32")
    X["ot"] = df["open_time"].values.astype(np.int64)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=[c for c in X.columns
                                   if c not in ("fwd",)]))
    if i % 150 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
print("строк:", len(data), "| кросс-ранги...", flush=True)
# кросс-рыночные признаки: считаются ПО МЕТКЕ ВРЕМЕНИ на всех монетах
up20 = (data["ret_20"] > 0).astype("float32")
data["xs_rank20"] = data.groupby("ot")["ret_20"].rank(pct=True) \
    .astype("float32")
data["xs_rank60"] = data.groupby("ot")["ret_60"].rank(pct=True) \
    .astype("float32")
data["breadth20"] = up20.groupby(data["ot"]).transform("mean") \
    .astype("float32")
del up20
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
print(f"строк {len(data)}, тест {test_q} | {time.time()-t0:.0f}с",
      flush=True)

BASE = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
        "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
        "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
        "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
        "hour", "dow", "sym_id"]
XS = BASE + ["xs_rank20", "xs_rank60", "breadth20"]

def gate_trades(d, gate):
    dl = d[d["p"] > gate].sort_values(["sym_id", "ot"])
    kept, last = [], {}
    for row in dl.itertuples():
        if row.ot - last.get(row.sym_id, -10**18) >= COOLDOWN_MS:
            kept.append((row.fwd, row.quarter))
            last[row.sym_id] = row.ot
    if not kept:
        return pd.DataFrame(columns=["fwd", "quarter"])
    T = pd.DataFrame(kept, columns=["fwd", "quarter"])
    T["pnl"] = T["fwd"].astype("float64") - FEE2
    return T[["pnl", "quarter"]]

def null_pct(te, n_real, seeds=50):
    rng = np.random.default_rng(7)
    sub = te[["fwd"]].dropna()
    if n_real < 20 or len(sub) < n_real:
        return None
    out = []
    for _ in range(seeds):
        s = sub.sample(n=n_real, random_state=rng.integers(1 << 30))
        out.append(((s["fwd"].values.astype("float64")) - FEE2).mean() * 100)
    return out

acc = {c: {g: [] for g in GATES} for c in ("A", "B")}
last2 = {g: [] for g in GATES}
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 5000 or len(trd) < 200_000:
        continue
    for cname, feats in (("A", BASE), ("B", XS)):
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(trd[feats], trd["y"], categorical_feature=["sym_id"])
        d = ted.assign(p=m.predict_proba(ted[feats])[:, 1])
        for g in GATES:
            T = gate_trades(d, g)
            acc[cname][g].append(T)
    if k >= 6:
        m = lgb.LGBMClassifier(**PARAMS)
        trA = data[data["quarter"] < qq]
        m.fit(trA[BASE], trA["y"], categorical_feature=["sym_id"])
        d = ted.assign(p=m.predict_proba(ted[BASE])[:, 1])
        for g in GATES:
            T = gate_trades(d, g)
            Tq = T[T["quarter"] == qq]
            nl = null_pct(ted, len(Tq))
            if nl and len(Tq):
                last2[g].append(float(np.mean(
                    np.array(nl) < Tq["pnl"].mean() * 100) * 100))
    print(f"фолд {qq} готов | {time.time()-t0:.0f}с", flush=True)

report = {"pre_reg": "B жив: улучшает A на >=0.03%, avg>=0.10%, 5/8",
          "quarters": test_q, "gates": {}}
for g in GATES:
    row = {}
    for cname in ("A", "B"):
        T = pd.concat(acc[cname][g], ignore_index=True)
        avg = float(T["pnl"].mean() * 100)
        by_q = {k: round(v * 100, 3) for k, v in
                T.groupby("quarter")["pnl"].mean().items()}
        pos_q = sum(1 for v in by_q.values() if v > 0)
        row[cname] = {"n": int(len(T)), "avg": round(avg, 4),
                      "pos_quarters": pos_q}
    nl_pct = int(np.mean(last2[g])) if last2[g] else None
    row["null_pct"] = nl_pct
    improve = row["B"]["avg"] - row["A"]["avg"]
    alive = bool(improve >= 0.03 and row["B"]["avg"] >= 0.10
                 and row["B"]["pos_quarters"] >= 5 and (nl_pct or 0) >= 95)
    row["improve"] = round(improve, 4)
    row["alive"] = alive
    report["gates"][str(g)] = row
    print(f"gate {g}: A {row['A']['avg']:+.4f}% | B {row['B']['avg']:+.4f}% "
          f"(улучшение {improve:+.4f}) | нуль {nl_pct}% -> "
          f"{'ЖИВ' if alive else 'нет'}", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("КРОСС-РЫНОК завершён", flush=True)
