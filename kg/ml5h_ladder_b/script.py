"""ЛЕСТНИЦА ТАЙМФРЕЙМОВ ЧАСТЬ B (предрегистрация 30.09, до запуска): 4ч и 1д.

План Б по владельцу: лестница 30м -> 1ч -> 4ч -> 1д. Часть B (это ядро,
параллельно usdc-ml5h-ladder-a):
  B1  4ч, горизонт 10 баров (40 часов)
  B2  1д, горизонт 10 баров (10 дней)
  B3  1д, горизонт 3 бара (3 дня) — исходная идея плана Б «1-3 дня»

ДИЗАЙН и КРИТЕРИИ — идентичны части A: те же 24 формулы в барах,
BTCUSDT исключён, лонг p>gate, cooldown 10 баров, издержки taker x2,
8 кварталов (linspace) с переобучением, пороги 0.55/0.60/0.65 на общих
предсказаниях. Конфиг ЖИВ: avg >= +0.10%, плюсовых кварталов >= 5/8,
нуль-перцентиль >= 95. Экран ничего не деплоит.
"""
import glob, json, time, zipfile, io, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
COOLDOWN_BARS = 10
SLIP, FEE2 = 0.0005, 0.001
GATES = [0.55, 0.60, 0.65]
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
RULE_MS = {"4h": 4 * 3600_000, "1D": 24 * 3600_000}
CONFIGS = [("4h_h10", "4h", 10), ("1d_h10", "1D", 10), ("1d_h3", "1D", 3)]

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
if not zips:
    json.dump({"error": "panel.zip not found"},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
print("монет в панели:", len(members), flush=True)

def resample(df, rule):
    return df[["open_time", "high", "low", "close", "volume",
               "quote_volume", "taker_buy_base"]] \
        .resample(rule).agg({"open_time": "min", "high": "max",
                             "low": "min", "close": "last",
                             "volume": "sum", "quote_volume": "sum",
                             "taker_buy_base": "sum"}).dropna()

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
    X["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).astype("float32")
    X["skew"] = rets.rolling(20, min_periods=10).skew().astype("float32")
    sma50, sma20 = close.rolling(50).mean(), close.rolling(20).mean()
    X["sma_cross"] = ((close > sma50).astype("float32")
                      - (close > sma20).astype("float32"))
    if sym != "BTCUSDT" and btc_close is not None:
        b = btc_close.reindex(df.index).ffill()
        X["btc_ret_10"] = b.pct_change(10).astype("float32")
        X["btc_ret_20"] = b.pct_change(20).astype("float32")
        bret = b.pct_change()
        X["btc_corr"] = rets.rolling(50, min_periods=20).corr(bret).astype("float32")
        X["rel_ret_20"] = (rets.rolling(20).sum() - bret.rolling(20).sum()).astype("float32")
        X["rel_ret_60"] = (rets.rolling(60).sum() - bret.rolling(60).sum()).astype("float32")
        cov = rets.rolling(480).cov(bret); var = bret.rolling(480).var()
        X["beta_20"] = (cov / var.replace(0, np.nan)).astype("float32")
        X["corr_20"] = rets.rolling(480, min_periods=100).corr(bret).astype("float32")
    X["h4_momentum"] = close.pct_change(48).astype("float32")
    X["hour"] = df.index.hour.astype("float32")
    X["dow"] = df.index.dayofweek.astype("float32")
    X["sym_id"] = np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)
    return X

btc_raw = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
btc_raw["open_time"] = btc_raw["open_time"].astype(np.int64)
btc_raw.index = pd.to_datetime(btc_raw["open_time"], unit="ms")
btc30 = btc_raw["close"].astype("float32")
BTC = {r: btc30.resample(r).last().dropna() for r in RULE_MS}

frames = {c: [] for c, _, _ in CONFIGS}
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
    df = df.sort_index()
    for rule in RULE_MS:
        rd = resample(df, rule)
        Xb = make_feats(rd, BTC[rule], sym)
        for cname, crule, H in CONFIGS:
            if crule != rule:
                continue
            X = Xb.copy()
            close = rd["close"].values
            fut = np.full(len(rd), np.nan, dtype="float32")
            fut[:-H] = close[H:]
            X["y"] = np.where(np.isnan(fut), np.nan,
                              (fut / close > 1).astype("float32"))
            X["fwd"] = (fut / close - 1).astype("float32")
            X["close"] = close
            X["ot"] = rd["open_time"].values.astype(np.int64)
            X["fi"] = np.uint16(i)
            X = X.replace([np.inf, -np.inf], np.nan)
            frames[cname].append(X.dropna(
                subset=[c for c in X.columns if c not in ("fwd", "close")]))
    if i % 100 == 0:
        print(f"признаки {i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = {}
for cname, _, _ in CONFIGS:
    data[cname] = pd.concat(frames[cname], ignore_index=True)
    q = pd.PeriodIndex(pd.to_datetime(data[cname]["ot"], unit="ms"), freq="Q")
    data[cname]["quarter"] = q.astype(str)
    print(f"{cname}: {len(data[cname])} строк | {time.time()-t0:.0f}с", flush=True)
del frames

FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]

def gate_trades(d, gate, cd_ms):
    dl = d[d["p"] > gate].sort_values(["sym_id", "ot"])
    kept, last = [], {}
    for row in dl.itertuples():
        if row.ot - last.get(row.sym_id, -10**18) >= cd_ms:
            kept.append((row.close, row.p, row.quarter, row.fwd))
            last[row.sym_id] = row.ot
    if not kept:
        return pd.DataFrame(columns=["pnl", "p", "quarter"])
    T = pd.DataFrame(kept, columns=["close", "p", "quarter", "fwd"])
    T = T[~np.isnan(T["fwd"].values)].reset_index(drop=True)
    entry = T["close"].values * (1 + SLIP)
    exit_ = T["close"].values * (1 + T["fwd"].values) * (1 - SLIP)
    T["pnl"] = exit_ / entry - 1 - FEE2
    return T

def null_pct(te, n_real, seeds=30):
    rng = np.random.default_rng(7)
    sub = te[["close", "fwd"]].dropna()
    if n_real < 20 or len(sub) < n_real:
        return None
    out = []
    for _ in range(seeds):
        s = sub.sample(n=n_real, random_state=rng.integers(1 << 30))
        entry = s["close"].values * (1 + SLIP)
        exit_ = s["close"].values * (1 + s["fwd"].values) * (1 - SLIP)
        out.append((exit_ / entry - 1 - FEE2).mean() * 100)
    return out

acc = {c: {g: [] for g in GATES} for c, _, _ in CONFIGS}
last2 = {c: {g: [] for g in GATES} for c, _, _ in CONFIGS}
quarters = sorted(data["4h_h10"]["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
for k, qq in enumerate(test_q):
    for cname, rule, H in CONFIGS:
        dset = data[cname]
        trd = dset[dset["quarter"] < qq]
        ted = dset[dset["quarter"] == qq]
        if len(ted) < 500 or len(trd) < 50_000:
            continue
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(trd[FEATS], trd["y"], categorical_feature=["sym_id"])
        d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
        cd_ms = COOLDOWN_BARS * RULE_MS[rule]
        for g in GATES:
            T = gate_trades(d, g, cd_ms)
            acc[cname][g].append(T)
            if k >= 6:
                last2[cname][g].append((T, ted))
    print(f"фолд {qq} готов | {time.time()-t0:.0f}с", flush=True)

report = {"pre_reg": "конфиг жив: любой порог с avg>=+0.10%, 5/8 кварталов, "
                     "нуль>=95; часть B лестницы (4ч/1д)",
          "folds": test_q, "configs": {}}
for cname, _, _ in CONFIGS:
    row = {"gates": {}}
    for g in GATES:
        T = pd.concat(acc[cname][g], ignore_index=True) if acc[cname][g] \
            else pd.DataFrame()
        if not len(T):
            row["gates"][str(g)] = {"n": 0}
            continue
        avg = float(T["pnl"].mean() * 100)
        by_q = {kk: round(v * 100, 3)
                for kk, v in T.groupby("quarter")["pnl"].mean().items()}
        pos_q = sum(1 for v in by_q.values() if v > 0)
        pcts = []
        for (tT, ted) in last2[cname][g]:
            if not len(tT):
                continue
            nl = null_pct(ted, len(tT))
            if nl:
                pcts.append(float(np.mean(np.array(nl) < tT["pnl"].mean() * 100) * 100))
        nl_pct = int(np.mean(pcts)) if pcts else None
        alive = bool(avg >= 0.10 and pos_q >= 5 and (nl_pct or 0) >= 95)
        row["gates"][str(g)] = {"n": int(len(T)), "avg": round(avg, 4),
                                "pos_quarters": pos_q, "by_quarter": by_q,
                                "null_pct": nl_pct, "alive": alive}
        print(f"[{cname}] gate {g}: {len(T)}шт avg {avg:+.4f}% плюс-кв "
              f"{pos_q}/8 нуль {nl_pct} -> {'ЖИВ' if alive else 'нет'}",
              flush=True)
    row["alive_any_gate"] = any(v.get("alive") for v in row["gates"].values())
    report["configs"][cname] = row
    json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("ЧАСТЬ B завершена:", {c: v["alive_any_gate"]
                             for c, v in report["configs"].items()}, flush=True)
