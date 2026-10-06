"""РЕЖИМНЫЙ ФИЛЬТР (предрегистрация 05.10): 1ч-сигнал p>0.65 торгуем
ТОЛЬКО в режиме «тренд вверх + рынок в основном растёт».

ГИПОТЕЗА ИЗ НАШИХ ЖЕ ИЗМЕРЕНИЙ: плюс концентрируется в бурных бычьих
кварталах (2025Q4). Не воюем — входим только при: BTC выше своей
средней за 30 дней (дневной тренд) И доля монет с положительным
20-барным моментумом > 50%.

СТРОГИЕ КРИТЕРИИ (усилены из-за множественных сравнений — до этого дня
мы уже много раз смотрели на эти данные): avg >= +0.15% И плюсовых
кварталов >= 6/8 И нуль-перцентиль >= 97. Порог один: 0.65. Фолды —
те же 8 кварталов. Рядом для справки — тот же сигнал без фильтра.
"""
import glob, json, time, zipfile, io, zlib
import numpy as np, pandas as pd
print("СТАРТ: импорт lightgbm...", flush=True)
import lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 3600_000
SLIP, FEE2 = 0.0005, 0.001
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
print("монет:", len(members), flush=True)

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

btc_raw = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
btc_raw["open_time"] = btc_raw["open_time"].astype(np.int64)
btc_raw.index = pd.to_datetime(btc_raw["open_time"], unit="ms")
btc1h = btc_raw["close"].astype("float32").resample("1h").last().dropna()
# режим BTC: дневная свеча выше своей средней за 30 дней
btc1d = btc_raw["close"].astype("float32").resample("1D").last().dropna()
btc_regime = (btc1d > btc1d.rolling(30, min_periods=30).mean())
reg_map = pd.Series(btc_regime.values,
                    index=btc1d.index + pd.Timedelta(days=1))

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
    df = df.sort_index()
    rd = resample(df, "1h")
    X = make_feats(rd, btc1h, sym)
    close = rd["close"].values
    fut = np.full(len(rd), np.nan, dtype="float32")
    fut[:-H] = close[H:]
    X["y"] = np.where(np.isnan(fut), np.nan,
                      (fut / close > 1).astype("float32"))
    X["fwd"] = (fut / close - 1).astype("float32")
    X["close"] = close
    X["ot"] = rd["open_time"].values.astype(np.int64)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=[c for c in X.columns
                                   if c not in ("fwd", "close")]))
    if i % 150 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
print("строк 1ч:", len(data), flush=True)
# широта рынка по часам: доля монет с ret_20 > 0 в этот час
up20 = (data["ret_20"] > 0).astype("float32")
data["breadth"] = up20.groupby(data["ot"]).transform("mean") \
    .astype("float32")
del up20
dt = pd.to_datetime(data["ot"], unit="ms")
day_key = dt.dt.floor("D")
data["regime"] = day_key.map(reg_map).astype("float32").fillna(0.0)
q = pd.PeriodIndex(dt, freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
print(f"строк {len(data)}, тест {test_q} | {time.time()-t0:.0f}с",
      flush=True)

FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]

def gate_trades(d, regime_only):
    dl = d[(d["p"] > GATE) & ((d["regime"] > 0) & (d["breadth"] > 0.5)
                              if regime_only else True)] \
        .sort_values(["sym_id", "ot"])
    kept, last = [], {}
    for row in dl.itertuples():
        if row.ot - last.get(row.sym_id, -10**18) >= COOLDOWN_MS:
            kept.append((row.fwd, row.quarter))
            last[row.sym_id] = row.ot
    if not kept:
        return pd.DataFrame(columns=["pnl", "quarter"])
    T = pd.DataFrame(kept, columns=["fwd", "quarter"])
    T["pnl"] = T["fwd"].astype("float64") - FEE2
    return T[["pnl", "quarter"]]

def null_pct(te, n_real, seeds=50):
    rng = np.random.default_rng(7)
    sub = te[["close", "fwd"]].dropna()
    if n_real < 20 or len(sub) < n_real:
        return None
    out = []
    for _ in range(seeds):
        s = sub.sample(n=n_real, random_state=rng.integers(1 << 30))
        out.append((s["fwd"].values.astype("float64") - FEE2).mean() * 100)
    return out

acc = {"off": [], "on": []}
last2 = []
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 2000 or len(trd) < 100_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(trd[FEATS], trd["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    acc["off"].append(gate_trades(d, False))
    Ton = gate_trades(d, True)
    acc["on"].append(Ton)
    if k >= 6 and len(Ton):
        nl = null_pct(ted, len(Ton))
        if nl:
            last2.append(float(np.mean(np.array(nl)
                                       < Ton["pnl"].mean() * 100) * 100))
    print(f"фолд {qq} готов | {time.time()-t0:.0f}с", flush=True)

report = {"pre_reg": "СТРОГО: avg>=+0.15%, 6/8 кв, нуль>=97 (множ.сравн)",
          "quarters": test_q, "scenarios": {}}
for name in ("off", "on"):
    T = pd.concat(acc[name], ignore_index=True)
    avg = float(T["pnl"].mean() * 100)
    by_q = {k: round(float(v) * 100, 3) for k, v in
            T.groupby("quarter")["pnl"].mean().items()} if len(T) else {}
    pos_q = sum(1 for v in by_q.values() if v > 0)
    nl_pct = int(np.mean(last2)) if (name == "on" and last2) else None
    strict = bool(avg >= 0.15 and pos_q >= 6 and (nl_pct or 0) >= 97)
    report["scenarios"][name] = {
        "n": int(len(T)), "avg": round(avg, 4), "pos_quarters": pos_q,
        "by_quarter": by_q, "null_pct": nl_pct, "strict_pass": strict}
    print(f"фильтр {name}: {len(T)} сделок, avg {avg:+.4f}%, "
          f"плюс-кв {pos_q}/8, нуль {nl_pct}% -> "
          f"{'ПРОШЁЛ СТРОГО' if strict else 'нет'}", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("РЕЖИМНЫЙ ФИЛЬТР завершён", flush=True)


