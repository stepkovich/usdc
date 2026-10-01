"""ЭКРАН ПОРОГОВ УВЕРЕННОСТИ (предрегистрация 30.09, до запуска).

ПОВОД: вскрытие (usdc-ml5h-ablation) на сплите 2025Q1+ показало, что
80% сигналов при gate 0.55 — плоский шум (−0.20%/сделку), а топ-квинтиль
уверенности +0.037%/сделку даже при тейкер-издержках 0.2%/круг.
Гипотеза: более строгий порог отделяет сигнал от шума.

ДИЗАЙН: та же панель, признаки, ярлыки, параметры и 8 тестовых
кварталов (linspace по всей истории), что у боевого широкого ядра v3:
медведь 2021-22 включён. На каждом фолде ОДНО обучение и ОДНО
предсказание; пороги 0.55 / 0.60 / 0.65 / 0.70 оцениваются на ОДНИХ И
ТЕХ ЖЕ предсказаниях (без подгонки порога по результату). Лонг p>gate,
cooldown 10 баров, pnl = вход close*(1+slip), выход close[t+10]*(1-slip),
− комиссии taker x2. Стопы не моделируем (задевают 0.27% сделок — на
сравнение порогов не влияют).

ЗАЯВЛЕННЫЕ КРИТЕРИИ ЭКРАНА (как у боевого экзамена): порог жив, если
avg >= +0.10% И плюсовых кварталов >= 5/8 И нуль-перцентиль >= 95
(50 случайных портфелей той же плотности на последних двух фолдах).
Экран ничего не деплоит: победитель идёт в полный боевой экзамен
с реальной механикой стопов и шортов.

ВТОРИЧНОЕ (для контекста, не критерий): навык топ-минус-боттом квинтиль.
"""
import glob, json, time, zipfile, io, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
H = 10
COOLDOWN_MS = 10 * 1800_000
SLIP, FEE2 = 0.0005, 0.001
GATES = [0.55, 0.60, 0.65, 0.70]
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")

def sym_id(sym): return np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
if not zips:
    json.dump({"error": "panel.zip not found"},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
print("монет в панели:", len(members), flush=True)
btc_pre = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
btc_pre["open_time"] = btc_pre["open_time"].astype(np.int64)
btc_pre.index = pd.to_datetime(btc_pre["open_time"], unit="ms")
btc = btc_pre["close"].astype("float32")

parts = []
for i, name in enumerate(members, 1):
    sym = name.replace("_30m.csv", "")
    df = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["open_time", "high", "low", "close",
                              "volume", "quote_volume", "taker_buy_base"])
    for c in df.columns:
        if c != "open_time":          # 13-значные метки не влезают в float32
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
    X["sym_id"] = sym_id(sym)
    fut_close = np.full(len(df), np.nan, dtype="float32")
    fut_close[:-H] = close.values[H:]
    X["y"] = np.where(np.isnan(fut_close), np.nan,
                      (fut_close / close.values > 1).astype("float32"))
    X["fwd"] = (fut_close / close.values - 1).astype("float32")
    X["close"] = close.values
    X["ot"] = df["open_time"].values
    X = X.replace([np.inf, -np.inf], np.nan)
    parts.append(X.dropna(subset=[c for c in X.columns
                                  if c not in ("fwd", "close")]))
    if i % 100 == 0:
        print(f"признаки {i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(parts, ignore_index=True)
del parts
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
feats = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
print(f"строк {len(data)}, тестовые кварталы: {test_q}", flush=True)

def gate_trades(d, gate):
    """Сигналы p>gate с cooldown; pnl по fwd из той же строки."""
    dl = d[d["p"] > gate].sort_values(["sym_id", "ot"])
    kept, last = [], {}
    for row in dl.itertuples():
        if row.ot - last.get(row.sym_id, -10**18) >= COOLDOWN_MS:
            kept.append((row.ot, row.sym_id, row.close, row.p,
                         row.quarter, row.fwd))
            last[row.sym_id] = row.ot
    if not kept:
        return pd.DataFrame(columns=["pnl", "p", "quarter"])
    T = pd.DataFrame(kept, columns=["ot", "sym_id", "close", "p",
                                    "quarter", "fwd"])
    T = T[~np.isnan(T["fwd"].values)].reset_index(drop=True)
    entry = T["close"].values * (1 + SLIP)
    exit_ = T["close"].values * (1 + T["fwd"].values) * (1 - SLIP)
    T["pnl"] = exit_ / entry - 1 - FEE2
    return T

def null_pct(te, n_real, seeds=50):
    """Доля случайных портфелей той же плотности, хуже реального."""
    rng = np.random.default_rng(7)
    out = []
    sub = te[["close", "fwd"]].dropna()
    for _ in range(seeds):
        s = sub.sample(n=min(n_real, len(sub)), random_state=rng.integers(1 << 30))
        entry = s["close"].values * (1 + SLIP)
        exit_ = s["close"].values * (1 + s["fwd"].values) * (1 - SLIP)
        out.append((exit_ / entry - 1 - FEE2).mean() * 100)
    return out

acc = {g: [] for g in GATES}
null_evals = {g: [] for g in GATES}
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 5000 or len(trd) < 200_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(trd[feats], trd["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[feats])[:, 1])
    line = f"фолд {qq}:"
    for g in GATES:
        T = gate_trades(d, g)
        acc[g].append(T)
        avg = T["pnl"].mean() * 100 if len(T) else float("nan")
        line += f" | g{g}: {len(T)} шт {avg:+.3f}%"
        if k >= 6 and len(T):          # нуль-тест на двух последних фолдах
            null_evals[g].append((avg, len(T), ted))
    print(line + f" | {time.time()-t0:.0f}с", flush=True)

report = {"pre_reg": "пороги 0.55/0.60/0.65/0.70 на общих предсказаниях; "
                     "критерий: avg>=+0.10%, плюс-кварталов>=5/8, нуль>=95",
          "gates": {}, "folds": test_q}
all_out = {}
for g in GATES:
    T = pd.concat(acc[g], ignore_index=True) if acc[g] else pd.DataFrame()
    if not len(T):
        report["gates"][str(g)] = {"n": 0}
        continue
    avg = float(T["pnl"].mean() * 100)
    by_q = {k: round(v * 100, 3)
            for k, v in T.groupby("quarter")["pnl"].mean().items()}
    pos_q = sum(1 for v in by_q.values() if v > 0)
    T2 = T.sort_values("p")
    n5 = max(1, len(T2) // 5)
    skill = float(T2["pnl"].tail(n5).mean() - T2["pnl"].head(n5).mean()) * 100
    nl_pct = None
    if null_evals[g]:
        pcts = []
        for real_avg, n_real, ted in null_evals[g]:
            nl = null_pct(ted, n_real)
            pcts.append(float(np.mean(np.array(nl) < real_avg) * 100))
        nl_pct = int(np.mean(pcts))
    alive = bool(avg >= 0.10 and pos_q >= 5 and (nl_pct or 0) >= 95)
    report["gates"][str(g)] = {
        "n": int(len(T)), "avg": round(avg, 4),
        "wr": round(float((T["pnl"] > 0).mean() * 100), 1),
        "pos_quarters": pos_q, "by_quarter": by_q,
        "skill": round(skill, 3), "null_pct": nl_pct, "alive": alive}
    all_out[g] = T
    print(f"ИТОГ gate {g}: {len(T)} сделок, avg {avg:+.4f}%, "
          f"плюс-кв. {pos_q}/8, навык {skill:+.3f}, нуль {nl_pct}% "
          f"-> {'ЖИВ' if alive else 'нет'}", flush=True)
    T[["ot", "sym_id", "p", "pnl", "quarter"]].to_csv(
        f"/kaggle/working/trades_g{g}.csv", index=False)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("экран порогов завершён", flush=True)
