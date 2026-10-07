"""ШИРОКОЕ ПЕРЕОБУЧЕНИЕ МОДЕЛИ-БУРИ (v3, 06.10): тот же вопрос, что у
боевой модели (метка ±0.5 ATR, обе стороны, gate 0.65). Экзамен:
8 кварталов linspace walk-forward, порог 0.65, мейкер-вход с тестом
наполнения, стоп -15%. Критерии деплоя: avg >= +0.10% И плюсовых
кварталов >= 5/8 И нуль >= 95. Сдало -> финальная модель на всей
панели -> ml5h.txt + мета (gate 0.65, label storm). Не сдало -> старая
модель остаётся. Вердикт ОБЯЗАТЕЛЬНО в meta и report.json (читает
синхронизатор)."""
import glob, json, time, zipfile, io, zlib, datetime
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000
SLIP, TAKER_FEE, MAKER_FEE = 0.0005, 0.0005, 0.0002
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
if not zips:
    json.dump({"verdict": False, "error": "panel.zip not found"},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
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
    X["fwd"] = fwd.astype("float32")
    X["close"] = close
    X["low"] = df["low"].values
    X["high"] = df["high"].values
    X["ot"] = df["open_time"].values.astype(np.int64)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=[c for c in X.columns
                                   if c not in ("fwd", "close", "low",
                                                "high")]))
    if i % 150 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
print(f"строк {len(data)}, тест {test_q} | {time.time()-t0:.0f}с",
      flush=True)

def collect(d):
    pools = {int(fi): gg.sort_values("ot") for fi, gg in d.groupby("sym_id")}
    rows = []
    for side in (1, -1):
        sel = d[(d["p"] > GATE) if side == 1 else (d["p"] < 1 - GATE)] \
            .sort_values(["sym_id", "ot"])
        last = {}
        for row in sel.itertuples():
            if row.ot - last.get(row.sym_id, -10**18) < COOLDOWN_MS:
                continue
            last[row.sym_id] = row.ot
            pool = pools.get(int(row.sym_id))
            if pool is None:
                continue
            ots = pool["ot"].values
            j = int(np.searchsorted(ots, row.ot))
            if j >= len(ots) or ots[j] != row.ot or j + H >= len(pool) \
                    or j + 1 >= len(pool):
                continue
            exit_close = pool["close"].values[j + H]
            if np.isnan(exit_close):
                continue
            c = row.close
            seg_lo = pool["low"].values[j + 1: j + 1 + H]
            seg_hi = pool["high"].values[j + 1: j + 1 + H]
            filled = (seg_lo[0] <= c) if side == 1 else (seg_hi[0] >= c)
            hit = (seg_lo.min() <= c * 0.85) if side == 1 \
                else (seg_hi.max() >= c * 1.15)
            exit_px = (c * 0.85 * (1 - SLIP)) if (hit and side == 1) else (
                (c * 1.15 * (1 + SLIP)) if (hit and side == -1)
                else exit_close * (1 - side * SLIP))
            if side == 1:
                pnl = exit_px / c - 1 - MAKER_FEE - TAKER_FEE - SLIP
            else:
                pnl = (c - exit_px) / c - MAKER_FEE - TAKER_FEE - SLIP
            if not filled:
                continue
            rows.append({"pnl": pnl, "quarter": row.quarter, "side": side})
    return pd.DataFrame(rows)

def null_pct(te, n_real, seeds=50):
    rng = np.random.default_rng(7)
    sub = te[["fwd"]].dropna()
    if n_real < 20 or len(sub) < n_real:
        return None
    out = []
    for _ in range(seeds):
        s = sub.sample(n=n_real, random_state=rng.integers(1 << 30))
        out.append((s["fwd"].values.astype("float64")
                    - 2 * TAKER_FEE).mean() * 100)
    return out

all_T = []
nulls = []
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 5000 or len(trd) < 200_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(trd[FEATS], trd["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    T = collect(d)
    T["quarter"] = qq
    all_T.append(T)
    if k >= 6 and len(T):
        nl = null_pct(ted, len(T))
        if nl:
            nulls.append(float(np.mean(np.array(nl)
                                       < T["pnl"].mean() * 100) * 100))
    print(f"фолд {qq}: сделок {len(T)} | {time.time()-t0:.0f}с", flush=True)

T = pd.concat(all_T, ignore_index=True)
avg = float(T["pnl"].mean() * 100)
by_q = {kk: round(v * 100, 3) for kk, v in
        T.groupby("quarter")["pnl"].mean().items()}
pos_q = sum(1 for v in by_q.values() if v > 0)
nl_pct = int(np.mean(nulls)) if nulls else 0
verdict = bool(avg >= 0.10 and pos_q >= 5 and nl_pct >= 95)
print(f"ИТОГ: {len(T)} сделок, avg {avg:+.4f}%, плюс-кв {pos_q}/8, "
      f"нуль {nl_pct}% -> {'ДЕПЛОИМ' if verdict else 'НЕ СДАЛО'}",
      flush=True)

syms = sorted({n.replace("_30m.csv", "") for n in members})
if verdict:
    mf = lgb.LGBMClassifier(**PARAMS)
    dat = data.dropna(subset=["y"])
    mf.fit(dat[FEATS], dat["y"], categorical_feature=["sym_id"])
    mf.booster_.save_model("/kaggle/working/ml5h.txt")
    json.dump({"trained_at": datetime.datetime.now(
        datetime.timezone.utc).isoformat(),
        "features": FEATS, "symbols": syms, "gate": GATE,
        "hold_bars": H, "bar_minutes": 30,
        "label": "storm_05atr_both_sides", "train_rows": int(len(dat)),
        "universe": "wide", "verdict": True},
        open("/kaggle/working/ml5h_meta.json", "w"), indent=1)
    print("модель сохранена", flush=True)
json.dump({"verdict": verdict, "avg": round(avg, 4),
           "pos_quarters": pos_q, "null_pct": nl_pct,
           "by_quarter": by_q},
          open("/kaggle/working/report.json", "w"), indent=1)
