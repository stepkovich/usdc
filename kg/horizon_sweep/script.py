"""РАУНД 1 — ГОРИЗОНТ ВЫХОДА у боевой связки (бой не трогаем).

Панель 30м (~527 монет, ~5 лет). Один и тот же сигнал (p>0.65 / p<0.35,
кулдаун 5ч, заполнение первым баром) доводим до выхода НЕ только через
10 баров (5 часов, как сейчас), а через каждый из 17 горизонтов
(1..24 бара = 30 минут .. 12 часов). Родная симуляция: аварийный стоп
-15%, издержки 12 бп, мета в замере НЕ участвует.

Два замера:
  ФОЛДЫ  — 8 экзаменационных кварталов как в месячном экзамене
           (модель учится ТОЛЬКО на данных до своего квартала):
           что даёт РЕЦЕПТ (24 признака, те же параметры) на горизонтах.
  ЖИВАЯ  — ml5h.txt с сервера (обучена 06.10) на честном окне 07-26.10,
           две руки признака btc_corr:
             live_B = по доходностям (починка 8b16f583 — так в бою сейчас),
             live_A = по ценам (так было в бою до 27.10).

Выход: horizon_sweep.json + trades.csv.gz + печатная таблица.
"""
import io, json, os, time, zipfile, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000          # 5ч между сигналами по монете (родное)
SLIP, TAKER_FEE, MAKER_FEE = 0.0005, 0.0005, 0.0002
STOP = 0.15
HOR = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24]
H_LABEL = 10                          # родная метка: t+10 баров, порог 0.5*ATR
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")


def log(m):
    print(f"{m} | {time.time()-t0:.0f}с", flush=True)


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


# ---------------- панель ----------------
zpath = None
for root, _d, fs in os.walk("/kaggle/input"):
    if "panel.zip" in fs:
        zpath = os.path.join(root, "panel.zip")
        break
assert zpath, "panel.zip не найден"
zf = zipfile.ZipFile(zpath)
members = sorted(n for n in zf.namelist() if n.endswith("_30m.csv"))
log(f"панель {os.path.basename(os.path.dirname(zpath))}: {len(members)} монет")

btc_pre = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
btc_pre["open_time"] = btc_pre["open_time"].astype(np.int64)
btc_pre.index = pd.to_datetime(btc_pre["open_time"], unit="ms")
btc = btc_pre["close"].astype("float32")
del btc_pre

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
    # рука А: btc_corr по ЦЕНАМ (так было в бою до починки 27.10)
    X["btc_corr_px"] = (df["close"].rolling(50, min_periods=20)
                        .corr(btc.reindex(df.index).ffill())
                        ).astype("float32")
    close = df["close"].values
    atr = (tr.rolling(14).mean() / close).values.astype("float64")
    fut = np.full(len(df), np.nan, dtype="float32")
    fut[:-H_LABEL] = close[H_LABEL:]
    fwd = (fut / close - 1).astype("float64")
    thr = 0.5 * np.nan_to_num(atr, nan=np.nan)
    y = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                 np.where(fwd > thr, 1.0,
                          np.where(fwd < -thr, 0.0, np.nan)))
    X["y"] = y.astype("float32")
    X["close"] = close
    X["low"] = df["low"].values
    X["high"] = df["high"].values
    X["ot"] = df["open_time"].values.astype(np.int64)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=[c for c in X.columns
                                   if c not in ("close", "low", "high")]))
    if i % 150 == 0:
        log(f"{i}/{len(members)}")

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
FEATS_A = [f if f != "btc_corr" else "btc_corr_px" for f in FEATS]
log(f"строк {len(data)}, экзамен-кварталы: {test_q}")


# ---------------- сигналы и симуляция ----------------
def signals(d, p):
    """Обе стороны, гейт 0.65, кулдаун 5ч/монета -> [(sym_id, ot, side)]."""
    d = d.assign(p=p)
    picked = []
    for side in (1, -1):
        sel = d[(d["p"] > GATE) if side == 1 else (d["p"] < 1 - GATE)] \
            .sort_values(["sym_id", "ot"])
        last = {}
        for row in sel.itertuples():
            if row.ot - last.get(int(row.sym_id), -10**18) < COOLDOWN_MS:
                continue
            last[int(row.sym_id)] = row.ot
            picked.append((int(row.sym_id), int(row.ot), side))
    return picked


def eval_horizons(d, picked, rows_out, tag):
    """Родная симуляция, но выход проверяется на КАЖДОМ горизонте HOR.
    Заполнение входа первым баром; стоп -15% внутри окна; иначе выход
    закрытием бара j+h; издержки 12 бп как в родной collect()."""
    pools = {int(fi): (g["close"].values, g["low"].values,
                       g["high"].values, g["ot"].values)
             for fi, g in d.groupby("sym_id")}
    for sym_id, ot, side in picked:
        pool = pools.get(sym_id)
        if pool is None:
            continue
        pc, pl, ph, pots = pool
        j = int(np.searchsorted(pots, ot))
        if j >= len(pots) or pots[j] != ot or j + 1 >= len(pc):
            continue
        c = pc[j]
        filled = (pl[j + 1] <= c) if side == 1 else (ph[j + 1] >= c)
        if not filled:
            continue
        maxh = min(max(HOR), len(pc) - 1 - j)
        if maxh < 1:
            continue
        run_lo = np.minimum.accumulate(pl[j + 1: j + 1 + maxh])
        run_hi = np.maximum.accumulate(ph[j + 1: j + 1 + maxh])
        hit = (run_lo <= c * (1 - STOP)) if side == 1 \
            else (run_hi >= c * (1 + STOP))
        stop_at = int(np.argmax(hit)) if hit.any() else 10**9
        for h in HOR:
            if h > maxh:
                continue
            if stop_at < h:      # стоп случился внутри первых h баров
                exit_px = (c * (1 - STOP) * (1 - SLIP) if side == 1
                           else c * (1 + STOP) * (1 + SLIP))
                stopped = 1
            else:
                exit_px = pc[j + h] * (1 - side * SLIP)
                stopped = 0
            pnl = (exit_px / c - 1) if side == 1 else (1 - exit_px / c)
            pnl -= MAKER_FEE + TAKER_FEE + SLIP
            rows_out.append((tag, sym_id, ot, side, h, pnl, stopped))


rows = []

# ---------------- ФОЛДЫ (рецепт) ----------------
for qq in test_q:
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 5000 or len(trd) < 200_000:
        log(f"{qq}: пропущен (мало данных)")
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(trd[FEATS], trd["y"], categorical_feature=["sym_id"])
    del trd
    p = m.predict_proba(ted[FEATS])[:, 1]
    picked = signals(ted, p)
    eval_horizons(ted, picked, rows, f"fold|{qq}")
    log(f"фолд {qq}: сигналов {len(picked)}")

# ---------------- ЖИВАЯ МОДЕЛЬ (окно 07-26.10) ----------------
mfile = None
for root, _d, fs in os.walk("/kaggle/input"):
    if "ml5h.txt" in fs:
        mfile = os.path.join(root, "ml5h.txt")
        break
assert mfile, "ml5h.txt не найден"
burja = lgb.Booster(model_file=mfile)
log(f"живая модель: {burja.num_trees()} деревьев")

dom = pd.to_datetime(data["ot"], unit="ms", utc=True).dt.day.values
octm = (data["quarter"] == "2026Q4") & (dom >= 7)
d = data[octm].copy()
log(f"живое окно 07-26.10: {len(d)} баров, {d['sym_id'].nunique()} монет")
for tag, feats in (("live_B", FEATS), ("live_A", FEATS_A)):
    p = burja.predict(d[feats]).astype("float64")
    picked = signals(d, p)
    eval_horizons(d, picked, rows, f"{tag}|2026Q4oct")
    log(f"{tag}: сигналов {len(picked)}")

# ---------------- сводка ----------------
R = pd.DataFrame(rows, columns=["tag", "sym_id", "ot", "side",
                                "h", "pnl", "stopped"])
R["model"] = R["tag"].str.split("|").str[0]
R["quarter"] = R["tag"].str.split("|").str[1]
R.to_csv("/kaggle/working/trades.csv.gz", index=False, compression="gzip")


def block(g):
    b = {"n": int(len(g)),
         "avg_bp": round(g["pnl"].mean() * 1e4, 2),
         "win_pct": round((g["pnl"] > 0).mean() * 100, 1),
         "stop_pct": round(g["stopped"].mean() * 100, 1)}
    for s, name in ((1, "long"), (-1, "short")):
        gs = g[g["side"] == s]
        if len(gs) >= 20:
            b[f"{name}_bp"] = round(gs["pnl"].mean() * 1e4, 2)
            b[f"{name}_n"] = int(len(gs))
    return b


summary = {}
for model, g in R.groupby("model"):
    per_h = {}
    for h, gh in g.groupby("h"):
        e = block(gh)
        if model == "fold":
            qs = gh.groupby("quarter")["pnl"].agg(["mean", "count"])
            good = qs[qs["count"] >= 30]
            e["plus_q"] = int((good["mean"] > 0).sum())
            e["n_q"] = int(len(good))
        per_h[int(h)] = e
    summary[model] = per_h

with open("/kaggle/working/horizon_sweep.json", "w") as f:
    json.dump({"horizons_bars": HOR, "bar_minutes": 30, "gate": GATE,
               "cooldown_hours": 5, "costs_bp": 12, "stop": STOP,
               "meta_applied": False, "summary": summary}, f, indent=1)

hdr = ("горизонт | ФОЛДЫ: avg бп (плюс-кв, n) | live_B: avg бп (n) | "
       "live_A: avg бп (n) | стопы% фолды")
print("\n" + hdr, flush=True)
for h in HOR:
    f_ = summary.get("fold", {}).get(h, {})
    b_ = summary.get("live_B", {}).get(h, {})
    a_ = summary.get("live_A", {}).get(h, {})
    print(f"{h*30/60:5.1f}ч | {f_.get('avg_bp','-'):>8} "
          f"({f_.get('plus_q','-')}/{f_.get('n_q','-')}, n={f_.get('n','-')}) "
          f"| {b_.get('avg_bp','-'):>8} (n={b_.get('n','-')}) "
          f"| {a_.get('avg_bp','-'):>8} (n={a_.get('n','-')}) "
          f"| {f_.get('stop_pct','-')}", flush=True)

fold_h = summary.get("fold", {})
cands = [(v["avg_bp"], h, v) for h, v in fold_h.items() if v.get("n", 0) >= 1000]
if cands:
    best = max(cands)
    log(f"ЛУЧШИЙ фолд-горизонт: {best[1]*30/60:.1f}ч avg {best[0]:+.1f} бп "
        f"(родные 5ч = {fold_h.get(10, {}).get('avg_bp', '-')} бп)")
log("ГОРИЗОНТ-ЗАМОЕР ГОТОВ")
