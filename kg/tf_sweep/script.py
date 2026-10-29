"""РАУНД 2 v2 — ШАГ СВЕЧИ: та же связка (24 признака, гейт 0.65, метка
t+10 баров, порог 0.5*ATR) на 5м / 15м / 30м, честные фолды
2024Q4-2026Q4, перебор ГОРИЗОНТА ВЫХОДА в настенном времени.
v2: v1 убила память Кегля (79М строк 5м: дубли на concat + строки-кварталы
+ копия обучающей выборки). Теперь держим только numpy-массивы, шаги
свечи обрабатываем ПО ОЧЕРЕДИ с полной выгрузкой, обучающая выборка —
булев-маской по четверти (одна копия на фолд, потом свобода).
Фолд = модель учится ТОЛЬКО на прошлом. Симуляция родная: заполнение
первым баром, стоп -15%, издержки 12 бп, мета не участвует.
"""
import gc, io, json, os, time, zipfile, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
GATE = 0.65
COOLDOWN_MS = 5 * 3600_000
SLIP, TAKER_FEE, MAKER_FEE = 0.0005, 0.0005, 0.0002
STOP = 0.15
HOURS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 12]
TEST_Q = ["2024Q4", "2025Q4", "2026Q4"]
MIN_OT = 1704067200000 + 7 * 86400_000   # 2024 + неделя прогрева окон
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


def find_zip(name):
    for root, _d, fs in os.walk("/kaggle/input"):
        if name in fs:
            return os.path.join(root, name)
    return None


def qcode_of(ot):
    idx = pd.to_datetime(ot, unit="ms")
    return ((idx.year - 2020) * 4 + idx.quarter).astype(np.int16)


def qcode_from_str(qq):
    y, q = qq.split("Q")
    return (int(y) - 2020) * 4 + int(q)


def build_arrays(zpath, suffix, resample_from=None):
    """Панель -> numpy-блоки без промежуточного гигантского DataFrame."""
    zf = zipfile.ZipFile(zpath)
    members = sorted(n for n in zf.namelist() if n.endswith(suffix))
    log(f"{suffix}: {len(members)} монет из {os.path.basename(zpath)}")
    btc = pd.read_csv(io.BytesIO(zf.read(f"BTCUSDT{suffix}")),
                      usecols=["open_time", "close"])
    btc["open_time"] = btc["open_time"].astype(np.int64)
    btc.index = pd.to_datetime(btc["open_time"], unit="ms")
    btc = btc["close"].astype("float32")

    Fs, ys, cs, ls, hs, ots = [], [], [], [], [], []
    for i, name in enumerate(members, 1):
        sym = name.replace(suffix + ".csv", "")
        df = pd.read_csv(io.BytesIO(zf.read(name)),
                         usecols=["open_time", "high", "low", "close",
                                  "volume", "taker_buy_base"])
        for c in df.columns:
            if c != "open_time":
                df[c] = df[c].astype("float32")
        df["open_time"] = df["open_time"].astype(np.int64)
        df.index = pd.to_datetime(df["open_time"], unit="ms")
        df = df[df["open_time"] >= MIN_OT]
        if not len(df):
            continue
        if resample_from:
            r = df.resample(resample_from).agg(
                {"high": "max", "low": "min", "close": "last",
                 "volume": "sum", "taker_buy_base": "sum"})
            r = r.dropna(subset=["close"])
            r["open_time"] = (r.index.asi8 // 10**6).astype(np.int64)
            df = r
            for c in ("high", "low", "close", "volume", "taker_buy_base"):
                df[c] = df[c].astype("float32")
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
        keep = X[FEATS + ["y"]].notna().all(axis=1).values
        if keep.any():
            Fs.append(X.loc[keep, FEATS].to_numpy(dtype="float32"))
            ys.append(y[keep].astype("float32"))
            cs.append(close[keep])
            ls.append(df["low"].values[keep])
            hs.append(df["high"].values[keep])
            ots.append(df["open_time"].values[keep])
        del X, df
        if i % 100 == 0:
            log(f"{suffix} {i}/{len(members)}")
    A = {"F": np.concatenate(Fs),
         "y": np.concatenate(ys),
         "close": np.concatenate(cs),
         "low": np.concatenate(ls),
         "high": np.concatenate(hs),
         "ot": np.concatenate(ots)}
    del Fs, ys, cs, ls, hs, ots
    A["qc"] = qcode_of(A["ot"])
    log(f"{suffix}: строк {len(A['F'])}")
    return A


def signals(ted, p):
    d = ted.assign(p=p)
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


def eval_horizons(ted, picked, rows_out, tag, bars_of_hour):
    pools = {int(fi): (g["close"].values, g["low"].values,
                       g["high"].values, g["ot"].values)
             for fi, g in ted.groupby("sym_id")}
    hor = [int(round(h * bars_of_hour)) for h in HOURS]
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
        maxh = min(max(hor), len(pc) - 1 - j)
        if maxh < 1:
            continue
        run_lo = np.minimum.accumulate(pl[j + 1: j + 1 + maxh])
        run_hi = np.maximum.accumulate(ph[j + 1: j + 1 + maxh])
        hit = (run_lo <= c * (1 - STOP)) if side == 1 \
            else (run_hi >= c * (1 + STOP))
        stop_at = int(np.argmax(hit)) if hit.any() else 10**9
        for h_bars, h_lab in zip(hor, HOURS):
            if h_bars > maxh:
                continue
            if stop_at < h_bars:
                exit_px = (c * (1 - STOP) * (1 - SLIP) if side == 1
                           else c * (1 + STOP) * (1 + SLIP))
                stopped = 1
            else:
                exit_px = pc[j + h_bars] * (1 - side * SLIP)
                stopped = 0
            pnl = (exit_px / c - 1) if side == 1 else (1 - exit_px / c)
            pnl -= MAKER_FEE + TAKER_FEE + SLIP
            rows_out.append((tag, sym_id, ot, side, h_lab, pnl, stopped))


rows = []
zp5 = find_zip("panel_5m.zip")
zp30 = find_zip("panel.zip")
assert zp5 and zp30, f"панели не найдены: 5m={zp5} 30m={zp30}"

for tf_name, zpath, suffix, rs in (("5m", zp5, "_5m.csv", None),
                                   ("15m", zp5, "_5m.csv", "15min"),
                                   ("30m", zp30, "_30m.csv", None)):
    A = build_arrays(zpath, suffix, resample_from=rs)
    bars_of_hour = 60 / {"5m": 5, "15m": 15, "30m": 30}[tf_name]
    for qq in TEST_Q:
        code = qcode_from_str(qq)
        mask_tr = A["qc"] < code
        mask_te = A["qc"] == code
        if mask_te.sum() < 5000 or mask_tr.sum() < 200_000:
            log(f"{tf_name} {qq}: пропущен (мало данных)")
            continue
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(A["F"][mask_tr], A["y"][mask_tr],
              categorical_feature=[SYM_IDX])
        p = m.predict_proba(A["F"][mask_te])[:, 1]
        sl = np.flatnonzero(mask_te)
        ted = pd.DataFrame({"sym_id": A["F"][sl, SYM_IDX],
                            "ot": A["ot"][sl],
                            "close": A["close"][sl],
                            "low": A["low"][sl],
                            "high": A["high"][sl]})
        picked = signals(ted, p)
        eval_horizons(ted, picked, rows, f"{tf_name}|{qq}", bars_of_hour)
        log(f"{tf_name} фолд {qq}: сигналов {len(picked)}")
        del m, p, ted, picked
        gc.collect()
    del A
    gc.collect()
    log(f"{tf_name}: готово")

R = pd.DataFrame(rows, columns=["tag", "sym_id", "ot", "side",
                                "h", "pnl", "stopped"])
del rows
R["tf"] = R["tag"].str.split("|").str[0]
R["quarter"] = R["tag"].str.split("|").str[1]
R.to_csv("/kaggle/working/tf_trades.csv.gz", index=False, compression="gzip")

summary = {}
for tf, g in R.groupby("tf"):
    per_h = {}
    for h, gh in g.groupby("h"):
        qs = gh.groupby("quarter")["pnl"].agg(["mean", "count"])
        good = qs[qs["count"] >= 30]
        e = {"n": int(len(gh)),
             "avg_bp": round(float(gh["pnl"].mean() * 1e4), 2),
             "win_pct": round(float((gh["pnl"] > 0).mean() * 100), 1),
             "stop_pct": round(float(gh["stopped"].mean() * 100), 1),
             "plus_q": f"{int((good['mean'] > 0).sum())}/{int(len(good))}"}
        for s, name in ((1, "long"), (-1, "short")):
            gs = gh[gh["side"] == s]
            if len(gs) >= 20:
                e[f"{name}_bp"] = round(float(gs["pnl"].mean() * 1e4), 1)
                e[f"{name}_n"] = int(len(gs))
        per_h[float(h)] = e
    summary[tf] = per_h

with open("/kaggle/working/tf_sweep.json", "w") as f:
    json.dump({"hours": HOURS, "test_quarters": TEST_Q, "gate": GATE,
               "costs_bp": 12, "stop": STOP, "summary": summary},
              f, indent=1, default=float)

print("\nГОРИЗОНТ |   5м: avg (плюс-кв) |  15м: avg (плюс-кв) "
      "|  30м: avg (плюс-кв)", flush=True)
for h in HOURS:
    line = f"{h:6.1f}ч |"
    for tf in ("5m", "15m", "30m"):
        e = summary.get(tf, {}).get(float(h), {})
        line += f" {e.get('avg_bp','-'):>7} ({e.get('plus_q','-')}) |"
    print(line, flush=True)
log("ШАГ-СВЕЧИ ЗАМЕР ГОТОВ")
