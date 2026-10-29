"""МАТРИЦА «ШАГ × ГОРИЗОНТ» боевой связки Буря+Мета, по кругу, без
подглядывания. Вся панель (527 монет, 2020-01..2026-10), 8 честных
экзаменационных кварталов той же решёткой, что в замере горизонтов
(модель учится ТОЛЬКО на прошлом, предсказывает невиданный квартал).
Ступени выхода: от 30 минут до 12 часов, только те, что ложатся на
сетку шага ровно. Симуляция боевая: заполнение первым баром, стоп
-15%, издержки 12 бп.
МЕТА ПО КРУГУ (как месячный цикл): фолд 1 — контроль без меты; с фолда
2 маленькое дерево (признаки сигнала + уверенность + сторона) учится
на ИСХОДАХ OOS-сигналов ПРОШЛЫХ фолдов и вето-ит текущие при
p_win < 0.5. Отдельная линия «с метой» в сводке.
"""
import gc, io, json, os, time, zipfile, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
GATE = 0.65
COOLDOWN_MS = 5 * 3600_000          # 5ч между сигналами по монете
SLIP, TAKER_FEE, MAKER_FEE = 0.0005, 0.0005, 0.0002
STOP = 0.15
HOURS = [0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 12]
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
META_PARAMS = dict(n_estimators=120, learning_rate=0.05, max_depth=3,
                   num_leaves=7, subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=4, verbosity=-1)
FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
SYM_IDX = FEATS.index("sym_id")

# ===== НАСТРОЙКА ЭКЗЕМПЛЯРА (меняется генератором) =====
TF_NAME = "2h"
ZP_NAME = "panel.zip"
SUFFIX = "_30m.csv"
RESAMPLE = "2h"
TRAIN_STEP = 1   # 5м учится на каждой 6-й строке (сетка 30м)
# =======================================================

BOH = {"5m": 12, "15m": 4, "30m": 2, "1h": 1, "2h": 0.5}[TF_NAME]


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
    log(f"{TF_NAME}: строк {len(A['F'])}")
    return A


def hor_pairs():
    out = []
    for h in HOURS:
        b = int(round(h * BOH))
        if b >= 1 and abs(b / BOH - h) < 1e-9:
            out.append((b, h))
    return out


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
            picked.append((int(row.sym_id), int(row.ot), side, row.Index))
    return picked


def eval_horizons(ted, picked, rows_out, hp, tag):
    """Родная симуляция на каждой ступени. Возвращает посигнальные
    исходы (sym_id, ot, side, ridx, {часы: pnl}) для копилки меты."""
    pools = {int(fi): (g["close"].values, g["low"].values,
                       g["high"].values, g["ot"].values)
             for fi, g in ted.groupby("sym_id")}
    out_sig = []
    for sym_id, ot, side, ridx in picked:
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
        maxb = min(max(b for b, _ in hp), len(pc) - 1 - j)
        if maxb < 1:
            continue
        run_lo = np.minimum.accumulate(pl[j + 1: j + 1 + maxb])
        run_hi = np.maximum.accumulate(ph[j + 1: j + 1 + maxb])
        hit = (run_lo <= c * (1 - STOP)) if side == 1 \
            else (run_hi >= c * (1 + STOP))
        stop_at = int(np.argmax(hit)) if hit.any() else 10**9
        pnl_by_h = {}
        for h_bars, h_lab in hp:
            if h_bars > maxb:
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
            pnl_by_h[h_lab] = pnl
        out_sig.append((sym_id, ot, side, ridx, pnl_by_h))
    return out_sig


zp = find_zip(ZP_NAME)
assert zp, f"{ZP_NAME} не найден"
A = build_arrays(zp, SUFFIX, resample_from=RESAMPLE)
hp = hor_pairs()
qc = A["qc"]
codes = sorted(int(c) for c in np.unique(qc))
qq_all = [f"{2020 + c // 4}Q{c % 4 + 1}" for c in codes]
_idx = np.linspace(0, len(qq_all) - 1, 8).round().astype(int)
TEST_Q = [qq_all[i] for i in _idx]
log(f"экзамен-кварталы: {TEST_Q}; ступени часов: {[h for _, h in hp]}")

rows, rows_meta = [], []
hist_F, hist_p, hist_side, hist_pnls = [], [], [], []

for qq in TEST_Q:
    code = qcode_from_str(qq)
    mtr_idx = np.flatnonzero(qc < code)
    mte_idx = np.flatnonzero(qc == code)
    if len(mte_idx) < 5000 or len(mtr_idx) < 200_000:
        log(f"{qq}: пропущен (мало данных)")
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(A["F"][mtr_idx[::TRAIN_STEP]], A["y"][mtr_idx[::TRAIN_STEP]],
          categorical_feature=[SYM_IDX])
    p = m.predict_proba(A["F"][mte_idx])[:, 1]
    Fq = A["F"][mte_idx]
    ted = pd.DataFrame({"sym_id": Fq[:, SYM_IDX], "ot": A["ot"][mte_idx],
                        "close": A["close"][mte_idx],
                        "low": A["low"][mte_idx],
                        "high": A["high"][mte_idx]})
    picked = signals(ted, p)
    sigs = eval_horizons(ted, picked, rows, hp, f"{TF_NAME}|{qq}")
    log(f"фолд {qq}: сигналов {len(picked)}, с исходами {len(sigs)}")
    del m
    gc.collect()
    if not sigs:
        continue
    curF = np.array([Fq[s[3]] for s in sigs])
    curP = np.array([p[s[3]] for s in sigs], dtype="float32")
    curS = np.array([s[2] for s in sigs], dtype="float32")
    # --- мета по кругу: учится ТОЛЬКО на прошлых фолдах, потом вето ---
    if hist_F:
        Xh = np.column_stack([np.array(hist_F), np.array(hist_p),
                              np.array(hist_side, dtype="float32")])
        Xc = np.column_stack([curF, curP, curS])
        for h_lab, _ in [(h, 0) for _, h in hp]:
            yh, okh = [], []
            for d in hist_pnls:
                v = d.get(h_lab)
                if v is not None:
                    yh.append(1.0 if v > 0 else 0.0)
                    okh.append(True)
                else:
                    okh.append(False)
            okh = np.array(okh)
            if okh.sum() < 300:
                continue
            yh = np.array(yh)
            mt = lgb.LGBMClassifier(**META_PARAMS)
            mt.fit(Xh[okh], yh, categorical_feature=[SYM_IDX])
            pw = mt.predict_proba(Xc)[:, 1]
            kept = pw >= 0.5
            for keep, (sym_id, ot, side, ridx, pnl_by_h) in zip(kept, sigs):
                v = pnl_by_h.get(h_lab)
                if v is not None:
                    rows_meta.append((tag_h := f"{TF_NAME}|{qq}", sym_id,
                                      ot, side, h_lab, v, 0 if keep else 1))
            del mt
        gc.collect()
    # --- пополнение копилки ТОЛЬКО текущим фолдом ---
    for (sym_id, ot, side, ridx, pnl_by_h) in sigs:
        hist_F.append(Fq[ridx])
        hist_p.append(p[ridx])
        hist_side.append(side)
        hist_pnls.append(pnl_by_h)
    del sigs, curF, curP, curS, Fq, ted
    gc.collect()

R = pd.DataFrame(rows, columns=["tag", "sym_id", "ot", "side",
                                "h", "pnl", "stopped"])
R["quarter"] = R["tag"].str.split("|").str[1]
RM = pd.DataFrame(rows_meta, columns=["tag", "sym_id", "ot", "side",
                                      "h", "pnl", "vetoed"])
RM["quarter"] = RM["tag"].str.split("|").str[1]
del rows, rows_meta
R.to_csv(f"/kaggle/working/trades_{TF_NAME}.csv.gz", index=False,
         compression="gzip")


def block(g):
    if not len(g):
        return None
    qs = g.groupby("quarter")["pnl"].agg(["mean", "count"])
    good = qs[qs["count"] >= 30]
    e = {"n": int(len(g)),
         "avg_bp": round(float(g["pnl"].mean() * 1e4), 2),
         "win_pct": round(float((g["pnl"] > 0).mean() * 100), 1),
         "stop_pct": (round(float(g["stopped"].mean() * 100), 1)
                      if "stopped" in g.columns else None),
         "plus_q": f"{int((good['mean'] > 0).sum())}/{int(len(good))}"}
    for s, name in ((1, "long"), (-1, "short")):
        gs = g[g["side"] == s]
        if len(gs) >= 20:
            e[f"{name}_bp"] = round(float(gs["pnl"].mean() * 1e4), 1)
    return e


summary = {"raw": {}, "meta": {}}
for h_lab in [h for _, h in hp]:
    summary["raw"][str(h_lab)] = block(R[R["h"] == h_lab])
    kept = RM[(RM["h"] == h_lab) & (RM["vetoed"] == 0)]
    e = block(kept)
    if e is not None:
        allr = RM[RM["h"] == h_lab]
        e["kept_pct"] = round(100 * len(kept) / max(1, len(allr)), 1)
    summary["meta"][str(h_lab)] = e

with open(f"/kaggle/working/matrix_{TF_NAME}.json", "w") as f:
    json.dump({"tf": TF_NAME, "hours": [h for _, h in hp],
               "test_quarters": TEST_Q, "gate": GATE, "costs_bp": 12,
               "stop": STOP, "train_step": TRAIN_STEP, "summary": summary},
              f, indent=1, default=float)

print(f"\n{TF_NAME}: горизонт | без меты: avg (пк) | с метой: avg (пк) "
      "| принято% | стопы%", flush=True)
for h_lab in [h for _, h in hp]:
    e1 = summary["raw"].get(str(h_lab)) or {}
    e2 = summary["meta"].get(str(h_lab)) or {}
    print(f"{h_lab:6.2f}ч | {e1.get('avg_bp','-'):>8} ({e1.get('plus_q','-')}) "
          f"| {e2.get('avg_bp','-'):>8} ({e2.get('plus_q','-')}) "
          f"| {e2.get('kept_pct','-')} | {e1.get('stop_pct','-')}", flush=True)
log(f"МАТРИЦА {TF_NAME} ГОТОВА")
