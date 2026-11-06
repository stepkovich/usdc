"""ТИХИЕ ДНИ (вопрос владельца 06.11): были ли раньше дни без кандидатов?
Протокол: 8 честных фолдов (обучение строго до своего квартала), гейт
0.65, обе стороны, кулдаун 5ч — как в бою. Кандидат дня = сигнал после
кулдауна. Два разреза: ВСЕ монеты и ТОП-200 по обороту за 30 дней до
дня (как боевой фильтр). Считаем: распределение дней по числу
кандидатов, серии подряд «пустых» дней (<1 кандидата), по годам.
"""
import io, json, os, time, zipfile, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
GATE = 0.65
COOLDOWN_MS = 5 * 3600_000
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
SYM_IDX = FEATS.index("sym_id")
EXCLUDE = {zlib.crc32(b"BTCUSDT") & 0xFFFF,
           zlib.crc32(b"ETHUSDT") & 0xFFFF}


def log(m):
    print(f"{m} | {time.time()-t0:.0f}с", flush=True)


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
    return X, tr


def find_zip(name):
    for root, _d, fs in os.walk("/kaggle/input"):
        if name in fs:
            return os.path.join(root, name)
    return None


def build_arrays(zpath):
    zf = zipfile.ZipFile(zpath)
    members = sorted(n for n in zf.namelist() if n.endswith("_30m.csv"))
    log(f"панель: {len(members)} монет")
    btc = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
    btc["open_time"] = btc["open_time"].astype(np.int64)
    btc.index = pd.to_datetime(btc["open_time"], unit="ms")
    btc = btc["close"].astype("float32")
    Fs, ys, ots = [], [], []
    sym_vol = {}
    for i, name in enumerate(members, 1):
        sym = name.replace("_30m.csv", "")
        df = pd.read_csv(io.BytesIO(zf.read(name)),
                         usecols=["open_time", "high", "low", "close",
                                  "volume", "quote_volume",
                                  "taker_buy_base"])
        for c in df.columns:
            if c != "open_time":
                df[c] = df[c].astype("float32")
        df["open_time"] = df["open_time"].astype(np.int64)
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
        fcols = [c for c in FEATS if c in X.columns]
        keep = X[fcols + ["y"]].notna().all(axis=1).values
        ot_i = df["open_time"].values.astype(np.int64)
        if keep.any():
            sid = np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)
            Fs.append(X.loc[keep, fcols]
                      .reindex(columns=FEATS).to_numpy(dtype="float32"))
            ys.append(y[keep].astype("float32"))
            ots.append(ot_i[keep])
            sym_vol[int(sid)] = (ot_i[keep],
                                 df["quote_volume"].values[keep])
        del X, df
        if i % 150 == 0:
            log(f"{i}/{len(members)}")
    F = np.concatenate(Fs); del Fs
    ot = np.concatenate(ots); del ots
    return {"F": F, "y": np.concatenate(ys), "ot": ot}, sym_vol


zp = find_zip("panel.zip")
assert zp, "panel.zip не найден"
A, sym_vol = build_arrays(zp)
qc = qcode = None
idx = pd.PeriodIndex(pd.to_datetime(A["ot"], unit="ms"), freq="Q")
A["qc"] = ((idx.year - 2020) * 4 + idx.quarter).astype(np.int16)
qc = A["qc"]
codes = sorted(int(c) for c in np.unique(qc))
qq_all = [f"{2020 + c // 4}Q{c % 4 + 1}" for c in codes]
_ix = np.linspace(0, len(qq_all) - 1, 8).round().astype(int)
TEST_Q = [qq_all[i] for i in _ix]
log(f"экзамен: {TEST_Q}")

day_cand = {}          # day -> [n_all, n_top200]
for qq in TEST_Q:
    code = (int(qq[:4]) - 2020) * 4 + int(qq[5])
    mtr = np.flatnonzero(qc < code)
    mte = np.flatnonzero(qc == code)
    if len(mte) < 5000 or len(mtr) < 200_000:
        log(f"{qq}: пропущен")
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(A["F"][mtr], A["y"][mtr], categorical_feature=[SYM_IDX])
    p = m.predict_proba(A["F"][mte])[:, 1]
    del m
    d = pd.DataFrame({"sym_id": A["F"][mte, SYM_IDX],
                      "ot": A["ot"][mte]}).assign(p=p)
    picked = []
    for side in (1, -1):
        sel = d[(d["p"] > GATE) if side == 1 else (d["p"] < 1 - GATE)] \
            .sort_values(["sym_id", "ot"])
        last = {}
        for row in sel.itertuples():
            if row.ot - last.get(int(row.sym_id), -10**18) < COOLDOWN_MS:
                continue
            last[int(row.sym_id)] = row.ot
            picked.append((int(row.sym_id), int(row.ot)))
    q_start = A["ot"][mte[0]]
    top200 = {}
    # топ-200 считается РАЗ В КВАРТАЛ по первым 30 дням до него —
    # внутри квартала вселенная фиксирована (как месячный цикл)
    vol30 = {}
    for sid, (sot, svol) in sym_vol.items():
        w = (sot >= q_start - 30 * 86400_000) & (sot < q_start)
        vol30[sid] = float(svol[w].sum())
    rank = [sid for sid, _ in sorted(vol30.items(), key=lambda kv: -kv[1])
            if sid not in EXCLUDE]
    T200 = set(rank[:200])
    for sid, ot_s in picked:
        day = pd.Timestamp(ot_s, unit="ms").strftime("%Y-%m-%d")
        e = day_cand.setdefault(day, [0, 0])
        e[0] += 1
        if sid in T200:
            e[1] += 1
    log(f"{qq}: сигналов {len(picked)}")
    del d, p
    import gc; gc.collect()

days = sorted(day_cand)
arr_all = np.array([day_cand[d][0] for d in days])
arr_t200 = np.array([day_cand[d][1] for d in days])


def streaks(arr):
    runs = []
    cur = 0
    for v in arr:
        if v == 0:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return runs


def report(name, arr):
    runs = streaks(arr)
    runs_s = sorted(runs, reverse=True)
    hist = {}
    for r in runs:
        hist[r] = hist.get(r, 0) + 1
    print(f"\n== {name}: дней {len(arr)} | пустых {int((arr==0).sum())} "
          f"({(arr==0).mean()*100:.0f}%) | медиана кандидатов/день "
          f"{np.median(arr):.0f} ==", flush=True)
    print(f"серий пустых дней: {len(runs)} | топ-5 длин: "
          f"{runs_s[:5]}", flush=True)
    print(f"гистограмма серий: {dict(sorted(hist.items()))}", flush=True)
    # самые длинные серии с датами
    out = []
    cur = 0; start = None
    for i, v in enumerate(arr):
        if v == 0:
            if not cur: start = days[i]
            cur += 1
        elif cur:
            out.append((cur, start, days[i-1])); cur = 0
    if cur:
        out.append((cur, start, days[-1]))
    for L, a, b in sorted(out, reverse=True)[:5]:
        print(f"  {L} дн: {a} .. {b}", flush=True)


report("ВСЕ МОНЕТЫ", arr_all)
report("ТОП-200 (как в бою)", arr_t200)
yrs = {}
for d, v in zip(days, arr_all):
    yrs.setdefault(d[:4], [0, 0])
    yrs[d[:4]][0] += 1
    yrs[d[:4]][1] += (v == 0)
print("\nпустые дни по годам (все монеты):", flush=True)
for y in sorted(yrs):
    print(f"  {y}: {yrs[y][1]}/{yrs[y][0]} ({yrs[y][1]/yrs[y][0]*100:.0f}%)",
          flush=True)
with open("/kaggle/working/quiet_report.json", "w") as f:
    json.dump({"days": days, "cand_all": arr_all.tolist(),
               "cand_top200": arr_t200.tolist()}, f)
log("ТИХИЕ-ДНИ ГОТОВ")
