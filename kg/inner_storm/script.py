"""INNER-STORM: внутр-признаки из книги как добавка к модели Бури
(предрегистрация 07.10, идея владельца «свеча прячет всё» — применить
на вопросе Бури).

A/B: панель 520 монет (30м, 8 кварталов, gate 0.65, будильник,
тейкер 0.2% круг), модель A = 24 признака (реплика боевой),
модель B = 24 + 6 внутрь (только на строках, где книга писалась;
остальные has_inner = 0).
ВНУТРЬ-ПРИЗНАКИ (6): inner_below, inner_up_max, inner_dn_max,
inner_drift, inner_jitter, inner_n — из записи книги 37 пар.
ВЕРДИКТ: B жив, если avg(B) − avg(A) >= +3 бп И avg(B) > 0.
Ничего не деплоится.
"""
import glob, io, json, time, zipfile, zlib
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000
FEE_ROUND = 0.0020
N_TEST_DAYS = 3
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
SYMS37 = ["1000BONKUSDC", "1000PEPEUSDC", "1000SHIBUSDC", "AAVEUSDC",
          "ADAUSDC", "ARBUSDC", "AVAXUSDC", "BCHUSDC", "BIOUSDC",
          "BNBUSDC", "BOMEUSDC", "BTCUSDC", "CRVUSDC", "DATAIPUSDC",
          "DOGEUSDC", "ENAUSDC", "ETHFIUSDC", "ETHUSDC", "FILUSDC",
          "HBARUSDC", "KAITOUSDC", "LINKUSDC", "LTCUSDC", "NEARUSDC",
          "NEOUSDC", "ORDIUSDC", "PENGUUSDC", "PNUTUSDC", "SOLUSDC",
          "SUIUSDC", "TIAUSDC", "TRUMPUSDC", "UNIUSDC", "WIFUSDC",
          "WLDUSDC", "WLFIUSDC", "XRPUSDC", "ZECUSDC"]
LOB_FEATS = ["spread_bp", "microprice_rel", "imb5", "imb10", "imb20",
             "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
             "ntr10", "vpin10", "d30", "d120"]
INNER = ["inner_below", "inner_up_max", "inner_dn_max", "inner_drift",
         "inner_jitter", "inner_n"]

# ---------- книга: внутр-признаки 37 пар ----------
book_frames = []
for f in sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True)):
    try:
        bf = pd.read_parquet(f, columns=["ts", "symbol", "mid",
                                         "spread_bp"])
        book_frames.append(bf.iloc[::2])
    except Exception:
        pass
book = pd.concat(book_frames, ignore_index=True)
del book_frames
book = book.dropna(subset=["mid", "spread_bp"])
book = book[book["symbol"].isin(SYMS37)]
book["bin30"] = (book["ts"] // 1_800_000) * 1_800_000
print(f"книга: {len(book)} строк | {time.time()-t0:.0f}с", flush=True)

inner_map = {}
for (sym, bin30), g in book.groupby(["symbol", "bin30"]):
    if len(g) < 10:
        continue
    g = g.sort_values("ts")
    open_px = g["mid"].iloc[0]
    rel = (g["mid"] / open_px - 1) * 1e4
    thirds = np.array_split(rel.values, 3)
    steps = np.diff(g["mid"].values)
    inner_map[(sym, int(bin30))] = {
        "inner_below": float((rel < 0).mean()),
        "inner_up_max": float(rel.max()),
        "inner_dn_max": float(rel.min()),
        "inner_drift": float(thirds[2].mean() - thirds[0].mean()),
        "inner_jitter": float(np.std(steps)) if len(steps) > 1 else 0.0,
        "inner_n": float(len(g))}
del book
print(f"внутрь-бинов {len(inner_map)}", flush=True)

# ---------- панель ----------
zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
print("монет в панели:", len(members), flush=True)
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
    X["zscore"] = ((close - sma) / std).replace([np.inf, -np.inf],
                                                np.nan).astype("float32")
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

FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]

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
    fut = np.full(len(df), np.nan, dtype="float64")
    fut[:-H] = close[H:]
    thr_mask = (~np.isnan(fwd)) if 'fwd' in dir() else None
    # пересчёт fwd
    fwd = fut / close - 1
    thr = 0.5 * np.nan_to_num(
        (pd.Series(df["high"].values - df["low"].values)
         .rolling(14).mean() / close).values.astype("float64"), nan=np.nan)
    X["y"] = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                      (np.abs(fwd) > thr).astype("float32"))
    X["fwd"] = fwd.astype("float32")
    X["ot"] = df["open_time"].values.astype(np.int64)
    X["sym"] = sym
    X["is37"] = 1.0 if sym in SYMS37 else 0.0
    # джойн внутр-признаков для 37 пар
    if sym in SYMS37:
        ot_set = set(int(x) // 1_800_000 * 1_800_000 for x in
                     df["open_time"].values)
        for row_ot in df["open_time"].values:
            key = (sym, int(row_ot // 1_800_000 * 1_800_000))
            pass  # слишком медленно, делаем через merge ниже
    frames.append(X)
    if i % 100 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
# джойн внутр через bin30
data["bin30"] = (data["ot"] // 1_800_000) * 1_800_000
im_df = pd.DataFrame([{"sym": k[0], "bin30": k[1], **v}
                      for k, v in inner_map.items()])
data = data.merge(im_df, left_on=["sym", "bin30"],
                  right_on=["sym", "bin30"], how="left")
data["has_inner"] = data["inner_below"].notna().astype("float32")
for c in INNER:
    data[c] = data[c].fillna(0.0).astype("float32")
data = data.dropna(subset=FEATS + ["y"])
print(f"строк {len(data)}, has_inner {int(data['has_inner'].sum())}",
      flush=True)

# ---------- walk-forward 8 кварталов ----------
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]

acc_rows = []
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 20000 or len(trd) < 500_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    trd_ok = trd.dropna(subset=["y"])
    m.fit(trd_ok[FEATS], trd_ok["y"], categorical_feature=["sym_id"])
    ted_ok = ted.dropna(subset=FEATS)
    p = m.predict_proba(ted_ok[FEATS])[:, 1]
    for sym, ot, pv in zip(ted_ok["sym"].values, ted_ok["ot"].values, p):
        acc_rows.append({"sym": sym, "ot": int(ot), "p": float(pv)})
    print(f"фолд {qq}: {len(ted_ok)} строк | {time.time()-t0:.0f}с",
          flush=True)

AR = pd.DataFrame(acc_rows)
AR = AR.drop_duplicates(subset=["sym", "ot"])
print(f"предсказаний {len(AR)}", flush=True)

report = {"combos": {}}
for nm, feats in (("A_24", FEATS), ("B_30", FEATS + INNER)):
    pnls = []
    for k, qq in enumerate(test_q):
        trd = data[data["quarter"] < qq]
        ted = data[data["quarter"] == qq]
        if len(ted) < 20000 or len(trd) < 500_000:
            continue
        m = lgb.LGBMClassifier(**PARAMS)
        trd_ok = trd.dropna(subset=["y"])
        m.fit(trd_ok[feats], trd_ok["y"], categorical_feature=["sym_id"])
        te_ok = ted.dropna(subset=feats)
        p = m.predict_proba(te_ok[feats])[:, 1]
        te_ok = te_ok.assign(p=p)
        sel = te_ok[(te_ok["p"] > GATE) | (te_ok["p"] < 1 - GATE)] \
            .sort_values(["sym_id", "ot"])
        last_t = {}
        for r in sel.itertuples():
            if r.ot - last_t.get(r.sym_id, -10**18) < COOLDOWN_MS:
                continue
            last_t[r.sym_id] = r.ot
            sig = 1 if r.p > GATE else -1
            pnl = sig * r.fwd - FEE_ROUND
            pnls.append(pnl)
    if pnls:
        a = np.array(pnls)
        report["combos"][nm] = {"n": int(len(a)),
                                "avg_bp": round(float(a.mean() * 10000), 2),
                                "win_pct": round(float((a > 0).mean()
                                                       * 100), 1)}
        print(f"{nm}: {len(a)} сд, avg {a.mean()*1e4:+.2f} бп, "
              f"win {float((a>0).mean()*100):.1f}%", flush=True)
    else:
        report["combos"][nm] = {"n": 0}
        print(f"{nm}: 0 сделок", flush=True)

a_combo = report["combos"].get("A_24", {})
b_combo = report["combos"].get("B_30", {})
if a_combo.get("n") and b_combo.get("n"):
    delta = b_combo["avg_bp"] - a_combo["avg_bp"]
    report["delta_bp"] = round(delta, 2)
    alive = bool(delta >= 3.0)
    report["alive"] = alive
    print(f"Δ = {delta:+.2f} бп | ВЕРДИКТ:",
          "B ЖИВ" if alive else "B не перебил A", flush=True)
else:
    print("ВЕРДИКТ: нет данных", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("INNER-STORM завершена", flush=True)
