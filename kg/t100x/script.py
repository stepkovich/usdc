"""ИДЕЯ ВЛАДЕЛЬЦА (04.11): торговать только топ-100 по объёму, БЕЗ BTC/ETH
(«шиткоины бешеные, закономерностей там искать не надо»). Максимально
честно: топ считается по обороту за 30 дней ДО экзамен-квартала (без
подглядывания), параметры боевого рецепта заморожены, фолды те же 8
кварталов 2020-2026, симуляция та же (стоп -15%, 12 бп, заполнение
первым баром, кулдаун 5ч, гейт 0.65, горизонт 5ч). Мета НЕ применяется
(замер рецепта, как в reentry-test 02.11 — сравнимо).
Вселенные: ALL (база) / T100 (реплика 02.11) / T100X (топ-100 без
BTC+ETH) / T50X (топ-50 без BTC+ETH). BTC строк не имеет (эталон
признаков -> NaN -> отброшен), ETH исключается из рейтинга и торговли.
"""
import io, json, os, time, zipfile, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
GATE = 0.65
COOLDOWN_MS = 5 * 3600_000
SLIP, TAKER_FEE, MAKER_FEE = 0.0005, 0.0005, 0.0002
STOP = 0.15
H_H = 10
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
SYM_IDX = FEATS.index("sym_id")
EXCLUDE = {zlib.crc32(b"BTCUSDT") & 0xFFFF, zlib.crc32(b"ETHUSDT") & 0xFFFF}
BTC_SID = zlib.crc32(b"BTCUSDT") & 0xFFFF


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


def qcode_of(ot):
    idx = pd.to_datetime(ot, unit="ms")
    return ((idx.year - 2020) * 4 + idx.quarter).astype(np.int16)


def build_arrays(zpath):
    zf = zipfile.ZipFile(zpath)
    members = sorted(n for n in zf.namelist() if n.endswith("_30m.csv"))
    log(f"панель: {len(members)} монет")
    btc = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
    btc["open_time"] = btc["open_time"].astype(np.int64)
    btc.index = pd.to_datetime(btc["open_time"], unit="ms")
    btc = btc["close"].astype("float32")
    Fs, ys, cs, ls, hs, ots = [], [], [], [], [], []
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
            cs.append(close[keep]); ls.append(df["low"].values[keep])
            hs.append(df["high"].values[keep])
            ots.append(ot_i[keep])
            sym_vol[int(sid)] = (ot_i[keep],
                                 df["quote_volume"].values[keep])
        del X, df
        if i % 100 == 0:
            log(f"{i}/{len(members)}")
    A = {"F": np.concatenate(Fs), "y": np.concatenate(ys),
         "close": np.concatenate(cs), "low": np.concatenate(ls),
         "high": np.concatenate(hs), "ot": np.concatenate(ots)}
    del Fs, ys, cs, ls, hs, ots
    A["qc"] = qcode_of(A["ot"])
    log(f"строк {len(A['F'])}")
    return A, sym_vol


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


def eval_pnl(ted, picked):
    pools = {int(fi): (g["close"].values, g["low"].values,
                       g["high"].values, g["ot"].values)
             for fi, g in ted.groupby("sym_id")}
    out = []
    for sym_id, ot, side, ridx in picked:
        pool = pools.get(sym_id)
        if pool is None:
            continue
        pc, pl, ph, pots = pool
        j = int(np.searchsorted(pots, ot))
        if j >= len(pots) or pots[j] != ot or j + 1 >= len(pc):
            continue
        c = pc[j]
        if H_H >= len(pc) - j:
            continue
        filled = (pl[j + 1] <= c) if side == 1 else (ph[j + 1] >= c)
        if not filled:
            continue
        seg_lo = pl[j + 1: j + 1 + H_H]
        seg_hi = ph[j + 1: j + 1 + H_H]
        hit = (seg_lo.min() <= c * (1 - STOP)) if side == 1 \
            else (seg_hi.max() >= c * (1 + STOP))
        if hit:
            exit_px = (c * (1 - STOP) * (1 - SLIP) if side == 1
                       else c * (1 + STOP) * (1 + SLIP))
        else:
            exit_px = pc[j + H_H] * (1 - side * SLIP)
        pnl = (exit_px / c - 1) if side == 1 else (1 - exit_px / c)
        pnl -= MAKER_FEE + TAKER_FEE + SLIP
        out.append((sym_id, ot, side, pnl))
    return out


zp = find_zip("panel.zip")
assert zp, "panel.zip не найден"
A, sym_vol = build_arrays(zp)
qc = A["qc"]
codes = sorted(int(c) for c in np.unique(qc))
qq_all = [f"{2020 + c // 4}Q{c % 4 + 1}" for c in codes]
idx = np.linspace(0, len(qq_all) - 1, 8).round().astype(int)
TEST_Q = [qq_all[i] for i in idx]
log(f"экзамен: {TEST_Q}")


def qcode_from_str(qq):
    y, q = qq.split("Q")
    return (int(y) - 2020) * 4 + int(q)


UNIS = ["ALL", "T100", "T100X", "T50X"]
RES = {u: [] for u in UNIS}

for qq in TEST_Q:
    code = qcode_from_str(qq)
    mtr = np.flatnonzero(qc < code)
    mte = np.flatnonzero(qc == code)
    if len(mte) < 5000 or len(mtr) < 200_000:
        log(f"{qq}: пропущен")
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(A["F"][mtr], A["y"][mtr], categorical_feature=[SYM_IDX])
    p = m.predict_proba(A["F"][mte])[:, 1]
    sl = mte
    Fq = A["F"][sl]
    ted = pd.DataFrame({"sym_id": Fq[:, SYM_IDX], "ot": A["ot"][sl],
                        "close": A["close"][sl], "low": A["low"][sl],
                        "high": A["high"][sl]})
    picked = signals(ted, p)
    del m
    # топ по обороту за 30 дней ДО квартала (без подглядывания)
    q_start = A["ot"][mte[0]]
    vol30 = {}
    for sid, (sot, svol) in sym_vol.items():
        w = (sot >= q_start - 30 * 86400_000) & (sot < q_start)
        vol30[sid] = float(svol[w].sum())
    rank = [sid for sid, _ in sorted(vol30.items(),
                                     key=lambda kv: -kv[1])
            if sid not in EXCLUDE]
    t100 = set(rank[:100])
    t100x = set(rank[:102])  # топ-100 без BTC/ETH = топ-102 минус 2
    t50x = set(rank[:52])
    sets = {"ALL": None, "T100": t100, "T100X": t100x, "T50X": t50x}
    for u in UNIS:
        sel = picked if sets[u] is None \
            else [t for t in picked if t[0] in sets[u]]
        trs = eval_pnl(ted, sel)
        RES[u].extend((qq, t) for t in trs)
    log(f"{qq}: сигналов {len(picked)}")
    del Fq, ted, picked
    import gc; gc.collect()

print("\nВСЕЛЕННАЯ | сделок | avg бп | win% | плюс-кварталов", flush=True)
report = {}
for u in UNIS:
    trs = [t for _q, t in RES[u]]
    if not trs:
        print(f"{u}: 0", flush=True)
        continue
    pnl = np.array([t[3] for t in trs])
    qs = {}
    for _q, t in RES[u]:
        qs.setdefault(_q, []).append(t[3])
    good = [q for q, ps in qs.items() if len(ps) >= 30]
    pos = sum(1 for q in good if np.mean(qs[q]) > 0)
    report[u] = {"n": len(trs),
                 "avg_bp": round(float(pnl.mean() * 1e4), 2),
                 "win": round(float((pnl > 0).mean() * 100), 1),
                 "plus_q": f"{pos}/{len(good)}"}
    print(f"{u:6} | {len(trs):6} | {pnl.mean()*1e4:+7.2f} | "
          f"{(pnl>0).mean()*100:4.1f}% | {pos}/{len(good)}", flush=True)

with open("/kaggle/working/t100x_report.json", "w") as f:
    json.dump({"universes": report, "test_quarters": TEST_Q,
               "note": "T100X = топ-100 по 30д обороту без BTC/ETH; "
                       "рейтинг из прошлого, без подглядывания"},
              f, indent=1, default=float)
log("ЗАМЕР ГОТОВ")
