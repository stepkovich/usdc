"""ДИП-ВХОД + ДНЕВНОЙ ТРЕНД, ЛОНГ-ТОЛЬКО (предрегистрация 30.09).

ГИПОТЕЗА — скрещивание двух измеренных половинок: (а) дип-экзамен
подтвердил, что вход лимитом в откат чинит конфликт сигнала со стопом
(лонги -5%: +0.35%/сд, стоп от входа не враг), но плюс концентрирован
в бурных бычьих кварталах, а шорты минусуют всегда; (б) фильтр дневного
тренда в лестнице дал максимальную стабильность (6/8 кварталов).
СБОРКА: только лонги p>0.65, вход лимитом close*(1-dip), dip {3,5,8}%,
сделка только если последняя ЗАКРЫТАЯ дневная свеча выше своей SMA50
(тренд дня вверх); стоп -15% от входа; выход close[t+10] тейкером;
мейкер-вход 0.02%. 8 кварталов linspace, переобучение каждый фолд.

ЗАЯВЛЕННЫЕ КРИТЕРИИ: avg >= +0.10% И плюсовых кварталов >= 5/8 И
нуль-перцентиль >= 95. Плечо: -3% ок 5x; -5% на грани; -8% => 3x.
Шорты исключены заявленно (минусовали во всех конфигах дип-экзамена).
Ничего не деплоируется.
"""
import glob, json, time, zipfile, io, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 3600_000
SLIP, TAKER_FEE, MAKER_FEE = 0.0005, 0.0005, 0.0002
STOP = 0.15
DIPS = [0.03, 0.05, 0.08]
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")

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
btc1h = btc_raw["close"].astype("float32").resample("1h").last().dropna()

frames = []
SYMS = {}
senior_maps = {}
for i, name in enumerate(members, 1):
    sym = name.replace("_30m.csv", "")
    SYMS[i] = sym
    df = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["open_time", "high", "low", "close",
                              "volume", "quote_volume", "taker_buy_base"])
    for c in df.columns:
        if c != "open_time":
            df[c] = df[c].astype("float32")
    df["open_time"] = df["open_time"].astype(np.int64)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_index()
    sd = resample(df, "1D")
    sma_d = sd["close"].rolling(50, min_periods=50).mean()
    trend_d = (sd["close"] > sma_d).values
    close_t_d = sd.index.values.astype("datetime64[ns]").astype(np.int64) \
        // 10**6 + 24 * 3600_000
    senior_maps[sym] = (close_t_d, trend_d)
    rd = resample(df, "1h")
    X = make_feats(rd, btc1h, sym)
    close = rd["close"].values
    fut = np.full(len(rd), np.nan, dtype="float32")
    fut[:-H] = close[H:]
    X["y"] = np.where(np.isnan(fut), np.nan,
                      (fut / close > 1).astype("float32"))
    X["fwd"] = (fut / close - 1).astype("float32")
    X["close"] = close
    X["low"] = rd["low"].values
    X["high"] = rd["high"].values
    X["ot"] = rd["open_time"].values.astype(np.int64)
    X["fi"] = np.uint16(i)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=[c for c in X.columns
                                   if c not in ("close", "fwd")]))
    if i % 100 == 0:
        print(f"признаки {i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
print(f"строк {len(data)}, кварталы {test_q}", flush=True)

FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]

def dip_trades_long(d, dip):
    """Лонги p>0.65: фильтр дневного аптренда + вход лимитом в откат."""
    pools = {int(fi): g.sort_values("ot") for fi, g in d.groupby("fi")}
    rows = []
    last_cd = {}
    for row in d[d["p"] > GATE].sort_values(["sym_id", "ot"]).itertuples():
        if row.ot - last_cd.get(row.sym_id, -10**18) < COOLDOWN_MS:
            continue
        last_cd[row.sym_id] = row.ot
        sym = SYMS[int(row.fi)]
        ct, trend_d = senior_maps.get(sym, (None, None))
        if ct is None:
            continue
        jd = int(np.searchsorted(ct, row.ot, side="right")) - 1
        if jd < 0 or not trend_d[jd]:        # день не в аптренде
            continue
        pool = pools.get(int(row.fi))
        if pool is None:
            continue
        ots = pool["ot"].values
        j = int(np.searchsorted(ots, row.ot))
        if j >= len(ots) or ots[j] != row.ot or j + H >= len(pool):
            continue
        exit_close = pool["close"].values[j + H]
        if np.isnan(exit_close):
            continue
        c = row.close
        entry = c * (1 - dip)
        seg = pool["low"].values[j + 1: j + 1 + H]
        kf = np.nonzero(seg <= entry)[0]
        if len(kf) == 0:
            continue
        jf = kf[0]
        post = seg[jf:]
        stop_px = entry * (1 - STOP) * (1 - SLIP)
        exit_px = (stop_px if post.min() <= entry * (1 - STOP)
                   else exit_close * (1 - SLIP))
        pnl = exit_px / entry - 1 - MAKER_FEE - TAKER_FEE - SLIP
        rows.append({"side": "L", "quarter": row.quarter, "pnl": pnl,
                     "p": row.p, "fill_bar": int(jf), "sym": sym,
                     "ot": int(row.ot)})
    return pd.DataFrame(rows)

def null_pct(te, n_real, seeds=50):
    rng = np.random.default_rng(7)
    sub = te[["close", "fwd"]].dropna()
    if n_real < 20 or len(sub) < n_real:
        return None
    out = []
    for _ in range(seeds):
        s = sub.sample(n=n_real, random_state=rng.integers(1 << 30))
        entry = s["close"].values * (1 + SLIP)
        exit_ = s["close"].values * (1 + s["fwd"].values) * (1 - SLIP)
        out.append((exit_ / entry - 1 - 2 * TAKER_FEE).mean() * 100)
    return out

acc = {d: [] for d in DIPS}
nulls = {d: [] for d in DIPS}
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 1000 or len(trd) < 100_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(trd[FEATS], trd["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    nL = int((d["p"] > GATE).sum())
    for dip in DIPS:
        T = dip_trades_long(d, dip)
        acc[dip].append(T)
        if k >= 6 and len(T):
            nl = null_pct(ted, len(T))
            if nl:
                nulls[dip].append(
                    float(np.mean(np.array(nl) < T["pnl"].mean() * 100) * 100))
    print(f"фолд {qq}: сигналов L={nL} | сделки "
          f"{({int(dp): len(a[-1]) for dp, a in acc.items() if a})} | "
          f"{time.time()-t0:.0f}с", flush=True)

report = {"pre_reg": "лонг-только, дип-вход, дневной аптренд: avg>=+0.10%, "
                     "5/8 кв, нуль>=95; плечо 3% ок 5x, 8% => 3x",
          "folds": test_q, "dips": {}}
for dip in DIPS:
    T = pd.concat(acc[dip], ignore_index=True) if acc[dip] else pd.DataFrame()
    if not len(T):
        report["dips"][str(dip)] = {"n": 0}
        continue
    avg = float(T["pnl"].mean() * 100)
    by_q = {kk: round(v * 100, 3) for kk, v in
            T.groupby("quarter")["pnl"].mean().items()}
    pos_q = sum(1 for v in by_q.values() if v > 0)
    nl_pct = int(np.mean(nulls[dip])) if nulls[dip] else None
    n_sym = int(T["sym"].nunique())
    top5 = T.groupby("sym")["pnl"].sum().sort_values(ascending=False)
    top5_share = float(top5.head(5).sum() / T["pnl"].sum() * 100) \
        if T["pnl"].sum() > 0 else None
    passed = bool(avg >= 0.10 and pos_q >= 5 and (nl_pct or 0) >= 95)
    report["dips"][str(dip)] = {
        "n": int(len(T)), "avg": round(avg, 4),
        "pos_quarters": pos_q, "by_quarter": by_q,
        "null_pct": nl_pct, "symbols": n_sym,
        "top5_pnl_share": None if top5_share is None else round(top5_share, 1),
        "passed": passed}
    T.to_csv(f"/kaggle/working/trades_diptrend{int(dip*100)}.csv", index=False)
    print(f"дип -{int(dip*100)}%: {len(T)} сделок, avg {avg:+.4f}%, "
          f"плюс-кв {pos_q}/8, нуль {nl_pct}%, символов {n_sym}, "
          f"топ-5 дают {None if top5_share is None else round(top5_share)}% "
          f"-> {'СДАЛ' if passed else 'нет'}", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("экзамен дип+тренд завершён", flush=True)
