"""БОЕВОЙ ЭКЗАМЕН СИГНАЛА «БУРЯ ВВЕРХ» (предрегистрация 05.10).

ЭКРАН ПРОЙДЕН (usdc-atrlabel): метка y=1 если цена за 10 баров выросла
> 0.5 ATR, y=0 если упала > 0.5 ATR, болото NaN; порог 0.65 дал
+0.264%/сд после тейкера, 6/8 кварталов, нуль 100%. Теперь проверяем
ВЫЖИВЕТ ЛИ он в боевой механике (стопы убили прошлый кандидат 1ч).

МЕХАНИКА КАК В БОЮ: 30м панель, те же 24 признака, лонг p>0.65 /
шорт p<0.35, cooldown 10 баров. ДВА сценария издержек на одних сигналах:
  TAKER (справка): вход/выход тейкер 0.2% круг;
  MAKER (вердикт): вход GTX по цене сигнального бара (исполняется,
    если следующий бар коснулся; иначе сделки нет, комиссия 0.02%),
    выход тейкер.
СТОПЫ: аварийный -15% ОТ ВХОДА по экстремумам баров удержания, плюс
чувствительность 0/20/25/30% (столбцами на тех же сделках).

ЗАЯВЛЕННЫЙ КРИТЕРИЙ СДАЧИ (мейкер, все сделки, стоп -15%): avg >= +0.10%
И плюсовых кварталов >= 5/8 И нуль-перцентиль >= 95. Отдельно: лонг,
шорт, чувствительность к стопу.
"""
import glob, json, time, zipfile, io, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000
SLIP, TAKER_FEE, MAKER_FEE = 0.0005, 0.0005, 0.0002
STOPS = [None, 0.15, 0.20, 0.25, 0.30]
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
print("монет:", len(members), flush=True)
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
    """Сигналы обеих сторон с сырьём для всех сценариев сразу."""
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
            rows.append({"side": side, "quarter": row.quarter, "c": c,
                         "exit_close": exit_close,
                         "seg_lo": float(seg_lo.min()),
                         "seg_hi": float(seg_hi.max()),
                         "filled": bool(filled)})
    return pd.DataFrame(rows)

def pnl_col(T, stop, scen):
    """Заполняет колонку pnl_<scen>_<stop> на тех же сделках."""
    out = np.full(len(T), np.nan)
    c = T["c"].values.astype("float64")
    ex = T["exit_close"].values.astype("float64")
    lo, hi = T["seg_lo"].values, T["seg_hi"].values
    long_ = (T["side"] == 1).values
    for mask, sign in ((long_, 1), (~long_, -1)):
        m = mask & (T["filled"].values.astype(bool)
                    if scen == "maker" else np.ones(len(T), bool))
        if not m.any():
            continue
        if stop is None:
            hit = np.zeros(int(m.sum()), bool)
            stop_px = np.zeros(int(m.sum()))
        else:
            hit = (lo[m] <= c[m] * (1 - stop)) if sign == 1 \
                else (hi[m] >= c[m] * (1 + stop))
            stop_px = c[m] * (1 - sign * stop) * (1 - sign * SLIP)
        exit_px = np.where(hit, stop_px, ex[m] * (1 - sign * SLIP))
        if scen == "taker":
            entry = c[m] * (1 + sign * SLIP)
            out[m] = sign * (exit_px - entry) / entry - 2 * TAKER_FEE
        else:
            entry = c[m]
            out[m] = sign * (exit_px - entry) / entry \
                - MAKER_FEE - TAKER_FEE - SLIP
    return out

def null_pct(te, n_real, seeds=50):
    rng = np.random.default_rng(7)
    sub = te[["fwd"]].dropna()
    if n_real < 20 or len(sub) < n_real:
        return None
    out = []
    for _ in range(seeds):
        s = sub.sample(n=n_real, random_state=rng.integers(1 << 30))
        out.append((s["fwd"].values.astype("float64") - FEE2).mean() * 100)
    return out

FEE2 = 2 * TAKER_FEE
all_T = []
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
    print(f"фолд {qq}: лонг {(T['side']==1).sum()} / шорт "
          f"{(T['side']==-1).sum()} | {time.time()-t0:.0f}с", flush=True)

T = pd.concat(all_T, ignore_index=True)
print("всего сигналов:", len(T), flush=True)
for stop in STOPS:
    key = "none" if stop is None else str(int(stop * 100))
    T[f"taker_{key}"] = pnl_col(T, stop, "taker")
    T[f"maker_{key}"] = pnl_col(T, stop, "maker")

def null_pct2(ted, n_real, seeds=50):
    rng = np.random.default_rng(7)
    sub = ted[["fwd"]].dropna()
    if n_real < 20 or len(sub) < n_real:
        return None
    out = []
    for _ in range(seeds):
        s = sub.sample(n=n_real, random_state=rng.integers(1 << 30))
        out.append((s["fwd"].values.astype("float64") - FEE2).mean() * 100)
    return out

report = {"pre_reg": "сдал: maker_15 все avg>=+0.10%, 5/8 кв, нуль>=95",
          "stops": {}}
verdict_pass = False
for stop in STOPS:
    key = "none" if stop is None else str(int(stop * 100))
    row = {}
    for scen in ("taker", "maker"):
        col = f"{scen}_{key}"
        s = T.dropna(subset=[col])
        avg = float(s[col].mean() * 100)
        by_q = {kk: round(v * 100, 3) for kk, v in
                s.groupby("quarter")[col].mean().items()}
        pos_q = sum(1 for v in by_q.values() if v > 0)
        Ls = s[s["side"] == 1][col].mean() * 100
        Ss = s[s["side"] == -1][col].mean() * 100
        row[scen] = {"n": int(len(s)), "avg": round(avg, 4),
                     "pos_quarters": pos_q, "by_quarter": by_q,
                     "long_avg": round(float(Ls), 4),
                     "short_avg": round(float(Ss), 4)}
    # нуль-тест: стоп -15, мейкер, все сделки, два последних фолда
    if key == "15":
        nulls = []
        for qq in test_q[-2:]:
            Tq = T[(T["quarter"] == qq)]
            col = "maker_15"
            s_q = Tq.dropna(subset=[col])
            nl = null_pct2(data[data["quarter"] == qq], len(s_q))
            if nl and len(s_q):
                nulls.append(float(np.mean(np.array(nl)
                                           < s_q[col].mean() * 100) * 100))
        row["null_pct"] = int(np.mean(nulls)) if nulls else None
    else:
        row["null_pct"] = None
    if key == "15":
        v = row["maker"]
        nl = row["null_pct"]
        verdict_pass = bool(v["avg"] >= 0.10 and v["pos_quarters"] >= 5
                            and (nl or 0) >= 95)
        row["passed"] = verdict_pass
    report["stops"][key] = row
    mk = row["maker"]
    print(f"стоп {key:>4}: тейкер {row['taker']['avg']:+.4f}% | мейкер "
          f"{mk['avg']:+.4f}% (лонг {mk['long_avg']:+.3f} / шорт "
          f"{mk['short_avg']:+.3f}), плюс-кв {mk['pos_quarters']}/8, "
          f"нуль {row['null_pct']}% "
          f"{'<== ВЕРДИКТ' if key == '15' else ''}", flush=True)
print("ВЕРДИКТ:", "СДАЛ" if verdict_pass else "НЕ СДАЛ", flush=True)
report["verdict"] = "СДАЛ" if verdict_pass else "НЕ СДАЛ"
T.to_csv("/kaggle/working/trades_storm.csv", index=False)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("БОЕВОЙ ЭКЗАМЕН БУРИ завершён", flush=True)
