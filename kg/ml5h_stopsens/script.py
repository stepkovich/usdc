"""ЧУВСТВИТЕЛЬНОСТЬ К ШИРИНЕ СТОПА (предрегистрация 30.09, до запуска).

ПОВОД: боевой экзамен 1ч@0.65 провалился из-за аварийного стопа -15%
(4.5% лонгов зацепило, каждый -15.2% вместо -3..-9% по времени; сдвиг
края -0.22%/сделку). Сигналы и модель те же, что прошли экран (кварталы
без стопов совпадают с экраном бит-в-бит).

ИЗМЕРЕНИЕ (не деплой): конфиг 1ч@0.65, лонг p>0.65 / шорт p<0.35,
cooldown 10 баров, те же 8 фолдов, ОДНИ И ТЕ ЖЕ сигналы и наполнения —
меняется только ширина аварийного стопа: БЕЗ СТОПА / -20% / -25% / -30%.
Сценарии: тейкер (справка) и мейкер (вход GTX с испытанием наполнения
следующим баром, выход тейкер).

ОГОВОРКА О ПЛЕЧЕ (заявлена до результата): при плече 5x ликвидация
примерно -20% против позиции — стопы -25/-30% имеют смысл только при
плече 3x (ликвидация ~-33%); сайзинг от потери сохраняет долларовый
риск: позиция = бюджет потери / ширина стопа, т.е. широкий стоп = чуть
меньше размер, тот же 0.10 доллара риска.

ЗАЯВЛЕННОЕ ПРАВИЛО ЧТЕНИЯ: ширина — КАНДИДАТ, если мейкер-все
avg >= +0.10% И плюсовых кварталов >= 5/8. Нуль-тест здесь не
пересчитывается (измерение); он будет пересчитан полным экзаменом на
выбранной владельцем ширине. Ничего не деплоится.
"""
import glob, json, time, zipfile, io, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 3600_000
SLIP = 0.0005
TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
STOPS = [None, 0.20, 0.25, 0.30]
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
    df = df.sort_index()
    rd = resample(df, "1h")
    X = make_feats(rd, btc1h, sym)
    close = rd["close"].values
    fut = np.full(len(rd), np.nan, dtype="float32")
    fut[:-H] = close[H:]
    X["y"] = np.where(np.isnan(fut), np.nan,
                      (fut / close > 1).astype("float32"))
    X["close"] = close
    X["low"] = rd["low"].values
    X["high"] = rd["high"].values
    X["ot"] = rd["open_time"].values.astype(np.int64)
    X["fi"] = np.uint16(i)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=[c for c in X.columns
                                   if c != "close"]))
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

def collect_trades(d):
    """Сигналы с обеих сторон; сырьё для всех ширин стопа сразу."""
    pools = {int(fi): g.sort_values("ot") for fi, g in d.groupby("fi")}
    rows = []
    for side in ("L", "S"):
        sel = d[(d["p"] > GATE) if side == "L" else (d["p"] < 1 - GATE)] \
            .sort_values(["sym_id", "ot"])
        last = {}
        for row in sel.itertuples():
            if row.ot - last.get(row.sym_id, -10**18) < COOLDOWN_MS:
                continue
            last[row.sym_id] = row.ot
            pool = pools.get(int(row.fi))
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
            seg_lo = pool["low"].values[j + 1: j + 1 + H].min()
            seg_hi = pool["high"].values[j + 1: j + 1 + H].max()
            nxt_lo = pool["low"].values[j + 1]
            nxt_hi = pool["high"].values[j + 1]
            filled = (nxt_lo <= c) if side == "L" else (nxt_hi >= c)
            rows.append({"side": side, "quarter": row.quarter, "c": c,
                         "exit_close": exit_close, "seg_lo": seg_lo,
                         "seg_hi": seg_hi, "filled": filled})
    return pd.DataFrame(rows)

all_T = []
for qq in test_q:
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 1000 or len(trd) < 100_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(trd[FEATS], trd["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    T = collect_trades(d)
    all_T.append(T)
    print(f"фолд {qq}: лонг {(T['side']=='L').sum()} / шорт "
          f"{(T['side']=='S').sum()} | {time.time()-t0:.0f}с", flush=True)

T = pd.concat(all_T, ignore_index=True)
print(f"всего сигналов {len(T)}", flush=True)

def pnl_for(T, stop, scen):
    out = np.full(len(T), np.nan)
    c = T["c"].values
    ex = T["exit_close"].values
    lo, hi = T["seg_lo"].values, T["seg_hi"].values
    filled = T["filled"].values.astype(bool)
    long_ = (T["side"] == "L").values
    for mask, sign in ((long_, 1), (~long_, -1)):
        m = mask & (filled if scen == "maker" else np.ones(len(T), bool))
        if not m.any():
            continue
        if stop is None:
            hit = np.zeros(m.sum(), bool)
            stop_px = np.zeros(m.sum())
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

report = {"pre_reg": "кандидат: мейкер-все avg>=+0.10% и 5/8 кварталов "
                     "на ширине; плечо: -20% ок при 5x, -25/-30% => 3x; "
                     "измерение, не деплой; нуль-тест вне этого прогона",
          "stops": {}}
for stop in STOPS:
    key = "none" if stop is None else str(int(stop * 100))
    T[f"taker_{key}"] = pnl_for(T, stop, "taker")
    T[f"maker_{key}"] = pnl_for(T, stop, "maker")
    row = {}
    for scen in ("taker", "maker"):
        col = f"{scen}_{key}"
        s = T.dropna(subset=[col])
        avg = float(s[col].mean() * 100)
        by_q = {kk: round(v * 100, 3) for kk, v in
                s.groupby("quarter")[col].mean().items()}
        pos_q = sum(1 for v in by_q.values() if v > 0)
        Ls = s[s["side"] == "L"][col].mean() * 100
        Ss = s[s["side"] == "S"][col].mean() * 100
        row[scen] = {"n": int(len(s)), "avg": round(avg, 4),
                     "pos_quarters": pos_q, "by_quarter": by_q,
                     "long_avg": round(float(Ls), 4),
                     "short_avg": round(float(Ss), 4)}
    row["candidate"] = bool(row["maker"]["avg"] >= 0.10
                            and row["maker"]["pos_quarters"] >= 5)
    report["stops"][key] = row
    print(f"стоп {key:>4}: тейкер {row['taker']['avg']:+.4f}% "
          f"({row['taker']['pos_quarters']}/8) | мейкер "
          f"{row['maker']['avg']:+.4f}% ({row['maker']['pos_quarters']}/8) "
          f"лонг {row['maker']['long_avg']:+.4f}% шорт "
          f"{row['maker']['short_avg']:+.4f}% -> "
          f"{'КАНДИДАТ' if row['candidate'] else 'нет'}", flush=True)
T.to_csv("/kaggle/working/trades_stopsens.csv", index=False)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("измерение завершено", flush=True)
