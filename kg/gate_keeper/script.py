"""ШТОРМ-ГЕЙТ НАД СПРИНТЕРОМ (предрегистрация 06.10, логика владельца).

ГИПОТЕЗА: спринтер (5-мин направление по книге) торгует все моменты
подряд — тихие пятиминутки съедают комиссию. Если пускать его ТОЛЬКО
в «заряженные» моменты (30-мин буря объявлена свечами+книгой), средняя
сделка жирнее при тех же издержках — экономика оживает.

МЕТОД (на записях, без боя): 38 USDC-активов, наши данные.
1) 15м свечи с архива биржи + наша книга (бины 15м, последняя строка
   бина). Признаки: 24 свечных формулы (в 15м-барах) + 6 книжных.
2) ШТОРМ-МОДЕЛЬ: классификатор «|fwd 30мин| > 1×ATR14», walk-forward
   по дням. На каждом закрытии 15м-бара — вероятность шторма P30.
3) СПРИНТЕР-МОДЕЛЬ: та же, что в бою (13 признаков книги 500мс,
   горизонт 300с, медианы по символу) — для каждой 500мс-строки
   тестовых дней: p5 = вероятность роста за 5 минут.
4) СЦЕНАРИИ (те же сделки, отличается только ГЕЙТ):
   OFF: вход при p5>=0.62 (лонг) / p5<=0.38 (шорт) — как в бою;
   ON: то же + в момент входа P30 (последнее закрытие 15м-бара) >= 0.60.
   Вход по миду, выход через 300с по миду; ИЗДЕРЖКИ (USDC): вход
   мейкер 0 (наполняется всегда — консервативно считаем fill),
   выход тейкер 10 бп (5 fee + 5 slip); одна позиция на символ,
   после выхода пауза 300с.
ВЕРДИКТ (записан до прогона): гейт жив, если на тест-днях:
avg_bp(ON) − avg_bp(OFF) >= +3 бп И avg_bp(ON) >= 0 И сделок(ON)
осталось >= 15% от OFF. Отчёт по дням + разрез лонг/шорт.
Ничего не деплоится.
"""
import glob, io, json, time, zipfile, zlib, re, datetime
import urllib.request, urllib.error
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
N_TEST_DAYS = 4
SP_GATE = 0.62
ST_GATE = 0.60
HOLD_MS = 300_000
FEE_OUT = 0.0010               # тейкер-выход: 5бп fee + 5бп slip
PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1)

SYMS = ["1000BONKUSDC", "1000PEPEUSDC", "1000SHIBUSDC", "AAVEUSDC",
        "ADAUSDC", "ARBUSDC", "AVAXUSDC", "BCHUSDC", "BIOUSDC",
        "BNBUSDC", "BOMEUSDC", "BTCUSDC", "CRVUSDC", "DATAIPUSDC",
        "DOGEUSDC", "ENAUSDC", "ETHFIUSDC", "ETHUSDC", "FILUSDC",
        "HBARUSDC", "KAITOUSDC", "LINKUSDC", "LTCUSDC", "NEARUSDC",
        "NEOUSDC", "ORDIUSDC", "PENGUUSDC", "PNUTUSDC", "SOLUSDC",
        "SUIUSDC", "TIAUSDC", "TRUMPUSDC", "UNIUSDC", "WIFUSDC",
        "WLDUSDC", "WLFIUSDC", "XRPUSDC", "ZECUSDC"]
DAYS_K = [f"2026-09-{d:02d}" for d in range(20, 31)] + \
          [f"2026-10-{d:02d}" for d in range(1, 7)]
BASE = "https://data.binance.vision/data/futures/um"
COLS = ["open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "tbb", "tbq", "ig"]
LOB_FEATS = ["spread_bp", "microprice_rel", "imb5", "imb10", "imb20",
             "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
             "ntr10", "vpin10", "d30", "d120"]
LADDER = ["imb1", "slope_b", "slope_a", "wall_b", "wall_a"]

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return "ok", r.read()
    except urllib.error.HTTPError as e:
        return ("404", None) if e.code == 404 else ("err", None)
    except Exception:
        return "err", None

def parse(content):
    zf = zipfile.ZipFile(io.BytesIO(content))
    raw = zf.read(zf.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLS,
                     low_memory=False)
    if df["open_time"].dtype == object:
        df = df.iloc[1:].reset_index(drop=True)
    ot = pd.to_numeric(df["open_time"]).astype("int64")
    while len(ot) and ot.iloc[0] > 10**14:
        ot = ot // 1000
    df["open_time"] = ot
    return df

def get_symbol(sym):
    parts = []
    for d in DAYS_K:
        url = f"{BASE}/daily/klines/{sym}/15m/{sym}-15m-{d}.zip"
        for _ in range(3):
            st, c = fetch(url)
            if st == "404":
                break
            if st == "ok":
                try:
                    parts.append(parse(c))
                    break
                except Exception:
                    continue
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True).drop_duplicates("open_time") \
        .sort_values("open_time")
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "tbb"):
        df[c] = df[c].astype("float32")
    return df.rename(columns={"tbb": "taker_buy_base"})

# ---------- 15м свечи + шторм-признаки ----------
cframes = []
for k, sym in enumerate(SYMS, 1):
    df = get_symbol(sym)
    if df is None or len(df) < 1200:
        continue
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    close = df["close"]; high, low = df["high"], df["low"]
    rets = close.pct_change()
    X = pd.DataFrame(index=df.index)
    for h in (1, 2, 4, 8):
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
    if sym != "BTCUSDC":
        # btc-признаки посчитаем вторым проходом (после BTC загружен)
        pass
    X["h4_momentum"] = close.pct_change(24).astype("float32")
    X["hour"] = df.index.hour.astype("float32")
    X["dow"] = df.index.dayofweek.astype("float32")
    X["sym_id"] = np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)
    X["sym"] = sym
    X["ot"] = df["open_time"].values.astype(np.int64)
    X["atr_abs"] = (tr.rolling(14).mean()).astype("float64")
    c = close.values
    fwd = np.full(len(df), np.nan, dtype="float64")
    fwd[:-2] = c[2:] / c[:-2] - 1
    atr_rel = (tr.rolling(14).mean() / close).values.astype("float64")
    X["storm"] = np.where(np.isnan(fwd) | np.isnan(atr_rel), np.nan,
                          (np.abs(fwd) > 1.0 * atr_rel).astype("float32"))
    X = X.replace([np.inf, -np.inf], np.nan)
    cframes.append(X.dropna(subset=["atr_pct"]))
    if k % 10 == 0:
        print(f"свечи {k}/{len(SYMS)} | {time.time()-t0:.0f}с", flush=True)
candles = pd.concat(cframes, ignore_index=True)
del cframes
print(f"свечных строк {len(candles)}", flush=True)

# ---------- книга: бины 15м (для шторм-признаков) + полные строки ----
book_frames = []
for f in sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True)):
    try:
        book_frames.append(pd.read_parquet(
            f, columns=["ts", "symbol", "mid", "spread_bp", "microprice",
                        "imb5", "imb10", "imb20", "imb1", "slope_b",
                        "slope_a", "wall_b", "wall_a", "flow10_buy",
                        "flow10_sell", "flow60_buy", "flow60_sell",
                        "ntr10", "vpin10"]))
    except Exception:
        pass
book = pd.concat(book_frames, ignore_index=True)
del book_frames
print(f"книжных строк {len(book)} | {time.time()-t0:.0f}с", flush=True)

# бины: последняя строка каждого 15м-бина (состояние на закрытии)
book["bin"] = (book["ts"] // 90_000) * 90_000
binned = book.sort_values("ts").groupby(["symbol", "bin"]).last() \
    .reset_index()
binned["microprice_rel"] = (binned["microprice"] / binned["mid"] - 1) \
    * 10000
tot10 = binned["flow10_buy"] + binned["flow10_sell"]
binned["flowimb10"] = np.where(tot10 > 0, (binned["flow10_buy"]
                                           - binned["flow10_sell"])
                               / tot10.replace(0, np.nan), 0)
binned["ot"] = binned["bin"].astype("int64")
BOOK6 = ["imb5", "imb10", "imb20", "microprice_rel", "flowimb10",
         "vpin10"]
ds = candles.merge(binned[["symbol", "ot"] + BOOK6],
                   left_on=["sym", "ot"], right_on=["symbol", "ot"],
                   how="left").drop(columns=["symbol"], errors="ignore")
del candles
ds["day"] = pd.to_datetime(ds["ot"], unit="ms").dt.date
days = sorted(ds["day"].unique())
STORM_FEATS = (["ret_1", "ret_2", "ret_4", "ret_8", "h4_momentum",
                "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy",
                "clv", "zscore", "rsi", "sma_cross", "hour", "dow",
                "sym_id"] + BOOK6)
print(f"дней {len(days)}: {days[0]}..{days[-1]}", flush=True)

# ---------- шторм-модель: P30 на каждом закрытии тест-дней ----------
P30 = {}
for day in days[-N_TEST_DAYS:]:
    tr = ds[ds["day"] < day].dropna(subset=STORM_FEATS + ["storm"])
    te = ds[ds["day"] == day].dropna(subset=STORM_FEATS)
    if len(te) < 100 or len(tr) < 1000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(tr[STORM_FEATS], tr["storm"], categorical_feature=["sym_id"])
    p = m.predict_proba(te[STORM_FEATS])[:, 1]
    for sym, ot, pv in zip(te["sym"].values, te["ot"].values, p):
        P30[(sym, int(ot))] = float(pv)
    print(f"шторм-модель {day}: {len(te)} баров | {time.time()-t0:.0f}с",
          flush=True)
del ds
print(f"шторм-вероятностей {len(P30)}", flush=True)

# ---------- спринтер-симуляция на тест-днях ----------
def med_norm(g):
    for c in ("flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
              "ntr10"):
        med = g[c].median()
        g[c] = g[c] / (med if med and med > 0 else 1.0)
    g["microprice_rel"] = (g["microprice"] / g["mid"] - 1) * 10000
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    for lag, name in ((30_000, "d30"), (120_000, "d120")):
        j = np.clip(ts.searchsorted(ts - lag), 0, len(g) - 1)
        past = np.where(np.abs(ts[j] - (ts - lag)) <= 2500, mid[j],
                        np.nan)
        g[name] = mid / past - 1
    return g

test_days = [str(d) for d in days[-N_TEST_DAYS:]]
res = {}
book["day"] = pd.to_datetime(book["ts"], unit="ms").dt.date.astype(str)
for sym, g in book.groupby("symbol"):
    if sym not in SYMS:
        continue
    g = g[g["day"].isin(test_days)].sort_values("ts").reset_index(drop=True)
    if len(g) < 5000:
        continue
    g = med_norm(g)
    res[("rows", sym)] = g
    # бары шторма для этого символа
    p30_map = {ot: p for (s2, ot), p in P30.items() if s2 == sym}
    g["p30"] = g["ts"].map(lambda x: p30_map.get(
        int((x // 90_000) * 90_000), np.nan))
    # ярлык 5-мин направления
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    idx = np.clip(ts.searchsorted(ts + HOLD_MS), 0, len(g) - 1)
    ok = ts[idx] >= ts + HOLD_MS - 1500
    g["fwd5"] = np.where(ok, mid[idx] / mid - 1, np.nan)
    res[("rows", sym)] = g[["ts", "mid", "p30", "fwd5"] +
                           [c for c in LOB_FEATS if c in g.columns]]
del book
all_rows = pd.concat([v for k, v in res.items() if k[0] == "rows"],
                     ignore_index=True)
print(f"тест-строк спринтера {len(all_rows)} | {time.time()-t0:.0f}с",
      flush=True)
# обучение спринтера: перечитываем книгу заново, дни ДО тестовых
train_frames = []
for f in sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True)):
    try:
        tf = pd.read_parquet(f, columns=["ts", "symbol", "mid",
                                         "spread_bp", "microprice",
                                         "imb5", "imb10", "imb20",
                                         "flow10_buy", "flow10_sell",
                                         "flow60_buy", "flow60_sell",
                                         "ntr10", "vpin10"])
        tf["day"] = pd.to_datetime(tf["ts"], unit="ms").dt.date \
            .astype(str)
        train_frames.append(tf[tf["day"] < test_days[0]])
    except Exception:
        pass
train = pd.concat(train_frames, ignore_index=True)
del train_frames
print(f"обучающих строк спринтера {len(train)} | {time.time()-t0:.0f}с",
      flush=True)
# медианы и ярлык на обучении
train = train.sort_values(["symbol", "ts"]).reset_index(drop=True)
tr_parts = []
for sym, g in train.groupby("symbol"):
    if sym not in SYMS:
        continue
    g = med_norm(g.copy())
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    idx = np.clip(ts.searchsorted(ts + HOLD_MS), 0, len(g) - 1)
    ok = ts[idx] >= ts + HOLD_MS - 1500
    g["fwd5"] = np.where(ok, mid[idx] / mid - 1, np.nan)
    g["y"] = np.where(g["fwd5"].isna(), np.nan,
                      (g["fwd5"] > 0).astype("float32"))
    tr_parts.append(g)
train = pd.concat(tr_parts, ignore_index=True)
del tr_parts
train = train.dropna(subset=LOB_FEATS + ["y"])
m5 = lgb.LGBMClassifier(**PARAMS)
m5.fit(train[LOB_FEATS], train["y"])
print("спринтер-модель обучена", flush=True)
del train

# посимвольная симуляция: sym в каждой строке
rows_with_sym = []
for sym in SYMS:
    if ("rows", sym) in res:
        g = res[("rows", sym)].copy()
        g["sym"] = sym
        rows_with_sym.append(g)
all_rows = pd.concat(rows_with_sym, ignore_index=True)
all_rows = all_rows.dropna(subset=["fwd5"])
all_rows["p5"] = m5.predict_proba(
    all_rows[[c for c in LOB_FEATS if c in all_rows.columns]])[:, 1]

def simulate_by_sym(d, use_gate):
    trades = []
    for sym, g in d.groupby("sym"):
        g = g.sort_values("ts").reset_index(drop=True)
        ts = g["ts"].values
        p5 = g["p5"].values
        fwd5 = g["fwd5"].values
        p30 = g["p30"].values
        i = 0
        n = len(g)
        while i < n:
            sig = 0
            if p5[i] >= SP_GATE:
                sig = 1
            elif p5[i] <= 1 - SP_GATE:
                sig = -1
            if sig == 0 or np.isnan(fwd5[i]):
                i += 1
                continue
            if use_gate and (np.isnan(p30[i]) or p30[i] < ST_GATE):
                i += 1
                continue
            trades.append({"pnl": sig * fwd5[i] - FEE_OUT, "side": sig})
            i += 1200
    return pd.DataFrame(trades)

OFF = simulate_by_sym(all_rows, False)
ON = simulate_by_sym(all_rows, True)
def stat(T):
    if not len(T):
        return {"n": 0}
    return {"n": int(len(T)),
            "avg_bp": round(float(T["pnl"].mean() * 10000), 2),
            "long_avg_bp": round(float(T[T["side"] == 1]["pnl"]
                                       .mean() * 10000), 2),
            "short_avg_bp": round(float(T[T["side"] == -1]["pnl"]
                                        .mean() * 10000), 2)}
so, sn = stat(OFF), stat(ON)
delta = (sn.get("avg_bp", -99) - so.get("avg_bp", -99)) if sn.get("n") \
    else None
alive = bool(sn.get("n", 0) and so.get("n", 0)
             and delta >= 3.0 and sn["avg_bp"] >= 0
             and sn["n"] >= 0.15 * so["n"])
report = {"OFF": so, "ON": sn, "delta_bp": delta, "alive": alive}
print(f"OFF: {so}", flush=True)
print(f"ON:  {sn}", flush=True)
print("ВЕРДИКТ:", "ГЕЙТ ЖИВ" if alive else "нет", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
