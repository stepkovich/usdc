"""ПОДТВЕРЖДЕНИЕ «КУСОЧКА» (08.10, отложенное правило). Первый прогон
(тест 02-05.10) дал Δ+6.5 бп, НО 1/4 плюс-дней и день-локомотив 02.10.
Правило из шапки первого прогона: перед вносом — подтверждение на
СВЕЖИХ днях, которых прогон не видел. Этот прогон = тест 06-08.10
(последние 3 полных дня записи), обучение на всём до них.
ПОДТВЕРЖДЕНО, если Δ(B−A) >= +3 бп И avg(B) > 0. Подтвердится —
вносим книгу в Бурю для 37 монет с недельным наблюдением в бою.
"""
"""(архив шапки) БУРЯ + КНИГА: ПОЛНЫЙ А/Б-ЭКЗАМЕН (предрегистрация 07.10, «кусочек»).

ВОПРОС: находка выходных — книга заявок улучшает ПОИСК бури на +5пп
(4/4 дня, 30-мин окно). Превращается ли это в ДЕНЬГИ у боевой модели-
бури (30м бары, окно 5 часов)? Если да — первый живой кусочек комбо.

ДИЗАЙН: 37 USDC-активов (где пишем книгу). Свечи 30м с архива биржи
(с 18.09 — прогрев 480-барных окон); книга — наша запись (датасет
usdc-lob-features), бин 30м, ПОСЛЕДНЯЯ строка бина = состояние на
закрытии бара, строго без подглядывания.
МЕТКА как у боевой: |fwd за 10 баров| > 0.5 × ATR14(30м).
A: 24 свечных признака (как в бою). B: A + 6 книжных (imb5, imb10,
imb20, microprice_rel, flowimb10, vpin10).
ЭКЗАМЕН: walk-forward, обучение на всех днях до тестовых, тест =
последние 4 полных дня записи. Трейд-сим: порог 0.65, ОБЕ стороны,
вход мейкер 0 (USDC промо; наполняется, если следующий бар коснулся
цены входа), стоп −15% от входа по экстремумам баров, выход close[t+10]
тейкером 10 бп; cooldown 10 баров.
ВЕРДИКТ (записан до прогона): B ставим в кандидаты, если avg_bp(B) −
avg_bp(A) >= +3 бп на пуле тест-дней И avg_bp(B) > 0 И n(B) >= 100.
Оговорка: книга есть только ~11 дней и один режим рынка — это
кандидат, не приговор; перед боем потребуется подтверждение на
следующей неделе записи. Ничего не деплоится.
"""
import glob, io, json, time, zipfile, zlib, datetime
import urllib.request, urllib.error
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000
TAKER_OUT = 0.0010
STOP = 0.15
N_TEST_DAYS = 3
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")

SYMS = ["1000BONKUSDC", "1000PEPEUSDC", "1000SHIBUSDC", "AAVEUSDC",
        "ADAUSDC", "ARBUSDC", "AVAXUSDC", "BCHUSDC", "BIOUSDC",
        "BNBUSDC", "BOMEUSDC", "BTCUSDC", "CRVUSDC", "DATAIPUSDC",
        "DOGEUSDC", "ENAUSDC", "ETHFIUSDC", "ETHUSDC", "FILUSDC",
        "HBARUSDC", "KAITOUSDC", "LINKUSDC", "LTCUSDC", "NEARUSDC",
        "NEOUSDC", "ORDIUSDC", "PENGUUSDC", "PNUTUSDC", "SOLUSDC",
        "SUIUSDC", "TIAUSDC", "TRUMPUSDC", "UNIUSDC", "WIFUSDC",
        "WLDUSDC", "WLFIUSDC", "XRPUSDC", "ZECUSDC"]
DAYS_K = [f"2026-09-{d:02d}" for d in range(18, 31)] + \
          [f"2026-10-{d:02d}" for d in range(1, 8)]
BASE = "https://data.binance.vision/data/futures/um"
COLS = ["open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "tbb", "tbq", "ig"]

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
        url = f"{BASE}/daily/klines/{sym}/30m/{sym}-30m-{d}.zip"
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

btc = get_symbol("BTCUSDC")
btc.index = pd.to_datetime(btc["open_time"], unit="ms")
btc_c = btc["close"].astype("float64")

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
    if sym != "BTCUSDC":
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

cframes = []
for k, sym in enumerate(SYMS, 1):
    df = get_symbol(sym)
    if df is None or len(df) < 600:
        continue
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    X, tr = make_feats(df, btc_c, sym)
    close = df["close"].values
    atr = (tr.rolling(14).mean() / close).values.astype("float64")
    fut = np.full(len(df), np.nan, dtype="float64")
    fut[:-H] = close[H:]
    fwd = fut / close - 1
    thr = 0.5 * np.nan_to_num(atr, nan=np.nan)
    X["y"] = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                      (np.abs(fwd) > thr).astype("float32"))
    X["fwd"] = fwd.astype("float32")
    X["close"] = close
    X["low"] = df["low"].values
    X["high"] = df["high"].values
    X["ot"] = df["open_time"].values.astype(np.int64)
    X["sym"] = sym
    X = X.replace([np.inf, -np.inf], np.nan)
    cframes.append(X.dropna(subset=["atr_pct"]))
    if k % 10 == 0:
        print(f"свечи {k}/{len(SYMS)} | {time.time()-t0:.0f}с", flush=True)
candles = pd.concat(cframes, ignore_index=True)
del cframes
print(f"свечных строк {len(candles)}", flush=True)

# КНИГА: сразу сворачиваем в 30м бины при чтении — иначе 23М+ строк
# рвут память CPU-инстанса (виновник ERROR после 3ч вчера)
book_parts = []
for f in sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True)):
    try:
        bf = pd.read_parquet(
            f, columns=["ts", "symbol", "mid", "microprice", "imb5",
                        "imb10", "imb20", "flow10_buy", "flow10_sell",
                        "vpin10"])
        bf["bin"] = (bf["ts"] // 1_800_000) * 1_800_000
        g = bf.sort_values("ts").groupby(["symbol", "bin"]).last()             .reset_index()
        book_parts.append(g)
        del bf
    except Exception:
        pass
book = pd.concat(book_parts, ignore_index=True)
del book_parts
print(f"книжных строк {len(book)} | {time.time()-t0:.0f}с", flush=True)
binned = book.rename(columns={"bin": "bin"})
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
del candles, book, binned
ds["day"] = pd.to_datetime(ds["ot"], unit="ms").dt.date
days = sorted(ds["day"].unique())
test_days = days[-N_TEST_DAYS:]
FEATS_A = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
           "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
           "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
           "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
           "hour", "dow", "sym_id"]
FEATS_B = FEATS_A + BOOK6
print(f"дней {len(days)}: {days[0]}..{days[-1]}, тест {test_days}",
      flush=True)

def collect(d):
    pools = {s: g.sort_values("ot") for s, g in d.groupby("sym")}
    rows = []
    for side in (1, -1):
        sel = d[(d["p"] > GATE) if side == 1 else (d["p"] < 1 - GATE)] \
            .sort_values(["sym_id", "ot"])
        last = {}
        for row in sel.itertuples():
            if row.ot - last.get(row.sym_id, -10**18) < COOLDOWN_MS:
                continue
            last[row.sym_id] = row.ot
            pool = pools.get(row.sym)
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
            if not filled:
                continue
            hit = (seg_lo.min() <= c * (1 - STOP)) if side == 1 \
                else (seg_hi.max() >= c * (1 + STOP))
            exit_px = (c * (1 - STOP) * 0.9995) if (hit and side == 1) \
                else ((c * (1 + STOP) * 1.0005)
                      if (hit and side == -1)
                      else exit_close * (1 - side * 0.0005))
            pnl = side * (exit_px / c - 1) - TAKER_OUT
            rows.append({"pnl": pnl, "day": row.day, "side": side})
    return pd.DataFrame(rows)

report = {"pre_reg": "B жив: avg_bp(B)-avg_bp(A)>=+3, avg_bp(B)>0, "
                     "n(B)>=100; 11 дней один режим — кандидат",
          "days": str(days[0]) + ".." + str(days[-1])}
for name, feats in (("A", FEATS_A), ("B", FEATS_B)):
    tr = ds[ds["day"] < test_days[0]].dropna(
        subset=[f for f in feats if f != "imb5"] + ["y"])
    tr = tr.dropna(subset=["y"])
    if name == "B":
        tr = tr.dropna(subset=BOOK6)      # B учится только где есть книга
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(tr[feats], tr["y"], categorical_feature=["sym_id"])
    te = ds[ds["day"].isin(test_days)].copy()
    if name == "B":
        te = te.dropna(subset=BOOK6)
    te["p"] = m.predict_proba(te[feats])[:, 1]
    T = collect(te)
    if not len(T):
        report[name] = {"n": 0}
        continue
    avg_bp = float(T["pnl"].mean() * 10000)
    by_day = {str(k): round(v * 10000, 1) for k, v in
              T.groupby("day")["pnl"].mean().items()}
    pos_days = sum(1 for v in by_day.values() if v > 0)
    report[name] = {
        "n": int(len(T)), "avg_bp": round(avg_bp, 2),
        "pos_days": f"{pos_days}/{len(by_day)}", "by_day_bp": by_day,
        "long_bp": round(float(T[T["side"] == 1]["pnl"].mean() * 10000),
                         2),
        "short_bp": round(float(T[T["side"] == -1]["pnl"].mean()
                                * 10000), 2)}
    print(f"{name}: {len(T)} сделок, avg {avg_bp:+.1f} бп, "
          f"лонг {report[name]['long_bp']:+.1f} / шорт "
          f"{report[name]['short_bp']:+.1f}, дни {by_day}", flush=True)

a = report.get("A", {})
b = report.get("B", {})
delta = (b.get("avg_bp", -99) - a.get("avg_bp", -99)) \
    if (a.get("n") and b.get("n")) else None
alive = bool(a.get("n") and b.get("n") and delta is not None
             and delta >= 3.0 and b["avg_bp"] > 0 and b["n"] >= 100)
report["delta_bp"] = delta
report["alive"] = alive
verdict_txt = "B ЖИВ (кандидат)" if alive else "нет"
print(f"ВЕРДИКТ: {verdict_txt} (Δ {delta} бп)", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
