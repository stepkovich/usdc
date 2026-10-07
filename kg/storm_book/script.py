"""БУРЯ-30мин + СТAKAN-НАПРАВЛЕНИЕ (предрегистрация 06.10, идея владельца).

ВОПРОС: сжать окно бури с 5ч до 30 минут (каждые 15 мин) и брать
направление от книги заявок (навык спринтера) + BTC-поводок.
Экономика: USDC-пары, МЕЙКЕР = 0 (вход бесплатный), выход тейкер
с фолбэком (5бп + 5бп слип); справочный вариант — выход тоже мейкер.

ДАННЫЕ: 37 USDC-активов. Свечи 15м с архива биржи (с 22.09 — прогрев);
книга заявок — наша запись (датасет usdc-lob-features, строки 500мс,
с 25.09): к каждому закрытию 15м-бара берём ПОСЛЕДНЮЮ строку бина
[close-15м, close) — строго без подглядывания.

МЕТКИ (на закрытии бара t): fwd = close[t+2]/close[t] − 1 (окно 30 мин);
storm = |fwd| > 1.0 × ATR14(15м); dir = знак fwd.
ПРИЗНАКИ: (A) 24 свечные формулы в барах; (B) A + 6 книжных (imb5,
imb10, imb20, microprice_rel, flowimb10, vpin10) + btc15_4 (ход BTC
за час в 15м-барах).

ПРОТОКОЛ: walk-forward по дням (тест = последние 4 дня записи).
1) БУРЯ: acc/отбор шторма у A vs B (книга помогает найти бурю?).
2) НАПРАВЛЕНИЕ (главное): среди уверенных бурь (p_A > 0.65) — точность
   знака наклона книги (среднее imb) и знака момента свечей; торги:
   вход мейкер 0 на закрытии, выход через 30 мин.
ВЕРДИКТ (записан до прогона): направление живое, если на тест-днях
acc(книга) >= 52% при n >= 150 И avg после тейкер-выхода >= +3 бп
(вариант мейкер-выход >= +5 бп); И книга точнее момента свечей
минимум на 3 пп. Ничего не деплоится.
"""
import glob, io, json, time, zipfile, zlib, datetime
import urllib.request, urllib.error
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
N_TEST_DAYS = 4
STORM_GATE = 0.65
MAKER_IN = 0.0                     # USDC промо: мейкер = 0
TAKER_OUT = 0.0005 + 0.0005        # fee + slip
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
DAYS_K = [f"2026-09-{d:02d}" for d in range(20, 31)] + \
          [f"2026-10-{d:02d}" for d in range(1, 7)]
BASE = "https://data.binance.vision/data/futures/um"
COLS = ["open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "tbb", "tbq", "ig"]

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return "ok", r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "404", None
        return "err", None
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

def get_symbol(sym, rule="15m"):
    parts = []
    jobs = [f"{BASE}/daily/klines/{sym}/{rule}/{sym}-{rule}-{d}.zip"
            for d in DAYS_K]
    for url in jobs:
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

btc15 = get_symbol("BTCUSDC")
btc15.index = pd.to_datetime(btc15["open_time"], unit="ms")
btc_c = btc15["close"].astype("float64")
print(f"BTCUSDC 15м: {len(btc15)} баров | {time.time()-t0:.0f}с",
      flush=True)

def make_feats(df, sym):
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
        b = btc_c.reindex(df.index).ffill()
        X["btc_ret_10"] = b.pct_change(4).astype("float32")
        X["btc_ret_20"] = b.pct_change(8).astype("float32")
        bret = b.pct_change()
        X["btc_corr"] = rets.rolling(20, min_periods=10).corr(bret) \
            .astype("float32")
        X["rel_ret_20"] = (rets.rolling(8).sum()
                           - bret.rolling(8).sum()).astype("float32")
        X["rel_ret_60"] = (rets.rolling(24).sum()
                           - bret.rolling(24).sum()).astype("float32")
        cov = rets.rolling(96).cov(bret); var = bret.rolling(96).var()
        X["beta_20"] = (cov / var.replace(0, np.nan)).astype("float32")
        X["corr_20"] = rets.rolling(96, min_periods=40).corr(bret) \
            .astype("float32")
    X["h4_momentum"] = close.pct_change(24).astype("float32")
    X["hour"] = df.index.hour.astype("float32")
    X["dow"] = df.index.dayofweek.astype("float32")
    X["sym_id"] = np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)
    return X, tr

# --- свечи 15м по всем символам ---
frames = []
for k, sym in enumerate(SYMS, 1):
    df = get_symbol(sym)
    if df is None or len(df) < 1200:
        continue
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    X, tr = make_feats(df, sym)
    close = df["close"].values
    atr = (tr.rolling(14).mean() / close).values.astype("float64")
    fwd = np.full(len(df), np.nan, dtype="float64")
    fwd[:-2] = close[2:] / close[:-2] - 1     # окно 30 минут
    thr = 1.0 * np.nan_to_num(atr, nan=np.nan)
    X["storm"] = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                          (np.abs(fwd) > thr).astype("float32"))
    X["dir_up"] = np.where(np.isnan(fwd), np.nan,
                           (fwd > 0).astype("float32"))
    X["fwd"] = fwd.astype("float32")
    X["close"] = close
    X["ot"] = df["open_time"].values.astype(np.int64)
    X["sym"] = sym
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=["atr_pct"]))
    if k % 10 == 0:
        print(f"свечи {k}/{len(SYMS)} | {time.time()-t0:.0f}с", flush=True)
candles = pd.concat(frames, ignore_index=True)
del frames
print(f"свечных строк {len(candles)} "
      f"(штормов {int(candles['storm'].sum())})", flush=True)

# --- книга заявок: последняя строка каждого 15м-бина ---
book_frames = []
for f in sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True)):
    try:
        bf = pd.read_parquet(f, columns=["ts", "symbol", "mid",
                                         "microprice", "imb5", "imb10",
                                         "imb20", "flow10_buy",
                                         "flow10_sell", "vpin10"])
        book_frames.append(bf)
    except Exception:
        pass
book = pd.concat(book_frames, ignore_index=True)
del book_frames
print(f"книжных строк {len(book)} | {time.time()-t0:.0f}с", flush=True)
book["bin"] = (book["ts"] // 90_000) * 90_000      # открытие 15м-бара
book = book.sort_values("ts").groupby(["symbol", "bin"]).last() \
    .reset_index()
book["microprice_rel"] = (book["microprice"] / book["mid"] - 1) * 10000
tot = book["flow10_buy"] + book["flow10_sell"]
book["flowimb10"] = np.where(tot > 0,
                             (book["flow10_buy"] - book["flow10_sell"])
                             / tot.replace(0, np.nan), 0)
book["ot"] = book["bin"].astype("int64")
book = book[["symbol", "ot", "imb5", "imb10", "imb20", "microprice_rel",
             "flowimb10", "vpin10"]]
print(f"бинов книги {len(book)} | {time.time()-t0:.0f}с", flush=True)

ds = candles.merge(book, left_on=["sym", "ot"], right_on=["symbol", "ot"],
                   how="left")
del candles, book
ds["day"] = pd.to_datetime(ds["ot"], unit="ms").dt.date
days = sorted(ds["day"].unique())
print(f"дней {len(days)}: {days[0]}..{days[-1]}", flush=True)

BASE_FEATS = ["ret_1", "ret_2", "ret_4", "ret_8", "h4_momentum",
              "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy",
              "clv", "zscore", "rsi", "sma_cross", "btc_ret_10",
              "btc_ret_20", "btc_corr", "rel_ret_20", "rel_ret_60",
              "beta_20", "corr_20", "hour", "dow", "sym_id"]
BOOK_FEATS = ["imb5", "imb10", "imb20", "microprice_rel", "flowimb10",
              "vpin10"]

POOL = {}
def wf_acc(feats, target):
    rows = []
    pool = []
    for day in days[-N_TEST_DAYS:]:
        tr = ds[(ds["day"] < day)].dropna(subset=feats + [target])
        te = ds[ds["day"] == day].dropna(subset=feats + [target])
        if len(te) < 300 or len(tr) < 3000:
            continue
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(tr[feats], tr[target], categorical_feature=["sym_id"])
        p = m.predict_proba(te[feats])[:, 1]
        acc = float(((p > 0.5) == (te[target] == 1)).mean() * 100)
        rows.append({"day": str(day), "n": len(te), "acc": round(acc, 2)})
        pool.append(te.assign(p=p))
    if target == "storm":
        POOL[feats[0]] = pd.concat(pool, ignore_index=True) if pool \
            else pd.DataFrame()
    return rows

print("ЭКЗАМЕН БУРИ: A (свечи) vs B (свечи+книга):", flush=True)
accA = wf_acc(BASE_FEATS, "storm")
print("  A:", accA, flush=True)
accB = wf_acc(BASE_FEATS + BOOK_FEATS, "storm")
print("  B:", accB, flush=True)

# --- направление среди уверенных бурь ---
conf = POOL[BASE_FEATS[0]]
conf = conf[conf["p"] > STORM_GATE].dropna(subset=["dir_up", "fwd"])
res = {"storm_acc_A": accA, "storm_acc_B": accB, "conf_n": len(conf)}
if len(conf) >= 150:
    tilt = (conf[["imb5", "imb10", "imb20"]].mean(axis=1))
    ok_book = tilt.notna()
    acc_book = float((np.sign(tilt[ok_book])
                      == np.where(conf["fwd"][ok_book] > 0, 1, -1)
                      ).mean() * 100)
    acc_mom = float((np.sign(conf["ret_4"])
                     == np.where(conf["fwd"] > 0, 1, -1)).mean() * 100)
    side = np.sign(tilt)
    pnl_taker = side * conf["fwd"] - TAKER_OUT
    stats = {
        "acc_book": round(acc_book, 2), "acc_momentum": round(acc_mom, 2),
        "n_book": int(ok_book.sum()),
        "avg_bp_taker": round(float(pnl_taker[ok_book].mean() * 10000), 2),
        "median_bp_taker": round(float(pnl_taker[ok_book].median()
                                       * 10000), 2)}
    # мейкер-выход: если за следующий бар экстремум коснулся цены выхода
    res.update(stats)
    print(f"НАПРАВЛЕНИЕ на уверенных бурях (n={len(conf)}): книга "
          f"{acc_book:.1f}% vs момент {acc_mom:.1f}% | "
          f"avg {stats['avg_bp_taker']:+.1f} бп (тейкер-выход)",
          flush=True)
    by_day = {}
    for day, g in conf.groupby("day"):
        okd = g[["imb5", "imb10", "imb20"]].mean(axis=1).notna()
        s = np.sign(g[["imb5", "imb10", "imb20"]].mean(axis=1))
        by_day[str(day)] = [int(okd.sum()),
                            round(float((s[okd] == np.where(
                                g["fwd"][okd] > 0, 1, -1)).mean() * 100),
                                 1)]
    res["by_day"] = by_day
    alive = bool(acc_book >= 52 and ok_book.sum() >= 150
                 and stats["avg_bp_taker"] >= 3.0
                 and acc_book - acc_mom >= 3)
else:
    alive = False
    res["note"] = "слишком мало уверенных бурь"
res["alive"] = alive
print("ВЕРДИКТ:", "НАПРАВЛЕНИЕ ЖИВОЕ" if alive else "нет", flush=True)
json.dump(res, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
