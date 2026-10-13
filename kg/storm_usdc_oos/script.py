"""БУРЯ НА USDC — ЧЕСТНАЯ ПРОВЕРКА НА НЕВИДАННЫХ ДНЯХ (12.10, «го»).

Первый прогон (usdc-storm-usdc) снял сигналы с месяцев, которые модель
УЧИЛА — это была подсказка (+175бп), отброшена. Здесь signals ТОЛЬКО
с 1 по 11 октября 2026 — модель обучена по сентябрь включительно, эти
дни она не видела никогда. Сентябрьские свечи — только подпорка для
индикаторов (окна 480 баров).

КОМИССИИ: USDC вход мейкер 0, выход тейкер 5бп (консервативно; вторым
числом выход мейкер 0). Рядом — USDT-близнецы за те же дни с боевыми
комиссиями (вход 5бп + выход 5бп).

СТЕКИ: тихая треть (живая 1/3-квантиль atr) и подтверждающая модель
(meta.txt, pw>=0.5) — на тех же сигналах.

ВЕРДИКТ: n ожидаемо мало (десятки-сотни) — это проверка ЗНАКА средней,
не приговор. Плюс на невиданных днях с честными комиссиями = дорога к
пилоту; минус = идея закрыта.
"""
import io, json, os, glob as _glob, time, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
H = 10
TAKER = 0.0005
WARMUP = 300

_mounts = sorted(_glob.glob("/kaggle/input/*"))
_mdir = None
for cand in _glob.glob("/kaggle/input/**/ml5h.txt", recursive=True):
    _mdir = str(Path(cand).parent)
    break
META = json.load(open(f"{_mdir}/ml5h_meta.json"))
FEATS = list(META["features"])
GATE = float(META.get("gate", 0.65))
SYMS522 = set(META["symbols"])
model = lgb.Booster(model_file=f"{_mdir}/ml5h.txt")
meta_model = lgb.Booster(model_file=f"{_mdir}/meta.txt")
COOLDOWN_MS = 10 * 1800_000
print(f"модель: {len(FEATS)} фич, gate {GATE}", flush=True)

PAIRS = ['1000BONKUSDC', '1000PEPEUSDC', '1000SHIBUSDC', 'AAVEUSDC',
         'ADAUSDC', 'ARBUSDC', 'AVAXUSDC', 'BCHUSDC', 'BNBUSDC',
         'BOMEUSDC', 'BTCUSDC', 'CRVUSDC', 'DOGEUSDC', 'ENAUSDC',
         'ETHFIUSDC', 'ETHUSDC', 'FILUSDC', 'HBARUSDC', 'KAITOUSDC',
         'LINKUSDC', 'LTCUSDC', 'NEOUSDC', 'ORDIUSDC', 'PENGUUSDC',
         'PNUTUSDC', 'SOLUSDC', 'SUIUSDC', 'TIAUSDC', 'TRUMPUSDC',
         'UNIUSDC', 'WIFUSDC', 'WLDUSDC', 'WLFIUSDC', 'XRPUSDC',
         'ZECUSDC']
DAYS = [f"2026-10-{d:02d}" for d in range(1, 12)]   # невиданные дни

import urllib.request
def dl(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception:
        return None

def url_month(sym, mo):
    return (f"https://data.binance.vision/data/futures/um/monthly/klines/"
            f"{sym}/30m/{sym}-30m-{mo}.zip")

def url_day(sym, d):
    return (f"https://data.binance.vision/data/futures/um/daily/klines/"
            f"{sym}/30m/{sym}-30m-{d}.zip")

syms_all = PAIRS + [p.replace("USDC", "USDT") for p in PAIRS]
jobs = [(s, url_month(s, "2026-09")) for s in syms_all]
jobs += [(s, url_day(s, d)) for s in syms_all for d in DAYS]
blobs = {}
with ThreadPoolExecutor(16) as ex:
    for (sym, _u), data in zip(jobs, ex.map(dl, [j[1] for j in jobs])):
        if data is not None:
            blobs.setdefault(sym, []).append(data)
print(f"пакетов: {sum(len(v) for v in blobs.values())} | "
      f"{time.time()-t0:.0f}с", flush=True)

def read_zip(raw):
    zf = zipfile.ZipFile(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0])), header=None,
                       usecols=[0, 2, 3, 4, 5, 7, 9],
                       names=["open_time", "high", "low", "close", "volume",
                              "quote_volume", "taker_buy_base"])

def frame_of(sym):
    parts = []
    for raw in blobs.get(sym, []):
        try:
            parts.append(read_zip(raw))
        except Exception:
            pass
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    for c in df.columns:
        if c != "open_time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    ot = pd.to_numeric(df["open_time"], errors="coerce").astype("float64")
    keep = ot.notna() & df["close"].notna()
    df, ot = df[keep], ot[keep]
    df["open_time"] = np.where(ot > 1e14, ot / 1000.0, ot).astype(np.int64)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    return df if len(df) > 400 else None

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
    if btc_close is not None and sym not in ("BTCUSDT", "BTCUSDC"):
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
    return X

OCT0 = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp() * 1000)

def run_family(symbols, is_usdc, fee_in):
    frames = {}
    btc_frame = frame_of("BTCUSDC" if is_usdc else "BTCUSDT")
    btc_close = btc_frame["close"].astype("float32") \
        if btc_frame is not None else None
    reasons = {}
    for sym in symbols:
        twin = sym.replace("USDC", "USDT")
        if (twin if is_usdc else sym) not in SYMS522:
            reasons["близнеца нет"] = reasons.get("близнеца нет", 0) + 1
            continue
        df = frame_of(sym)
        if df is None:
            reasons["мало свечей"] = reasons.get("мало свечей", 0) + 1
            continue
        X = make_feats(df, btc_close, twin if is_usdc else sym)
        X = X.replace([np.inf, -np.inf], np.nan)
        X["sym_id"] = float(zlib.crc32(
            (twin if is_usdc else sym).encode()) & 0xFFFF)
        if not set(FEATS).issubset(X.columns):
            reasons["нет колонок"] = reasons.get("нет колонок", 0) + 1
            continue
        frames[sym] = (df, X)
    print(f"{'USDC' if is_usdc else 'USDT'}: в работе {len(frames)}, "
          f"скипы {reasons}, btc_close "
          f"{'есть' if btc_close is not None else 'НЕТ'}", flush=True)
    by_bar = {}
    for sym, (df, X) in frames.items():
        for k, v in zip(df["open_time"].values, X["atr_pct"].values):
            if np.isfinite(v):
                by_bar.setdefault(k, []).append(v)
    q33 = {k: float(np.quantile(v, 1/3)) for k, v in by_bar.items()
           if len(v) >= 12}
    trades = []
    for sym, (df, X) in frames.items():
        ok = X[FEATS].notna().all(axis=1).values
        ots = df["open_time"].values[ok]
        closes = df["close"].values[ok].astype("float64")
        atrs = X["atr_pct"].values[ok].astype("float64")
        sig_mask = ots >= OCT0
        idxs = np.nonzero(sig_mask)[0]
        if len(idxs) == 0:
            continue
        M = np.column_stack([X[f].values[ok] for f in FEATS])[idxs]
        p = model.predict(M)
        last = -10**18
        for i, j in enumerate(idxs):
            if ots[j] - last < COOLDOWN_MS:
                continue
            if not (p[i] > GATE or p[i] < 1 - GATE):
                continue
            if j + H >= len(closes):
                continue                     # хвост не доехал — сносим
            side = 1 if p[i] > GATE else -1
            fwd = side * (closes[j + H] / closes[j] - 1)
            atr = atrs[j]
            in_f = (np.isfinite(atr) and q33.get(int(ots[j])) is not None
                    and atr >= q33[int(ots[j])])
            pw = float(meta_model.predict(
                np.array([M[i].tolist() + [p[i], atr]]))[0])
            trades.append({
                "sym": sym, "ot": int(ots[j]), "side": side,
                "cons_bp": round(float(fwd * 1e4 - fee_in * 1e4
                                       - TAKER * 1e4), 2),
                "maker_bp": round(float(fwd * 1e4 - fee_in * 1e4), 2),
                "in_filter": bool(in_f), "meta_ok": bool(pw >= 0.5)})
            last = ots[j]
    return trades

report = {"note": "OOS: сигналы только 1-11 октября 2026", "usdc": None,
          "usdt": None}
for name, fam, fee_in in (("USDC", PAIRS, 0.0),
                          ("USDT", [p.replace("USDC", "USDT")
                                    for p in PAIRS], 0.0005)):
    tr = run_family(fam, name == "USDC", fee_in)
    if not tr:
        print(f"== {name}: сделок 0", flush=True)
        continue
    df = pd.DataFrame(tr)
    inf = df[df["in_filter"]]
    mo = df[df["meta_ok"]]
    both = df[df["in_filter"] & df["meta_ok"]]
    lo = df[df["side"] == 1]
    blk = {"n": int(len(df)),
           "avg_bp": round(float(df["cons_bp"].mean()), 2),
           "maker_exit_bp": round(float(df["maker_bp"].mean()), 2),
           "wr": round(float((df["cons_bp"] > 0).mean() * 100), 1),
           "longs": {"n": int(len(lo)),
                     "avg_bp": round(float(lo["cons_bp"].mean()), 2)
                     if len(lo) else None},
           "filter": {"n": int(len(inf)),
                      "avg_bp": round(float(inf["cons_bp"].mean()), 2)
                      if len(inf) else None},
           "meta": {"n": int(len(mo)),
                    "avg_bp": round(float(mo["cons_bp"].mean()), 2)
                    if len(mo) else None},
           "both": {"n": int(len(both)),
                    "avg_bp": round(float(both["cons_bp"].mean()), 2)
                    if len(both) else None}}
    report[name.lower()] = blk
    print(f"== {name}: {blk}", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("OOS-ПРОВЕРКА ЗАВЕРШЕНА", flush=True)
