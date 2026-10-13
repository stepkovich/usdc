"""БУРЯ НА USDC-ПАРАХ (предрегистрировано 12.10, идея владельца:
«а что если буря работала бы на парах USDC и мейкером = 0?»).

ЗАМЫСЕЛ: замороженная модель-буря (та же, что в бою, ml5h.txt) ставится
на 36 USDC-пар спринтера. Паспорт монеты (sym_id) берётся от USDT-
близнеца (BTCUSDC -> паспорт BTCUSDT), иначе модель не узнаёт монету.
Свечи 30м — месячные пакеты архива биржи, 2025-03..2026-09.
Комиссии: USDC вход мейкер 0 (промо), выход тейкер 5бп (консервативно;
вариант выход мейкер 0 — вторым числом). Для сравнения те же монеты на
USDT-близнецах с боевыми комиссиями (вход 5бп + выход 5бп).

СТЕКИ (вторично, только показать): «тихая треть» (живая 1/3-квантиль
atr по 36 парам бара) и подтверждающая модель meta.txt (pw>=0.5).

ПРЕДРЕГИСТРАЦИЯ: КАНДИДАТ если буря на USDC при КОНСЕРВАТИВНЫХ
комиссиях (выход тейкер) даёт среднюю >= +10 бп при n >= 300 сделок.
Сравнение с USDT-близнецами за тот же период — та же таблица.
Оговорки: модель не переобучается (заморожена — все сигналы вне её
обучения); btc-признаки для USDC считаются от BTCUSDC (цена почти
та же, что BTCUSDT); период 18 месяцев один режим; ничего не
деплоится.
"""
import io, json, os, glob as _glob, time, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
H = 10
TAKER = 0.0005
WARMUP = 300

_mounts = sorted(_glob.glob("/kaggle/input/*"))
print("примонтировано:", _mounts, flush=True)
_mdir = None
for cand in _glob.glob("/kaggle/input/**/ml5h.txt", recursive=True):
    _mdir = str(Path(cand).parent)
    break
if _mdir is None:
    json.dump({"error": "ml5h.txt не найден в /kaggle/input",
               "mounts": _mounts},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
print("модели из:", _mdir, flush=True)

META = json.load(open(f"{_mdir}/ml5h_meta.json"))
FEATS = list(META["features"])
GATE = float(META.get("gate", 0.65))
SYMS522 = set(META["symbols"])
model = lgb.Booster(model_file=f"{_mdir}/ml5h.txt")
meta_ok_model = lgb.Booster(model_file=f"{_mdir}/meta.txt")
MFEATS = FEATS + ["p_sig", "atr_rel"]
COOLDOWN_MS = 10 * 1800_000
print(f"модель: {len(FEATS)} фич, gate {GATE}", flush=True)

PAIRS = ['1000BONKUSDC', '1000PEPEUSDC', '1000SHIBUSDC', 'AAVEUSDC',
         'ADAUSDC', 'ARBUSDC', 'AVAXUSDC', 'BCHUSDC', 'BNBUSDC',
         'BOMEUSDC', 'BTCUSDC', 'CRVUSDC', 'DOGEUSDC', 'ENAUSDC',
         'ETHFIUSDC', 'ETHUSDC', 'FILUSDC', 'HBARUSDC', 'KAITOUSDC',
         'LINKUSDC', 'LTCUSDC', 'NEARUSDC', 'NEOUSDC', 'ORDIUSDC',
         'PENGUUSDC', 'PNUTUSDC', 'SOLUSDC', 'SUIUSDC', 'TIAUSDC',
         'TRUMPUSDC', 'UNIUSDC', 'WIFUSDC', 'WLDUSDC', 'WLFIUSDC',
         'XRPUSDC', 'ZECUSDC']

def months_back(n):
    out = []
    y, m = 2026, 9
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return sorted(out)

MONTHS = months_back(19)          # 2025-03 .. 2026-09

def dl(sym, mo):
    url = (f"https://data.binance.vision/data/futures/um/monthly/klines/"
           f"{sym}/30m/{sym}-30m-{mo}.zip")
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as r:
            return sym, mo, r.read()
    except Exception:
        return sym, mo, None

jobs = [(s, m) for s in PAIRS + [p.replace("USDC", "USDT") for p in PAIRS]
        for m in MONTHS]
blobs = {}
with ThreadPoolExecutor(16) as ex:
    for sym, mo, data in ex.map(lambda j: dl(*j), jobs):
        if data is not None:
            blobs.setdefault(sym, {})[mo] = data
print(f"пакетов скачано: {sum(len(v) for v in blobs.values())} | "
      f"{time.time()-t0:.0f}с", flush=True)

def read_zip(raw):
    zf = zipfile.ZipFile(io.BytesIO(raw))
    # пакеты архива идут БЕЗ строки заголовков — читаем позиционно
    df = pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0])), header=None,
                     usecols=[0, 2, 3, 4, 5, 7, 9],
                     names=["open_time", "high", "low", "close", "volume",
                            "quote_volume", "taker_buy_base"])
    return df

def frame_of(sym):
    parts = []
    for mo, raw in sorted(blobs.get(sym, {}).items()):
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
    return df if len(df) > WARMUP + 50 else None

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
    if btc_close is not None and sym != "BTCUSDT" and sym != "BTCUSDC":
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

def run_family(symbols, is_usdc, fee_in):
    """Сигналы замороженной модели на семействе; возвращает сделки."""
    frames, raw_frames = {}, {}
    btc_frame = frame_of("BTCUSDC" if is_usdc else "BTCUSDT")
    btc_close = btc_frame["close"].astype("float32") \
        if btc_frame is not None else None
    skipped = []
    for sym in symbols:
        twin = sym.replace("USDC", "USDT")
        if (twin if is_usdc else sym) not in SYMS522:
            skipped.append((sym, "близнеца нет в 522"))
            continue
        df = frame_of(sym)
        if df is None:
            skipped.append((sym, "мало свечей"))
            continue
        X = make_feats(df, btc_close, twin if is_usdc else sym)
        X = X.replace([np.inf, -np.inf], np.nan)
        X["sym_id"] = float(
            zlib.crc32((twin if is_usdc else sym).encode()) & 0xFFFF)
        if not set(FEATS).issubset(X.columns):
            miss = sorted(set(FEATS) - set(X.columns))
            skipped.append((sym, f"нет колонок {miss[:3]}"))
            continue
        frames[sym] = (df, X)
    reasons = {}
    for s, r in skipped:
        reasons[r] = reasons.get(r, 0) + 1
    print(f"{'USDC' if is_usdc else 'USDT'}: в работе {len(frames)}, "
          f"скипы {reasons} | btc_close "
          f"{'есть' if btc_close is not None else 'ОТСУТСТВУЕТ'}",
          flush=True)
    # живая терциль atr по парам семейства на каждый бар
    by_bar = {}
    for sym, (df, X) in frames.items():
        a = X["atr_pct"].values
        o = df["open_time"].values
        for k, v in zip(o, a):
            if np.isfinite(v):
                by_bar.setdefault(k, []).append(v)
    q33 = {k: float(np.quantile(v, 1/3)) for k, v in by_bar.items()
           if len(v) >= 15}
    trades = []
    for sym, (df, X) in frames.items():
        ok = X[FEATS].notna().all(axis=1).values
        ots = df["open_time"].values[ok]
        closes = df["close"].values[ok].astype("float64")
        atrs = X["atr_pct"].values[ok].astype("float64")
        if len(ots) <= WARMUP + H:
            continue
        M = np.column_stack([X[f].values[ok] for f in FEATS])[WARMUP:]
        p = model.predict(M)
        ots2 = ots[WARMUP:]; closes2 = closes[WARMUP:]
        atrs2 = atrs[WARMUP:]
        last = -10**18
        for i in range(len(p)):
            if ots2[i] - last < COOLDOWN_MS:
                continue
            if not (p[i] > GATE or p[i] < 1 - GATE):
                continue
            side = 1 if p[i] > GATE else -1
            if i + H >= len(closes2):
                continue
            fwd = side * (closes2[i + H] / closes2[i] - 1)
            atr = atrs2[i]
            in_f = (np.isfinite(atr) and q33.get(int(ots2[i])) is not None
                    and atr >= q33[int(ots2[i])])
            pw = float(meta_ok_model.predict(
                np.array([M[i].tolist() + [p[i], atr]]))[0]) \
                if meta_ok_model is not None else float("nan")
            base = fwd * 1e4 - fee_in * 1e4 - TAKER * 1e4   # выход тейкер
            trades.append({
                "sym": sym, "ot": int(ots2[i]), "side": side,
                "p": float(p[i]), "atr": float(atr),
                "cons_bp": round(float(base), 2),            # вых тейкер
                "maker_bp": round(float(fwd * 1e4 - fee_in * 1e4), 2),
                "in_filter": bool(in_f), "meta_ok": bool(pw >= 0.5),
                "pw": round(pw, 4)})
            last = ots2[i]
    return trades

out = {}
for name, fam, fee_in in (("USDC", PAIRS, 0.0),
                          ("USDT", [p.replace("USDC", "USDT")
                                    for p in PAIRS], 0.0005)):
    tr = run_family(fam, name == "USDC", fee_in)
    df = pd.DataFrame(tr)
    out[name] = df
    if len(df):
        cons = df["cons_bp"]
        mk = df["maker_bp"]
        inf = df[df["in_filter"]]
        mo = df[df["meta_ok"]]
        both = df[df["in_filter"] & df["meta_ok"]]
        print(f"== {name}: сделок {len(df)}, средняя (вых тейкер) "
              f"{cons.mean():+.2f}бп | (вых мейкер) {mk.mean():+.2f}бп | "
              f"WR {float((cons > 0).mean()*100):.0f}% | "
              f"тихая-треть: {len(inf)} {inf.cons_bp.mean():+.2f}бп | "
              f"мета: {len(mo)} {mo.cons_bp.mean():+.2f}бп | обе: "
              f"{len(both)} {both.cons_bp.mean() if len(both) else float('nan'):+.2f}бп",
              flush=True)
    else:
        print(f"== {name}: сделок 0", flush=True)

report = {"pre_reg": "КАНДИДАТ: USDC консервативно (вых тейкер) "
                     ">= +10бп при n>=300",
          "usdc": None, "usdt": None}
for name in ("USDC", "USDT"):
    df = out.get(name)
    if df is None or not len(df):
        continue
    cons = df["cons_bp"]
    inf = df[df["in_filter"]]
    mo = df[df["meta_ok"]]
    both = df[df["in_filter"] & df["meta_ok"]]
    blk = {"n": int(len(df)),
           "avg_bp": round(float(cons.mean()), 2),
           "maker_exit_bp": round(float(df["maker_bp"].mean()), 2),
           "wr": round(float((cons > 0).mean() * 100), 1),
           "filter": {"n": int(len(inf)),
                      "avg_bp": round(float(inf["cons_bp"].mean()), 2)
                      if len(inf) else None},
           "meta": {"n": int(len(mo)),
                    "avg_bp": round(float(mo["cons_bp"].mean()), 2)
                    if len(mo) else None},
           "both": {"n": int(len(both)),
                    "avg_bp": round(float(both["cons_bp"].mean()), 2)
                    if len(both) else None}}
    if name == "USDC":
        blk["candidate"] = bool(len(df) >= 300
                                and blk["avg_bp"] is not None
                                and blk["avg_bp"] >= 10.0)
        report["usdc"] = blk
    else:
        report["usdt"] = blk
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ЭКЗАМЕН USDC ЗАВЕРШЁН", flush=True)
