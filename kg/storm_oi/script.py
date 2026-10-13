"""ОТКРЫТЫЙ ИНТЕРЕС КАК ТОПЛИВО БУРИ (предрегистрировано 12.10, ход 1).

Сигналы боевой модели за три свежих фолда (2025Q1, 2025Q4, 2026Q3 —
модель училась только на прошлом, все сигналы OOS). К каждому клеим
метрики биржи: изменение открытого интереса за 1ч/6ч/24ч до сигнала,
коэффициент топ-трейдеров и его сдвиг, агрессивный поток.

ВОПРОС 1: сигналы при растущем/падающем ОИ зарабатывают по-разному?
(терцили, лонги и шорты отдельно)
ВОПРОС 2 (главный): подтверждающая модель С ОИ-признаками против
сегодняшней меты без ОИ (обе учатся на фолдах прошлого, применяются
к следующему) — принятые v2 лучше принятых v1?

ПРЕДРЕГИСТРАЦИЯ: подсказка ОИ = терцильная разница >= 10бп хотя бы у
одной стороны при n >= 1000/бакет; подсказка мета-v2 = принятые(v2) −
принятые(v1) >= +3бп при n >= 1000. Издержки 10бп внутри. Метрики с
шагом 5 мин существуют с ~2023; дыры по ближайшему значению в
пределах 10 минут. Ничего не деплоится.
"""
import io, glob, json, time, zipfile, zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np, pandas as pd
import lightgbm as lgb
import urllib.request

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000
FEE = 0.0010
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
META_PARAMS = dict(n_estimators=120, learning_rate=0.05, max_depth=3,
                   num_leaves=7, subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=4, verbosity=-1)
FOLDS = ["2025Q1", "2025Q4", "2026Q3"]
MONTHS = [f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13)
          if not (y == 2026 and m > 9)]

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
_mdir = None
for cand in glob.glob("/kaggle/input/**/ml5h_meta.json", recursive=True):
    _mdir = str(Path(cand).parent)
    break
META = json.load(open(f"{_mdir}/ml5h_meta.json"))
FEATS = list(META["features"])
GATE = float(META.get("gate", 0.65))
model = lgb.Booster(model_file=f"{_mdir}/ml5h.txt")
meta_v1 = lgb.Booster(model_file=f"{_mdir}/meta.txt")
MFEATS = FEATS + ["p_sig", "atr_rel"]
OIFEATS = ["oi_chg_1h", "oi_chg_6h", "oi_chg_24h", "top_ratio_now",
           "top_ratio_chg_6h", "taker_now"]
print(f"модель {len(FEATS)} фич, gate {GATE}", flush=True)

btc_df = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                     usecols=["open_time", "close"])
btc_df["open_time"] = btc_df["open_time"].astype(np.int64)
btc_df.index = pd.to_datetime(btc_df["open_time"], unit="ms")
BTC = btc_df["close"].astype("float32")

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
    if btc_close is not None and sym != "BTCUSDT":
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
    X, tr = make_feats(df, BTC, sym)
    close = df["close"].values
    atr_rel = (tr.rolling(14).mean() / close).values.astype("float64")
    fut = np.full(len(df), np.nan, dtype="float64")
    fut[:-H] = close[H:]
    fwd = fut / close - 1
    thr = 0.5 * np.nan_to_num(atr_rel, nan=np.nan)
    X["y"] = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                      np.where(fwd > thr, 1.0,
                               np.where(fwd < -thr, 0.0, np.nan)))
    X["fwd"] = fwd.astype("float32")
    X["atr_rel"] = atr_rel.astype("float32")
    X["ot"] = df["open_time"].values.astype(np.int64)
    X["sym_name"] = sym
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=["atr_pct"]))
    if i % 150 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
print(f"строк {len(data)} | {time.time()-t0:.0f}с", flush=True)

def collect(d):
    pools = {int(fi): gg.sort_values("ot") for fi, gg in d.groupby("sym_id")}
    sigs = []
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
            if j >= len(ots) or ots[j] != row.ot or j + H >= len(ots):
                continue
            base = side * float(row.fwd) - FEE
            vec = [float(getattr(row, f)) for f in FEATS]
            vec += [float(row.p), float(row.atr_rel)]
            sigs.append({"sym": row.sym_name, "ot": int(row.ot),
                         "side": side, "base_bp": base * 1e4,
                         "p": float(row.p), "atr_rel": float(row.atr_rel),
                         "vec": vec, "quarter": row.quarter})
    return sigs

sig_frames = []
for qq in FOLDS:
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 5000 or len(trd) < 500_000:
        print(f"фолд {qq}: мало данных", flush=True)
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    trd_ok = trd.dropna(subset=["y"])
    m.fit(trd_ok[FEATS], trd_ok["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    sigs = collect(d)
    sig_frames.append(pd.DataFrame(sigs))
    print(f"фолд {qq}: {len(sigs)} сигналов | {time.time()-t0:.0f}с",
          flush=True)

sg = pd.concat(sig_frames, ignore_index=True)
_mcols = pd.DataFrame(sg["vec"].tolist(), columns=MFEATS)
sg = pd.concat([sg.drop(columns=["vec", "p", "atr_rel"]), _mcols], axis=1)
print(f"всего сигналов: {len(sg)}, монет {sg.sym.nunique()}", flush=True)

# ---------- метрики ОИ: дневные файлы за дни сигналов ±2 ----------
need_syms = sorted(sg.sym.unique())
day_set = set()
for o in sg.ot:
    d0 = datetime.fromtimestamp(o / 1000, tz=timezone.utc).date()
    for k in range(-2, 3):
        day_set.add(str(d0 + pd.Timedelta(days=k)))
days_list = sorted(day_set)
jobs = [(s, d) for s in need_syms for d in days_list]
print(f"качаем metrics: {len(jobs)} файлов ({len(need_syms)} монет x "
      f"{len(days_list)} дней)", flush=True)

def dl(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception:
        return None

oi_raw = {}
with ThreadPoolExecutor(16) as ex:
    for (s, d), data in zip(jobs, ex.map(
            lambda j: dl(f"https://data.binance.vision/data/futures/um/"
                         f"daily/metrics/{j[0]}/{j[0]}-metrics-{j[1]}.zip"),
            jobs)):
        if data is not None:
            oi_raw.setdefault(s, []).append((d, data))
print(f"metrics скачано: {sum(len(v) for v in oi_raw.values())}/"
      f"{len(jobs)} | {time.time()-t0:.0f}с", flush=True)

MCOLS = ["create_time", "sum_open_interest",
         "count_toptrader_long_short_ratio",
         "sum_toptrader_long_short_ratio",
         "sum_taker_long_short_vol_ratio"]
oi_series = {}
for s, lst in oi_raw.items():
    parts = []
    for d, raw in sorted(lst):
        try:
            zf2 = zipfile.ZipFile(io.BytesIO(raw))
            parts.append(pd.read_csv(io.BytesIO(zf2.read(
                zf2.namelist()[0])), usecols=MCOLS))
        except Exception:
            pass
    if not parts:
        continue
    m = pd.concat(parts, ignore_index=True)
    m["t"] = pd.to_datetime(m["create_time"]).astype("int64") // 10**6
    m = m.drop_duplicates("t").sort_values("t")
    if len(m) > 50:
        oi_series[s] = m
print(f"монет с метриками: {len(oi_series)}", flush=True)

def oi_feats(sym, ot):
    m = oi_series.get(sym)
    if m is None:
        return None
    t = m["t"].values
    j = int(np.searchsorted(t, ot))
    if j >= len(t) or t[j] - ot > 10 * 60_000:
        return None
    r = m.iloc[j]
    out = {}
    v_now = float(r["sum_open_interest"])
    for lag_ms, nm in ((3600_000, "1h"), (6 * 3600_000, "6h"),
                       (24 * 3600_000, "24h")):
        k = int(np.searchsorted(t, ot - lag_ms))
        if k >= len(t) or t[k] < ot - lag_ms - 10 * 60_000:
            out[f"oi_chg_{nm}"] = np.nan
        else:
            v0 = float(m.iloc[k]["sum_open_interest"])
            out[f"oi_chg_{nm}"] = (v_now / v0 - 1) if v0 > 0 else np.nan
    out["top_ratio_now"] = float(r["sum_toptrader_long_short_ratio"])
    k6 = int(np.searchsorted(t, ot - 6 * 3600_000))
    out["top_ratio_chg_6h"] = \
        (out["top_ratio_now"] -
         float(m.iloc[k6]["sum_toptrader_long_short_ratio"])) \
        if k6 < len(t) else np.nan
    out["taker_now"] = float(r["sum_taker_long_short_vol_ratio"])
    return out

oi_rows = []
for r in sg.itertuples():
    f = oi_feats(r.sym, r.ot)
    oi_rows.append(f if f else {k: np.nan for k in OIFEATS})
sg = pd.concat([sg.reset_index(drop=True),
                pd.DataFrame(oi_rows, columns=OIFEATS)], axis=1)
sg_no = sg.dropna(subset=OIFEATS)
print(f"с ОИ: {len(sg_no)} из {len(sg)} | {time.time()-t0:.0f}с",
      flush=True)

report = {"pre_reg": "ОИ: терцильная разница >=10бп (n>=1000/бакет); "
                     "мета-v2: принятые(v2) − принятые(v1) >= +3бп, "
                     "n>=1000", "raw": {}, "meta_walk": {}}

for side, nm in ((1, "лонги"), (-1, "шорты")):
    d = sg_no[sg_no.side == side]
    if len(d) < 1500:
        print(f"{nm}: {len(d)} сигналов с ОИ — мало", flush=True)
        continue
    t = np.quantile(d["oi_chg_6h"], [1/3, 2/3])
    d2 = d.assign(b=np.searchsorted(t, d["oi_chg_6h"]) + 1)
    tab = {int(k): {"n": int(len(v)), "avg": round(float(v.mean()), 1),
                    "wr": round(float((v > 0).mean() * 100), 1)}
           for k, v in d2.groupby("b")["base_bp"]}
    report["raw"][nm] = tab
    print(f"{nm} по ОИ-терцилям (1=растёт..3=падает): {tab}", flush=True)

folds_sq = sorted(sg.quarter.unique())
acc1_all, acc2_all, base_all, n1, n2 = [], [], [], 0, 0
for i, qq in enumerate(folds_sq):
    if i == 0:
        continue
    hi = sg_no[sg_no.quarter.isin(folds_sq[:i])]
    te = sg_no[sg_no.quarter == qq]
    if len(te) < 500 or len(hi) < 2000:
        continue
    hy = (hi["base_bp"] > 0).astype("int64")
    p1 = lgb.LGBMClassifier(**META_PARAMS).fit(hi[MFEATS], hy) \
        .predict_proba(te[MFEATS])[:, 1]
    p2 = lgb.LGBMClassifier(**META_PARAMS).fit(hi[MFEATS + OIFEATS], hy) \
        .predict_proba(te[MFEATS + OIFEATS])[:, 1]
    a1 = te[p1 >= 0.5]; a2 = te[p2 >= 0.5]
    acc1_all.extend(a1.base_bp.tolist())
    acc2_all.extend(a2.base_bp.tolist())
    base_all.extend(te.base_bp.tolist())
    n1 += len(a1); n2 += len(a2)
    print(f"фолд {qq}: все {te.base_bp.mean():+.1f}бп | v1 приняла "
          f"{len(a1)} ({a1.base_bp.mean() if len(a1) else float('nan'):+.1f}бп) | "
          f"v2 приняла {len(a2)} "
          f"({a2.base_bp.mean() if len(a2) else float('nan'):+.1f}бп)",
          flush=True)

if acc1_all and acc2_all:
    a1 = float(np.mean(acc1_all)); a2 = float(np.mean(acc2_all))
    b_all = float(np.mean(base_all))
    hint = bool(n2 >= 1000 and a2 - a1 >= 3.0)
    report["meta_walk"] = {"n_v1": n1, "avg_v1": round(a1, 2),
                           "n_v2": n2, "avg_v2": round(a2, 2),
                           "delta": round(a2 - a1, 2),
                           "all_avg": round(b_all, 2), "hint": hint}
    print(f"ИТОГ: все {b_all:+.2f} | v1 {a1:+.2f} (n={n1}) | "
          f"v2+ОИ {a2:+.2f} (n={n2}) | Δ{a2-a1:+.2f} -> "
          f"{'НАМЁК' if hint else 'нет'}", flush=True)
sg_no.to_csv("/kaggle/working/signals_oi.csv", index=False)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ОИ-ЛАБОРАТОРИЯ ЗАВЕРШЕНА", flush=True)
