"""РЕНТГЕН БУРИ + ТРЕЙЛИНГ ПОБЕДИТЕЛЯ (предрегистрировано 12.10,
приказ владельца «го»; отвечает на «мы чего-то не видим»).

ЧАСТЬ А — ТРЕЙЛИНГ БЕЗ СТОПА (не тестировался никогда; ATR-браслеты
с фиксированным стопом кромку убивали, трейлинг — другой механизм:
уровень подтягивается только ЗА ценой и не выбивает на откате):
  BASE   будильник close[t+10] (как в бою);
  TRAIL_t  пик = max(закрытие входа, хаи пройденных баров) для лонга
           (зеркально минимум для шорта); выход по ЗАКРЫТИЮ бара,
           когда close <= пик − t×ATR; не сработал -> будильник t+10;
           t ∈ {0.5, 1.0, 1.5}; издержки те же 10бп круг у всех.
  ПРАВИЛО: кандидат если pooled avg(t) − avg(BASE) >= +3 бп, n>=1000.
  Выбор по закрытию бара (не по внутрибаровому касанию) — консервативно.

ЧАСТЬ Б — ГДЕ СИДИТ КРОМКА (разрез базовых сделок, никогда не делали):
  бакеты по: ликвидность монеты (терциль медианного дневного оборота),
  своя волатильность (терциль ATR), час сигнала (4ч блоки), день
  недели, сторона, квартал. Для каждого бакета n и средняя в бп.
  ЭТО КАРТА-РАЗВЕДКА, не гейт: по предрегистрации НИКАКОЙ сегментный
  фильтр из этого прогона не деплоится — интересный сегмент сначала
  подтверждается на свежих днях отдельным прогоном. Цель — увидеть,
  равномерна ли кромка, и найти кандидатов для проверки.

Панель 520 USDT-пар, 30м бары, walk-forward 8 фолдов, gate 0.65, обе
стороны, cooldown 10 баров — механика боевая, та же, что в экзамене.
"""
import glob, io, json, time, zipfile, zlib
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000
TRAILS = [0.5, 1.0, 1.5]
FEE = 0.0010
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
print("монет:", len(members), flush=True)

# ликвидность: медианный дневной оборот по монете (для терцилей)
qvol_med = {}
for name in members:
    sym = name.replace("_30m.csv", "")
    qv = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["quote_volume"])["quote_volume"]
    qvol_med[sym] = float(np.nanmedian(qv.values.astype("float64")))
bounds = np.quantile(list(qvol_med.values()), [1/3, 2/3])
def qrank(sym):
    v = qvol_med.get(sym, 0.0)
    return 1 if v < bounds[0] else (2 if v < bounds[1] else 3)
print("терцили оборота:", bounds, flush=True)

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
    X["close"] = close
    X["high"] = df["high"].values
    X["low"] = df["low"].values
    X["ot"] = df["open_time"].values.astype(np.int64)
    X["sym_name"] = sym
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=["atr_pct"]))
    if i % 100 == 0:
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
print(f"строк {len(data)}, фолды {test_q} | {time.time()-t0:.0f}с",
      flush=True)

def collect(d):
    pools = {int(fi): gg.sort_values("ot") for fi, gg in d.groupby("sym_id")}
    trades = []
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
            seg_h = pool["high"].values[j + 1: j + 1 + H].astype("float64")
            seg_l = pool["low"].values[j + 1: j + 1 + H].astype("float64")
            seg_c = pool["close"].values[j + 1: j + 1 + H].astype("float64")
            if len(seg_h) < H:
                continue
            hr = int(pd.to_datetime(row.ot, unit="ms").hour)
            trades.append({
                "side": side, "quarter": row.quarter,
                "entry": float(row.close), "atr_rel": float(row.atr_rel),
                "fwd": float(row.fwd) if row.fwd == row.fwd else np.nan,
                "seg_h": seg_h, "seg_l": seg_l, "seg_c": seg_c,
                "hour": hr, "dow": int(pd.to_datetime(row.ot,
                                                      unit="ms").dayofweek),
                "qr": qrank(row.sym_name)})
    return trades

def trail_pnl(t, tl):
    """Трейлинг по закрытиям баров; не сработал -> будильник t+10."""
    side, e = t["side"], t["entry"]
    atr_abs = t["atr_rel"] * e
    peak = e
    for i in range(H):
        c = t["seg_c"][i]
        if side == 1:
            peak = max(peak, t["seg_h"][i])
            if c <= peak - tl * atr_abs:
                return side * (c / e - 1) - FEE, i
        else:
            peak = min(peak, t["seg_l"][i])
            if c >= peak + tl * atr_abs:
                return side * (c / e - 1) - FEE, i
    return side * t["fwd"] - FEE, -1

acc = {"BASE": [], **{f"TRAIL_{t}": [] for t in TRAILS}}
exits = {f"TRAIL_{t}": {early: 0 for early in range(H)} for t in TRAILS}
seg_rows = []
all_folds = []
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 20000 or len(trd) < 500_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    trd_ok = trd.dropna(subset=["y"])
    m.fit(trd_ok[FEATS], trd_ok["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    trades = collect(d)
    for t in trades:
        base = t["side"] * t["fwd"] - FEE
        acc["BASE"].append(base)
        for tl in TRAILS:
            p, ei = trail_pnl(t, tl)
            acc[f"TRAIL_{tl}"].append(p)
            exits[f"TRAIL_{tl}"][ei] = exits[f"TRAIL_{tl}"].get(ei, 0) + 1
        hour_blk = t["hour"] // 4
        seg_rows.append((base, t["side"], t["quarter"], t["qr"],
                         hour_blk, t["dow"], t["atr_rel"]))
    all_folds.append(qq)
    print(f"фолд {qq}: сделок {len(trades)} | {time.time()-t0:.0f}с",
          flush=True)

report = {"folds": [str(f) for f in all_folds],
          "pre_reg_A": "кандидат: pooled avg(TRAIL_t) − avg(BASE) >= +3бп, "
                       "n>=1000; выход по закрытию бара",
          "pre_reg_B": "карта-разведка, ничего не деплоится",
          "trailing": {}, "segments": {}}
base_avg = float(np.mean(acc["BASE"]) * 10000)
nb = len(acc["BASE"])
report["trailing"]["BASE"] = {"n": nb, "avg_bp": round(base_avg, 2)}
print(f"BASE: {nb} сделок, {base_avg:+.2f} бп", flush=True)
for tl in TRAILS:
    a = np.array(acc[f"TRAIL_{tl}"], dtype="float64")
    avg = float(a.mean() * 10000)
    report["trailing"][f"TRAIL_{tl}"] = {
        "avg_bp": round(avg, 2), "delta": round(avg - base_avg, 2),
        "candidate": bool(avg - base_avg >= 3.0 and nb >= 1000)}
    print(f"TRAIL_{tl}: avg {avg:+.2f} бп (Δ{avg-base_avg:+.2f}) -> "
          f"{'КАНДИДАТ' if report['trailing'][f'TRAIL_{tl}']['candidate'] else 'нет'}",
          flush=True)

sg = pd.DataFrame(seg_rows, columns=["base", "side", "quarter", "qr",
                                     "hblk", "dow", "atr_rel"])
atrb = np.quantile(sg["atr_rel"], [1/3, 2/3])
sg["atr_b"] = np.searchsorted(atrb, sg["atr_rel"]) + 1

def dump(name, col, order=None):
    g = sg.groupby(col)["base"]
    out = {}
    for key, v in g:
        out[str(key)] = {"n": int(len(v)),
                         "avg_bp": round(float(v.mean()) * 10000, 2)}
    report["segments"][name] = out
    line = " ".join(f"{k}:{v['avg_bp']:+.1f}(n={v['n']})"
                    for k, v in sorted(out.items(), key=lambda x: int(x[0]))
                    if order is None or str(x[0]) in order)
    print(f"{name}: {line}", flush=True)

dump("qrank", "qr")
dump("atr_b", "atr_b")
dump("hour_4h", "hblk")
dump("dow", "dow")
dump("side", "side")
d = sg.groupby("quarter")["base"]
report["segments"]["quarter"] = {str(k): {"n": int(len(v)),
                                          "avg_bp": round(float(v.mean()) * 10000, 2)}
                                 for k, v in d}
print("кварталы:", {k: v["avg_bp"] for k, v in
                    report["segments"]["quarter"].items()}, flush=True)

json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("РЕНТГЕН ЗАВЕРШЁН", flush=True)
