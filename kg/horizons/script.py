"""ГОРИЗОНТЫ СПРИНТЕРА: 30с / 1м / 2м / 3м / 5м (предрегистрация 07.10).

ИДЕЯ ВЛАДЕЛЬЦА: 5 минут для крипты — вечность; может, спринтеру
горизонт короче (30с..3м)? На наших записях (1с каденс, 37 USDC) это
чистый эксперимент: одна и та же модель, одни и те же дни, отличается
ТОЛЬКО ярлык (mid через T секунд выше/ниже).

ГОРИЗОНТЫ: 30с, 60с, 120с, 180с, 300с (300с = база, как в бою).
МЕТРИКА 1 (навык): квинтильный разрыв fwd (бп, мид-мид, без издержек),
пул последних 4 полных дней, модель обучается на всех днях до теста.
МЕТРИКА 2 (экономика): сделочная симуляция (гейт 0.62, пауза 300с
после выхода, обе стороны): вход мейкер 0; выход — ДВА сценария:
  MID  = закрытие по миду на горизонте (идеальный мейкер, ~0);
  TAKER = закрытие рынком на горизонте (10бп).
ВЕРДИКТ (записан до прогона): горизонт ИНТЕРЕСЕН, если пул-навык >=
3 бп И avg_bp(TAKER) >= 0 (живёт даже без идеального выхода), и он
строже базового. Иначе — 5м остаётся. Ничего не деплоится.
"""
import glob, json, time
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
SP_GATE = 0.62
PAUSE_MS = 300_000
FEE_TAKER = 0.0010
PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1)
N_TEST_DAYS = 4
HORIZONS = [30_000, 60_000, 120_000, 180_000, 300_000]
NAMES = {30_000: "30с", 60_000: "1м", 120_000: "2м", 180_000: "3м",
         300_000: "5м"}
SYMS = ["1000BONKUSDC", "1000PEPEUSDC", "1000SHIBUSDC", "AAVEUSDC",
        "ADAUSDC", "ARBUSDC", "AVAXUSDC", "BCHUSDC", "BIOUSDC",
        "BNBUSDC", "BOMEUSDC", "BTCUSDC", "CRVUSDC", "DATAIPUSDC",
        "DOGEUSDC", "ENAUSDC", "ETHFIUSDC", "ETHUSDC", "FILUSDC",
        "HBARUSDC", "KAITOUSDC", "LINKUSDC", "LTCUSDC", "NEARUSDC",
        "NEOUSDC", "ORDIUSDC", "PENGUUSDC", "PNUTUSDC", "SOLUSDC",
        "SUIUSDC", "TIAUSDC", "TRUMPUSDC", "UNIUSDC", "WIFUSDC",
        "WLDUSDC", "WLFIUSDC", "XRPUSDC", "ZECUSDC"]
LOB_FEATS = ["spread_bp", "microprice_rel", "imb5", "imb10", "imb20",
             "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
             "ntr10", "vpin10", "d30", "d120"]

book_frames = []
for f in sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True)):
    try:
        bf = pd.read_parquet(f, columns=["ts", "symbol", "mid",
                                         "spread_bp", "microprice",
                                         "imb5", "imb10", "imb20",
                                         "flow10_buy", "flow10_sell",
                                         "flow60_buy", "flow60_sell",
                                         "ntr10", "vpin10"])
        book_frames.append(bf.iloc[::2])          # 1с каденс
    except Exception:
        pass
book = pd.concat(book_frames, ignore_index=True)
del book_frames
book["day"] = pd.to_datetime(book["ts"], unit="ms").dt.date.astype(str)
days = sorted(book["day"].unique())
test_days = days[-N_TEST_DAYS:]
print(f"строк {len(book)}, дней {len(days)}, тест {test_days} "
      f"| {time.time()-t0:.0f}с", flush=True)

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

train_frames = []
for sym, g in book[book["day"] < test_days[0]].groupby("symbol"):
    if sym not in SYMS or len(g) < 3000:
        continue
    g = med_norm(g.copy())
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    idx = np.clip(ts.searchsorted(ts + 300_000), 0, len(g) - 1)
    ok = ts[idx] >= ts + 300_000 - 1500
    g["y5"] = np.where(ok, mid[idx] > mid, np.nan)
    train_frames.append(g[["ts", "symbol", "day"] + LOB_FEATS + ["y5"]])
train = pd.concat(train_frames, ignore_index=True)
del train_frames
train = train.dropna(subset=LOB_FEATS + ["y5"])
m5 = lgb.LGBMClassifier(**PARAMS)
m5.fit(train[LOB_FEATS], train["y5"])
print("модель (5м ярлык, как в бою) обучена на", len(train), flush=True)
del train

# ---------- тест: одна таблица p5 + fwd для всех горизонтов ----------
tp_frames = []
for sym, g in book[book["day"].isin(test_days)].groupby("symbol"):
    if sym not in SYMS or len(g) < 3000:
        continue
    g = med_norm(g.sort_values("ts").reset_index(drop=True).copy())
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    rec = {"ts": ts, "mid": mid, "spread_bp": g["spread_bp"].values}
    ok_any = np.zeros(len(g), dtype=bool)
    for h in HORIZONS:
        idx = np.clip(ts.searchsorted(ts + h), 0, len(g) - 1)
        ok = ts[idx] >= ts + h - 1500
        rec[f"fwd_{h}"] = np.where(ok, mid[idx] / mid - 1, np.nan)
        ok_any |= ok
    p5 = np.full(len(g), np.nan)
    valid = g.iloc[:np.nonzero(ok_any)[0][-1] + 1 if ok_any.any() else 0]
    valid = valid.dropna(subset=LOB_FEATS)
    if len(valid) >= 200:
        p5[valid.index.values] = m5.predict_proba(valid[LOB_FEATS])[:, 1]
    rec["p5"] = p5
    rec["sym"] = sym
    rec["day"] = g["day"].values
    tp_frames.append(pd.DataFrame(rec))
del book
T = pd.concat(tp_frames, ignore_index=True)
del tp_frames
print(f"тест-строк {len(T)} | {time.time()-t0:.0f}с", flush=True)

def trade_sim(d, horizon):
    """Сделки на горизонте: вход p5-гейт, выход на +horizon (по fwd),
    пауза 300с; mid-выход (метрика fwd) и тейкер-выход (−10бп)."""
    d = d.dropna(subset=[f"fwd_{horizon}", "p5"]).sort_values("ts") \
        .reset_index(drop=True)
    ts = d["ts"].values
    fwd = d[f"fwd_{horizon}"].values.astype("float64")
    p = d["p5"].values
    n = len(d)
    i = 0
    rows = []
    while i < n:
        sig = 0
        if p[i] >= SP_GATE:
            sig = 1
        elif p[i] <= 1 - SP_GATE:
            sig = -1
        if sig == 0 or np.isnan(fwd[i]):
            i += 1
            continue
        rows.append({"fwd": sig * fwd[i], "side": sig})
        i += max(1, int(300_000 / 1000))     # пауза 300с (1с строки)
    return pd.DataFrame(rows)

report = {"horizons": {}}
pool = {}
for h in HORIZONS:
    col = f"fwd_{h}"
    d = T.dropna(subset=[col, "p5"])
    ok = ~np.isnan(d[col].values)
    dd = d[ok]
    p = dd["p5"].values
    fwd = dd[col].values.astype("float64")
    q20, q80 = np.quantile(p, [0.2, 0.8])
    sk = float((fwd[p >= q80].mean() - fwd[p <= q20].mean()) * 10000)
    acc = float(((p > 0.5) == (fwd > 0)).mean() * 100)
    # сделочная симуляция
    tr = trade_sim(dd, h)
    avg_mid = float(tr["fwd"].mean() * 10000) if len(tr) else None
    avg_tk = float((tr["fwd"] - FEE_TAKER).mean() * 10000) if len(tr) \
        else None
    report["horizons"][NAMES[h]] = {
        "n_rows": int(len(dd)), "acc": round(acc, 2),
        "skill_bp": round(sk, 2),
        "n_trades": int(len(tr)),
        "avg_mid_bp": None if avg_mid is None else round(avg_mid, 2),
        "avg_taker_bp": None if avg_tk is None else round(avg_tk, 2)}
    print(f"{NAMES[h]:4s}: acc {acc:.2f}% навык {sk:+.1f} бп | "
          f"{len(tr)} сделок: mid {avg_mid:+.1f} бп, тейкер {avg_tk:+.1f} бп"
          f" | {time.time()-t0:.0f}с", flush=True)

sk5 = report["horizons"]["5м"]["skill_bp"]
best_short = None
for nm, v in report["horizons"].items():
    if nm == "5м":
        continue
    if v["skill_bp"] >= 3.0 and (v["avg_taker_bp"] or -99) >= 0:
        best_short = nm
report["verdict"] = (f"короткий горизонт интересен: {best_short}"
                     if best_short else
                     "короче 5м не лучше — 5м остаётся")
print("ВЕРДИКТ:", report["verdict"], flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
