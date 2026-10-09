"""ЧАСЫ СУТОК ДЛЯ СПРИНТЕРА (предрегистрация 07.10, вариант 3 пути).

ВОПРОС: навык 5-мин направления (модель как в бою) по ЧАСАМ UTC —
концентрируется ли он во времени? Если да — фильтр часов бесплатный
(торгуем только умные часы, спим глухие).

МЕТОД: модель (5м ярлык, LOB_FEATS, как в бою) обучается на днях до
тестовых; на тестовых (последние 4 полных) предсказывает p5 каждой
строке. По каждому часу UTC: n, acc, квинтильный навык fwd5 (бп,
мид-мид). Сессии: Азия 00-08, Европа 08-16, Америка 16-24.
Плюс дни недели (7 бакетов).
ВЕРДИКТ (записан до прогона): фильтр часов ИНТЕРЕСЕН, если найдётся
блок >= 6 подряд часов с навыком >= +2 бп при n >= 10000 в часе.
Оговорка: 4 тест-дня = каждый час виден 4 раза — концентрация может
быть шумом; кандидата проверяем на свежих днях. Ничего не деплоится.
"""
import glob, json, time
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
SP_GATE = 0.62
N_TEST_DAYS = 4
FEE_TAKER = 0.0010
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
book = book.dropna(subset=["mid", "spread_bp"])
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

tr_frames = []
for sym, g in book[book["day"] < test_days[0]].groupby("symbol"):
    if sym not in SYMS or len(g) < 3000:
        continue
    g = med_norm(g.copy())
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    idx = np.clip(ts.searchsorted(ts + 300_000), 0, len(g) - 1)
    ok = ts[idx] >= ts + 300_000 - 1500
    g["fwd5"] = np.where(ok, mid[idx] / mid - 1, np.nan)
    g["y5"] = np.where(np.isnan(g["fwd5"]), np.nan,
                       (g["fwd5"] > 0).astype("float32"))
    tr_frames.append(g[["ts", "symbol", "day"] + LOB_FEATS + ["fwd5",
                                                              "y5"]])
train = pd.concat(tr_frames, ignore_index=True)
del tr_frames
train = train.dropna(subset=LOB_FEATS + ["y5"])
m5 = lgb.LGBMClassifier(**PARAMS)
m5.fit(train[LOB_FEATS], train["y5"])
print("модель обучена на", len(train), flush=True)
del train

te_frames = []
for sym, g in book[book["day"].isin(test_days)].groupby("symbol"):
    if sym not in SYMS or len(g) < 3000:
        continue
    g = med_norm(g.sort_values("ts").reset_index(drop=True).copy())
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    idx = np.clip(ts.searchsorted(ts + 300_000), 0, len(g) - 1)
    ok = ts[idx] >= ts + 300_000 - 1500
    g["fwd5"] = np.where(ok, mid[idx] / mid - 1, np.nan)
    te_frames.append(g[["ts", "mid"] + LOB_FEATS + ["fwd5"]])
del book
T = pd.concat(te_frames, ignore_index=True)
del te_frames
T = T.dropna(subset=LOB_FEATS + ["fwd5"]).reset_index(drop=True)
T["p5"] = m5.predict_proba(T[LOB_FEATS])[:, 1]
T["hour"] = pd.to_datetime(T["ts"], unit="ms").dt.hour
T["dow"] = pd.to_datetime(T["ts"], unit="ms").dt.dayofweek
print(f"тест-строк {len(T)} | {time.time()-t0:.0f}с", flush=True)

def qskill(d):
    p = d["p5"].values
    f = d["fwd5"].values.astype("float64")
    if len(d) < 2000:
        return None
    q20, q80 = np.quantile(p, [0.2, 0.8])
    return float((f[p >= q80].mean() - f[p <= q20].mean()) * 10000)

report = {"pre_reg": "фильтр часов интересен: блок >= 6 подряд часов "
                     "с навыком >= +2 бп при n >= 10000",
          "hours": {}, "sessions": {}, "dow": {}}
print("=== навык по часам UTC ===", flush=True)
hour_skills = {}
for h in range(24):
    d = T[T["hour"] == h]
    if len(d) < 500:
        print(f"  {h:02d}:00 — мало строк ({len(d)}), пропуск", flush=True)
        continue
    p = d["p5"].values
    f = d["fwd5"].values.astype("float64")
    q20, q80 = np.quantile(p, [0.2, 0.8])
    sk = float((f[p >= q80].mean() - f[p <= q20].mean()) * 10000)
    acc = float(((p > 0.5) == (f > 0)).mean() * 100)
    hour_skills[h] = (sk, len(d), acc)
    print(f"  {h:02d}:00 навык {sk:+6.2f} бп (n={len(d)})",
          flush=True)
    report["hours"][f"{h:02d}"] = {"skill_bp": round(sk, 2),
                                   "n": int(len(d)),
                                   "acc": round(acc, 2)}

blocks = []
cur = []
for h in range(24):
    sk = hour_skills[h][0]
    nn = hour_skills[h][1]
    if sk >= 2.0 and nn >= 10000:
        cur.append(h)
    else:
        if len(cur) >= 6:
            blocks.append(cur)
        cur = []
if len(cur) >= 6:
    blocks.append(cur)
report["good_blocks"] = blocks
print("БЛОКИ >= 6 часов с навыком >= +2 бп:", blocks, flush=True)

sess = {"Азия(00-08)": range(0, 8), "Европа(08-16)": range(8, 16),
        "Америка(16-24)": range(16, 24)}
for nm, rng in sess.items():
    d = T[T["hour"].isin(rng)]
    p = d["p5"].values
    f = d["fwd5"].values.astype("float64")
    q20, q80 = np.quantile(p, [0.2, 0.8])
    sk = float((f[p >= q80].mean() - f[p <= q20].mean()) * 10000)
    report["sessions"][nm] = {"skill_bp": round(sk, 2),
                              "n": int(len(d))}
    print(f"{nm}: навык {sk:+.2f} бп (n={len(d)})", flush=True)

print("=== дни недели ===", flush=True)
for dw in range(7):
    d = T[T["dow"] == dw]
    if len(d) < 500:
        continue
    p = d["p5"].values
    f = d["fwd5"].values.astype("float64")
    q20, q80 = np.quantile(p, [0.2, 0.8])
    sk = float((f[p >= q80].mean() - f[p <= q20].mean()) * 10000)
    report["dow"][str(dw)] = {"skill_bp": round(sk, 2), "n": int(len(d))}
    print(f"  dow {dw}: навык {sk:+6.2f} бп (n={len(d)})", flush=True)

json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ЧАСЫ СУТОК завершены", flush=True)
