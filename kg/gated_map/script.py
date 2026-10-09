"""ГЕЙТ-КАРТА: жатва спреда + ФИЛЬТР 30с-МОДЕЛИ (предрегистрация 07.10).

ШАГ 2 идеи владельца. Шаг 1 (карта без фильтра): все 27 ячеек
(тейк x стоп x время) слегка минусовые, лучшая (2.0/0.5/300с) =
-0.12/-0.27 бп — «механический налог» жатвы. ШАГ 2: налог 0.3 бп
МЕНЬШЕ навыка 30с-модели (0.4-0.8 бп направление) — проверяем,
перебивает ли фильтр уверенности налог.

СЕТКА: тейк {0.5,1,2} x spread, стоп {0.5,1,2} x spread,
время {60,120,300}с. Цикл: вход мейкером на биде (касание за 60с),
первое касание тейка/стопа -> мейкер 0, таймаут -> рынок (10 бп).
МОДЕЛЬ: ярлык «мид выше через 30с», обучение на днях до теста,
p30 на тест-днях. ГЕЙТЫ: без фильтра / p30>0.60 / p30>0.65.
ВЕРДИКТ (записан до прогона): связка жива, если ячейка 2.0/0.5/300
при каком-то гайте даёт avg >= +0.3 бп при n >= 200.
"""
import glob, json, time
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
N_TEST_DAYS = 4
PAUSE_S = 300
FEE_TAKER = 0.0010
TPK = [0.5, 1.0, 2.0]
SLK = [0.5, 1.0, 2.0]
GATES = [None, 0.60, 0.65]
KEY = (2.0, 0.5, 300)
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

# ---------- 30с-модель на train-днях ----------
tr_frames = []
for sym, g in book[book["day"] < test_days[0]].groupby("symbol"):
    if sym not in SYMS or len(g) < 3000:
        continue
    g = med_norm(g.copy())
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    idx = np.clip(ts.searchsorted(ts + 30_000), 0, len(g) - 1)
    ok = ts[idx] >= ts + 30_000 - 1000
    g["y30"] = np.where(ok, mid[idx] > mid, np.nan)
    tr_frames.append(g[["ts", "symbol", "day"] + LOB_FEATS + ["y30"]])
train = pd.concat(tr_frames, ignore_index=True)
del tr_frames
train = train.dropna(subset=LOB_FEATS + ["y30"])
m30 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05,
                         max_depth=4, subsample=0.8, colsample_bytree=0.8,
                         random_state=42, n_jobs=4, verbosity=-1)
m30.fit(train[LOB_FEATS], train["y30"])
print("30с-модель обучена на", len(train), flush=True)
del train

# ---------- тест-дни: p30 + сбор циклов ----------
BID_K = 0.5                       # вход на биде = mid - spread/2
C = []
for sym, g in book[book["day"].isin(test_days)].groupby("symbol"):
    if sym not in SYMS or len(g) < 3000:
        continue
    g = med_norm(g.sort_values("ts").reset_index(drop=True).copy())
    p30 = np.full(len(g), np.nan)
    v = g.dropna(subset=LOB_FEATS)
    p30[v.index.values] = m30.predict_proba(v[LOB_FEATS])[:, 1]
    d = pd.DataFrame({"ts": g["ts"].values, "mid": g["mid"].values,
                      "spread_bp": g["spread_bp"].values, "p30": p30,
                      "day": g["day"].values}) \
        .dropna(subset=["p30"]).reset_index(drop=True)
    ts = d["ts"].values
    mid = d["mid"].values.astype("float64")
    sp_bp = d["spread_bp"].values
    p30 = d["p30"].values
    day = d["day"].values
    n = len(d)
    i = 0
    while i < n:
        s_bp = sp_bp[i]
        entry_px = mid[i] * (1 - s_bp * 0.5 / 1e4)     # бид = mid-spread/2
        j = i
        f = None
        while j < n and ts[j] <= ts[i] + 60_000:
            if mid[j] <= entry_px:
                f = j
                break
            j += 1
        if f is None:
            i += 1
            continue
        k_end = int(np.searchsorted(ts, ts[f] + 345_000))
        s_frac = s_bp / 1e4
        tp_idx = []
        for k in TPK:
            w = np.nonzero(mid[f:k_end] >= entry_px * (1 + k * s_frac))[0]
            tp_idx.append(int(w[0]) if len(w) else 10**9)
        sl_idx = []
        for k in SLK:
            w = np.nonzero(mid[f:k_end] <= entry_px * (1 - k * s_frac))[0]
            sl_idx.append(int(w[0]) if len(w) else 10**9)
        last = float(mid[k_end - 1]) if k_end > f else np.nan
        C.append({"sym": sym, "day": str(day[i]), "entry": entry_px,
                  "s_bp": s_bp, "plen": k_end - f, "last": last,
                  "p30": float(p30[f]),
                  "tp_idx": tp_idx, "sl_idx": sl_idx})
        i = k_end + PAUSE_S * 2
print(f"циклов {len(C)} | {time.time()-t0:.0f}с", flush=True)

def eval_cell(cycles, tp_k, sl_k, ttl_s, gate_min):
    """Тейк-первым -> +tp_k x spread; стоп-первым -> -sl_k x spread;
    таймаут -> рынок (минус 10 бп). Гейт отсеивает p30 <= gate_min."""
    tp_i = TPK.index(tp_k)
    sl_i = SLK.index(sl_k)
    pnls = []
    for c in cycles:
        if gate_min is not None and c["p30"] <= gate_min:
            continue
        lim = min(ttl_s, c["plen"])
        t = c["tp_idx"][tp_i]
        u = c["sl_idx"][sl_i]
        t_hit = t < lim
        u_hit = u < lim
        if t_hit and (not u_hit or t <= u):
            pnls.append(tp_k * c["s_bp"] / 1e4)       # +тейк, мейкер 0
        elif u_hit and (not t_hit or u < t):
            pnls.append(-sl_k * c["s_bp"] / 1e4)      # -стоп, мейкер 0
        else:
            pnls.append((c["last"] / c["entry"] - 1) - FEE_TAKER)
    return pnls

report = {"pre_reg": "связка жива: ячейка 2.0/0.5/300 при гейте "
                     "avg >= +0.3 бп, n >= 200",
          "combos": {}}
for gname, gmin in (("БЕЗ_ФИЛЬТРА", None), ("gate_0.60", 0.60),
                    ("gate_0.65", 0.65)):
    pn = eval_cell(C, KEY[0], KEY[1], KEY[2], gmin)
    st = {"n": int(len(pn))}
    if pn:
        st["avg_bp"] = round(float(np.mean(pn) * 10000), 2)
        st["win_pct"] = round(float(np.mean(np.array(pn) > 0) * 100), 1)
        st["candidate"] = bool(st["avg_bp"] >= 0.3 and st["n"] >= 200)
    else:
        st["avg_bp"] = None
        st["candidate"] = False
    report["combos"][gname] = st
    print(f"{gname:12s}: {st.get('n')} сделок, avg {st.get('avg_bp')} бп, "
          f"win {st.get('win_pct')}% -> "
          f"{'КАНДИДАТ' if st.get('candidate') else ''}", flush=True)

alive = any(report["combos"][g].get("candidate")
            for g in report["combos"])
report["alive"] = alive
print("ВЕРДИКТ:", "СВЯЗКА ЖИВА" if alive else "нет", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ГЕЙТ-КАРТА завершена", flush=True)
