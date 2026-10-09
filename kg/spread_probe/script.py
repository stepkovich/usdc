"""СПРЕД-ЖАТВА НА 30с-МОДЕЛИ (предрегистрация 07.10, идея владельца).

ИДЕЯ: обучить модель ярлыку «цена выше через 30с» и войти мейкером
(бесплатно, по биду), тейк — лимиткой на «спред + 1 тик» от входа
(тоже бесплатно, мейкер). Обе ноги — GTX. Фильтр моментов — модель.
Сборщик спреда с умным фильтром.

СИМУЛЯЦИЯ ПО НАШИМ ЗАПИСЯМ (1с миды + спред; бид = mid − spread/2,
аск = mid + spread/2 — реконструкция из записи):
  вход: p30 > gate (модель обучена ярлыку 30с на днях до теста);
        лимитка на биде, TTL 60с: мид коснулся — исполнилась, иначе
        заявка снята (без убытка);
  тейк: вход + spread(на момент входа) + 0.2бп («+1 тик», аппрокс);
        живёт 120с: мид коснулся — прибыль;
  иначе (не дожались за 120с): выход рынком по последнему миду − 10бп;
  одна позиция на символ, после выхода пауза 60с.
ГАТЫ: 0.60 / 0.65 / 0.70. БАЗА: без фильтра (вход всегда, та же
механика) — измеряет чистую жатву спреда без ума.
ВЕРДИКТ (записан до прогона): связка жива, если на каком-то гате
avg >= +1 бп при n >= 300 И добавка к безфилтровой базе >= +1 бп.
Оговорка: миды 1с не видят очередь — заполнения аппроксимированы
касанием мидом уровня; реальная очередь может быть хуже.
"""
import glob, json, time
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
SP_LABEL_MS = 30_000
ENTRY_TTL_MS = 60_000
TP_TTL_MS = 120_000
PAUSE_MS = 60_000
TICK_BUF_BP = 0.2                  # «+1 тик» аппроксимация
GATES = [0.60, 0.65, 0.70]
FEE_TAKER = 0.0010
PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1)
N_TEST_DAYS = 4
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

tr_frames = []
for sym, g in book[book["day"] < test_days[0]].groupby("symbol"):
    if sym not in SYMS or len(g) < 3000:
        continue
    g = med_norm(g.copy())
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    idx = np.clip(ts.searchsorted(ts + SP_LABEL_MS), 0, len(g) - 1)
    ok = ts[idx] >= ts + SP_LABEL_MS - 1000
    g["y30"] = np.where(ok, mid[idx] > mid, np.nan)
    tr_frames.append(g[["ts", "symbol", "day"] + LOB_FEATS + ["y30"]])
train = pd.concat(tr_frames, ignore_index=True)
del tr_frames
train = train.dropna(subset=LOB_FEATS + ["y30"])
m30 = lgb.LGBMClassifier(**PARAMS)
m30.fit(train[LOB_FEATS], train["y30"])
print("30с-модель обучена на", len(train), flush=True)
del train

rows_all = []
for sym, g in book[book["day"].isin(test_days)].groupby("symbol"):
    if sym not in SYMS or len(g) < 3000:
        continue
    g = med_norm(g.sort_values("ts").reset_index(drop=True).copy())
    ts = g["ts"].values
    mid = g["mid"].values
    spread_bp = g["spread_bp"].values
    valid = g.dropna(subset=LOB_FEATS)
    p30 = np.full(len(g), np.nan)
    if len(valid) >= 200:
        p30[valid.index.values] = m30.predict_proba(valid[LOB_FEATS])[:, 1]
    d = pd.DataFrame({"ts": ts, "mid": mid, "spread_bp": spread_bp,
                      "p30": p30,
                      "day": pd.to_datetime(ts, unit="ms").date}) \
        .dropna(subset=["p30"]).reset_index(drop=True)
    ts = d["ts"].values
    mid = d["mid"].values
    sp = d["spread_bp"].values
    p30 = d["p30"].values
    day = d["day"].values
    n = len(d)
    i = 0
    while i < n:
        entry_px = mid[i] * (1 - sp[i] / 2 / 10000)
        dist_bp = sp[i] + TICK_BUF_BP
        tp_px = entry_px * (1 + dist_bp / 10000)
        # вход: касание бида в течение 60с
        j = i
        filled_at = None
        while j < n and ts[j] <= ts[i] + ENTRY_TTL_MS:
            if mid[j] <= entry_px:
                filled_at = j
                break
            j += 1
        if filled_at is None:
            i += 1
            continue
        # тейк: касание tp_px в течение 120с после входа
        k = filled_at
        tp_hit_at = None
        while k < n and ts[k] <= ts[filled_at] + TP_TTL_MS:
            if mid[k] >= tp_px:
                tp_hit_at = k
                break
            k += 1
        if tp_hit_at is not None:
            pnl = tp_px / entry_px - 1            # обе ноги мейкер = 0
        else:
            exit_px = mid[min(k, n - 1)]
            pnl = exit_px / entry_px - 1 - FEE_TAKER
        rec = {"sym": sym, "day": str(day[i]), "p30": float(p30[i]),
               "dist_bp": dist_bp, "pnl": pnl}
        for gt in GATES:
            rec[f"taken_{gt}"] = p30[i] > gt
        rows_all.append(rec)
        i = max(k, filled_at) + int(PAUSE_MS / 1000)

R = pd.DataFrame(rows_all)
print(f"сделок (без фильтра): {len(R)} | {time.time()-t0:.0f}с",
      flush=True)

def stat(df):
    if not len(df):
        return {"n": 0, "avg_bp": None}
    return {"n": int(len(df)),
            "avg_bp": round(float(df["pnl"].mean() * 10000), 2),
            "win_pct": round(float((df["pnl"] > 0).mean() * 100), 1)}

report = {"pre_reg": "жива: avg>=+1бп при n>=300 и добавка к базе>=+1бп",
          "levers": {}}
base = stat(R)
report["levers"]["БЕЗ_ФИЛЬТРА"] = base
print(f"БЕЗ ФИЛЬТРА: {base}", flush=True)
for gt in GATES:
    sub = R[R[f"taken_{gt}"]]
    st = stat(sub)
    delta = (st["avg_bp"] - base["avg_bp"]) if st.get("avg_bp") is not None \
        and base.get("avg_bp") is not None else None
    alive = bool(st.get("n", 0) >= 300 and st.get("avg_bp", -99) >= 1.0
                 and delta is not None and delta >= 1.0)
    report["levers"][f"gate_{gt}"] = {**st, "delta_vs_base":
                                      None if delta is None
                                      else round(delta, 2), "alive": alive}
    print(f"gate {gt}: {st} (Δ к базе {delta}) -> "
          f"{'ЖИВА' if alive else 'нет'}", flush=True)
by_day = {}
best = max((v["avg_bp"] for k, v in report["levers"].items()
            if k != "БЕЗ_ФИЛЬТРА" and v.get("avg_bp") is not None),
           default=-99)
for gt in GATES:
    if report["levers"][f"gate_{gt}"].get("avg_bp") == best \
            and best >= 1.0:
        sub = R[R[f"taken_{gt}"]]
        for day, g in sub.groupby("day"):
            by_day[f"gate{gt}|{day}"] = round(
                float(g["pnl"].mean() * 10000), 2)
report["by_day"] = by_day
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("СПРЕД-ЖАТВА завершена", flush=True)
