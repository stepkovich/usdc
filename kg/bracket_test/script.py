"""БРАСЛЕТ МЕЙКЕРАМИ (предрегистрация 07.10, идея владельца, «по классике»).

СХЕМА: вход мейкером по биду (GTX, 0). После входа — браслет:
  ТЕЙК: GTX-лимитка на entry + (spread + 1 тик);
  СТОП: уровень entry − (spread − 1 тик); при пробое ГАРАНТИРОВАННО
        ставим GTX-лимитку НА уровне стопа (она мейкерская, 0) и ждём
        возврата цены до конца окна; не вернулась — выход рынком
        по последней цене (честный хвост, 10 бп).
ПАРАМЕТР RR (прибыль : риск): тейк = RR × стоп-дистанция; RR = 1 / 2 / 3
(владелец просил 2:1 — «прибыль в два раза больше проигрыша»).
ГАТЫ: без фильтра + 0.60 / 0.65 (30с-модель).
Окно: 300с на весь браслет. Вход: касание бида за 60с, иначе отмена.
ВЕРДИКТ (записан до прогона): схема жива, если RR=2 при каком-то гате
avg >= +1 бп при n >= 300. Ничего не деплоится. Оговорка: стоп-мейкер
в симуляции 1с-мидов оптимистичен (не видит очередь).
"""
import glob, json, time
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
SP_LABEL_MS = 30_000
ENTRY_TTL_MS = 60_000
WINDOW_MS = 300_000
SL_BUF_BP = 0.2                    # «−1 тик» аппроксимация
FEE_TAKER = 0.0010
RATIOS = [1.0, 2.0, 3.0]
GATES = [None, 0.60, 0.65]
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
        book_frames.append(bf.iloc[::2])
    except Exception:
        pass
book = pd.concat(book_frames, ignore_index=True)
del book_frames
book["day"] = pd.to_datetime(book["ts"], unit="ms").dt.date.astype(str)
days = sorted(book["day"].unique())
test_days = days[-N_TEST_DAYS:]
print(f"строк {len(book)}, тест {test_days} | {time.time()-t0:.0f}с",
      flush=True)

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
    sp = g["spread_bp"].values
    valid = g.dropna(subset=LOB_FEATS)
    p30 = np.full(len(g), np.nan)
    if len(valid) >= 200:
        p30[valid.index.values] = m30.predict_proba(valid[LOB_FEATS])[:, 1]
    d = pd.DataFrame({"ts": ts, "mid": mid, "sp": sp, "p30": p30,
                      "day": pd.to_datetime(ts, unit="ms").date}) \
        .dropna(subset=["p30"]).reset_index(drop=True)
    ts = d["ts"].values
    mid = d["mid"].values
    sp = d["sp"].values
    p30 = d["p30"].values
    day = d["day"].values
    n = len(d)
    i = 0
    while i < n:
        entry_px = mid[i] * (1 - sp[i] / 2 / 10000)
        sl_dist_bp = max(sp[i] - SL_BUF_BP, 0.5)
        # вход: касание бида за 60с
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
        f_ts = ts[filled_at]
        end_ts = f_ts + WINDOW_MS
        for RR in RATIOS:
            tp_dist_bp = RR * sl_dist_bp
            tp_px = entry_px * (1 + tp_dist_bp / 10000)
            sl_px = entry_px * (1 - sl_dist_bp / 10000)
            # проход по секундам
            k = filled_at
            outcome = "timeout"
            sl_armed = False
            pnl = np.nan
            while k < n and ts[k] <= end_ts:
                if mid[k] >= tp_px:
                    outcome = "tp"
                    pnl = tp_px / entry_px - 1
                    break
                if not sl_armed and mid[k] <= sl_px:
                    sl_armed = True          # ставим GTX-лимитку на sl_px
                if sl_armed and mid[k] >= sl_px:
                    outcome = "sl_filled"
                    pnl = sl_px / entry_px - 1
                    break
                k += 1
            if outcome == "timeout":
                last_px = mid[min(k, n - 1)]
                pnl = last_px / entry_px - 1 - FEE_TAKER
                outcome = "sl_stuck" if sl_armed else "timeout"
            rec = {"sym": sym, "day": str(day[i]), "RR": RR,
                   "side_dist": sl_dist_bp, "p30": float(p30[i]),
                   "pnl": pnl, "outcome": outcome,
                   "gate_060": float(p30[i]) > 0.60,
                   "gate_065": float(p30[i]) > 0.65}
            rows_all.append(rec)
        i = k + 60                    # пауза 60с (1с строки)

R = pd.DataFrame(rows_all)
print(f"циклов браслета: {len(R)} | {time.time()-t0:.0f}с", flush=True)

report = {"pre_reg": "жива: RR=2, avg>=+1бп, n>=300 (какой-то гат)",
          "combos": {}}
for RR in RATIOS:
    for gname, gcol in (("БЕЗ_ФИЛЬТРА", None), ("gate_0.60", "gate_060"),
                        ("gate_0.65", "gate_065")):
        sub = R[R["RR"] == RR]
        if gcol is not None:
            sub = sub[sub[gcol]]
        if not len(sub):
            continue
        avg = float(sub["pnl"].mean() * 10000)
        oc = sub["outcome"].value_counts(normalize=True) * 100
        key = f"RR{int(RR)}|{gname}"
        alive = bool(RR == 2.0 and avg >= 1.0 and len(sub) >= 300)
        report["combos"][key] = {
            "n": int(len(sub)), "avg_bp": round(avg, 2),
            "tp_pct": round(float(oc.get("tp", 0)), 1),
            "sl_fill_pct": round(float(oc.get("sl_filled", 0)), 1),
            "stuck_pct": round(float(oc.get("sl_stuck", 0)), 1),
            "timeout_pct": round(float(oc.get("timeout", 0)), 1),
            "alive": alive if RR == 2.0 else None}
        print(f"RR{int(RR)} {gname:11s}: {len(sub):6d} циклов, "
              f"avg {avg:+7.2f} бп | тейк {report['combos'][key]['tp_pct']}% "
              f"стоп-филл {report['combos'][key]['sl_fill_pct']}% "
              f"застрял {report['combos'][key]['stuck_pct']}% "
              f"таймаут {report['combos'][key]['timeout_pct']}%",
              flush=True)
any_alive = any(v.get("alive") for v in report["combos"].values())
report["alive_RR2"] = any_alive
print("ВЕРДИКТ (RR=2):", "СХЕМА ЖИВА" if any_alive else "нет", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("БРАСЛЕТ завершён", flush=True)
