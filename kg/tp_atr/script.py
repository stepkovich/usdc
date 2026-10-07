"""ТЕЙК ПО ATR + БУДИЛЬНИК (предрегистрация 07.10, идея владельца).

ВОПРОС: сейчас выход один — будильник 300с (лимитка на середине
спреда, TTL 45с, иначе рынок). Что если сразу после входа ставить
ТЕЙК-ЛИМИТКУ на уровне k×ATR(5м) от входа? Победители закрываются
мейкером (0), остальные — будильником.

ВАРИАНТЫ (одни сделки, отличается только тейк):
  NOW      как в бою: будильник на mid, TTL 45с, иначе рынок;
  TP025    тейк на 0.25×ATR + будильник;
  TP05     тейк на 0.5×ATR + будильник;
  TP10     тейк на 1.0×ATR + будильник;
  TP20     тейк на 2.0×ATR + будильник.
ATR = 14×5м бары из НАШЕЙ записи mid (high/low/close по 5м бинам),
значение на закрытии бара ПЕРЕД входом. Тейк живёт весь холд 300с:
касание мидом -> мейкер по тейк-цене; иначе будильник (mid, TTL 45с,
иначе рынок 10бп).
ВЕРДИКТ (записан до прогона): вариант — кандидат, если добавляет
>= +2 бп к NOW И доля тейк-заполнений >= 40%. Ничего не деплоится.
"""
import glob, json, time
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
N_TEST_DAYS = 4
SP_GATE = 0.62
HOLD_MS = 300_000
BASE_TTL = 45_000
FEE_TAKER_OUT = 0.0010
PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1)
KS = [0.25, 0.5, 1.0, 2.0]
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
print(f"книжных строк {len(book)}, дней {len(days)}, тест {test_days}",
      flush=True)

# ---------- ATR по 5м бинам НАШЕГО mid ----------
book["bin5"] = (book["ts"] // 300_000) * 300_000
b5 = book.groupby(["symbol", "bin5"])["mid"].agg(["max", "min", "last"])
b5 = b5.reset_index().rename(columns={"max": "h", "min": "l",
                                      "last": "c"})
b5["pc"] = b5.groupby("symbol")["c"].shift(1)
tr = np.maximum(b5["h"] - b5["l"],
                np.maximum((b5["h"] - b5["pc"]).abs(),
                           (b5["l"] - b5["pc"]).abs()))
b5["atr"] = tr.rolling(14, min_periods=8).mean()
b5["atr_bp"] = (b5["atr"] / b5["c"] * 10000).astype("float32")
b5["ot"] = (b5["bin5"] + 300_000).astype("int64")   # закрытие бара
ATR = b5[["symbol", "ot", "atr_bp"]]
del b5
print(f"ATR-бинов {len(ATR)} | {time.time()-t0:.0f}с", flush=True)

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
    idx = np.clip(ts.searchsorted(ts + HOLD_MS), 0, len(g) - 1)
    ok = ts[idx] >= ts + HOLD_MS - 1500
    g["fwd5"] = np.where(ok, mid[idx] / mid - 1, np.nan)
    g["y"] = np.where(np.isnan(g["fwd5"]), np.nan,
                      (g["fwd5"] > 0).astype("float32"))
    tr_frames.append(g)
train = pd.concat(tr_frames, ignore_index=True)
del tr_frames
train = train.dropna(subset=LOB_FEATS + ["y"])
m5 = lgb.LGBMClassifier(**PARAMS)
m5.fit(train[LOB_FEATS], train["y"])
print("спринтер-модель обучена на", len(train), flush=True)
del train

def alarm_exit(m_ts, m_mid, e, side, t_exit, ttl):
    m = (m_ts >= t_exit) & (m_ts <= t_exit + ttl)
    seg = m_mid[m]
    if len(seg) == 0:
        return None, None, None
    exit_px = seg[-1] * (1 - 0.0005) if side == 1 else seg[-1] \
        * (1 + 0.0005)
    return exit_px, False, FEE_TAKER_OUT

def touch(m_ts, m_mid, px, side, t0, t1):
    m = (m_ts >= t0) & (m_ts <= t1)
    seg = m_mid[m]
    if len(seg) == 0:
        return False
    return bool((seg >= px).any()) if side == 1 else bool((seg <= px).any())

rows_out = []
for sym, g in book[book["day"].isin(test_days)].groupby("symbol"):
    if sym not in SYMS or len(g) < 3000:
        continue
    g = med_norm(g.sort_values("ts").reset_index(drop=True).copy())
    ts = g["ts"].values
    mid = g["mid"].values
    spread = g["spread_bp"].values / 10000.0
    idx = np.clip(ts.searchsorted(ts + HOLD_MS), 0, len(g) - 1)
    ok = ts[idx] >= ts + HOLD_MS - 1500
    fwd5 = np.where(ok, mid[idx] / mid - 1, np.nan)
    valid = g.dropna(subset=LOB_FEATS)
    p5 = np.full(len(g), np.nan)
    if len(valid) >= 200:
        p5[valid.index.values] = m5.predict_proba(valid[LOB_FEATS])[:, 1]
    ga = g.assign(day=g["day"]).merge(ATR[ATR["symbol"] == sym],
                                      left_on=(g["ts"].values
                                               // 300_000) * 300_000
                                      + 300_000, right_on="ot",
                                      how="left")
    d = pd.DataFrame({"ts": ts, "mid": mid, "spread": spread,
                      "p5": p5, "fwd5": fwd5,
                      "atr_bp": ga["atr_bp"].values})
    d["day"] = pd.to_datetime(d["ts"], unit="ms").dt.date
    d = d.dropna(subset=["p5", "fwd5"]).reset_index(drop=True)
    ts = d["ts"].values
    mid = d["mid"].values
    spread = d["spread"].values
    p5 = d["p5"].values
    fwd5 = d["fwd5"].values
    atr_bp = d["atr_bp"].values
    n = len(d)
    i = 0
    while i < n:
        sig = 0
        if p5[i] >= SP_GATE:
            sig = 1
        elif p5[i] <= 1 - SP_GATE:
            sig = -1
        if sig == 0 or np.isnan(atr_bp[i]):
            i += 1
            continue
        e = mid[i]
        t_exit = ts[i] + HOLD_MS
        j_exit = int(np.searchsorted(ts, t_exit))
        if j_exit >= n:
            break
        m_ts = ts[j_exit: j_exit + 400]
        m_mid = mid[j_exit: j_exit + 400]
        m_all_ts = ts[i: j_exit + 400]
        m_all_mid = mid[i: j_exit + 400]
        half = spread[j_exit] / 2.0
        alarm_px = mid[j_exit]
        ex_a, _, fee_a, _ = (None, None, None, None)
        # NOW: будильник на mid + TTL 45
        m = (m_ts >= t_exit) & (m_ts <= t_exit + BASE_TTL)
        seg = m_mid[m]
        if len(seg) == 0:
            i = j_exit + 300
            continue
        exit_px = seg[-1] * (1 - 0.0005) if sig == 1 else seg[-1] \
            * (1 + 0.0005)
        pnl_now = sig * (exit_px / e - 1) - FEE_TAKER_OUT
        rec = {"sym": sym, "day": d["day"].iloc[i], "side": sig,
               "atr_bp": float(atr_bp[i]), "pnl_now": pnl_now}
        for k in KS:
            key = f"TP{int(k*100)}"
            tp_px = e * ((1 + k * atr_bp[i] / 10000) if sig == 1
                         else (1 - k * atr_bp[i] / 10000))
            hit = touch(m_all_ts, m_all_mid, tp_px, sig, ts[i], t_exit)
            if hit:
                pnl = sig * (tp_px / e - 1) if sig == 1 \
                    else (1 - tp_px / e) * 1.0
                pnl = sig * 0 + ((tp_px / e - 1) if sig == 1
                                 else (1 - tp_px / e))
            else:
                m2 = (m_ts >= t_exit) & (m_ts <= t_exit + BASE_TTL)
                seg2 = m_mid[m2]
                if len(seg2) == 0:
                    rec[key] = np.nan
                    continue
                exit_px = seg2[-1] * (1 - 0.0005) if sig == 1 \
                    else seg2[-1] * (1 + 0.0005)
                pnl = sig * (exit_px / e - 1) - FEE_TAKER_OUT
            rec[key] = pnl
            rec[f"fill_{key}"] = hit
        rows_out.append(rec)
        i = j_exit + 300

R = pd.DataFrame(rows_out)
print(f"сделок в симуляции: {len(R)} | {time.time()-t0:.0f}с", flush=True)
report = {"pre_reg": "кандидат: добавка >= +2бп к NOW и fill>=40%",
          "levers": {}}
base_avg = float(R["pnl_now"].mean() * 10000)
for name, col, key in (("NOW", "pnl_now", None),
                       ("TP025", None, "fill_TP25"),
                       ("TP05", None, "fill_TP50"),
                       ("TP10", None, "fill_TP100"),
                       ("TP20", None, "fill_TP200")):
    if name == "NOW":
        s = R.dropna(subset=["pnl_now"])
        avg = float(s["pnl_now"].mean() * 10000)
        fill = None
    else:
        col = key.replace("fill_", "")
        if col not in R.columns:
            report["levers"][name] = {"n": 0}
            continue
        s = R.dropna(subset=[col])
        avg = float(s[col].mean() * 10000)
        fill = float(R[key].mean() * 100) if key in R else None
    delta = avg - base_avg
    long_bp = float(R[R["side"] == 1][col if name != "NOW"
                                      else "pnl_now"].mean() * 10000)
    short_bp = float(R[R["side"] == -1][col if name != "NOW"
                                        else "pnl_now"].mean() * 10000)
    alive = bool(delta >= 2.0 and (fill or 0) >= 40)
    report["levers"][name] = {
        "n": int(len(s)), "avg_bp": round(avg, 2),
        "delta_vs_now": round(delta, 2),
        "tp_fill_share": None if fill is None else round(fill, 1),
        "long_bp": round(long_bp, 2), "short_bp": round(short_bp, 2),
        "candidate": alive}
    print(f"{name:6s}: {len(s)} сд, avg {avg:+.2f} бп (Δ{delta:+.2f}), "
          f"тейк-заполнений {fill}%, лонг {long_bp:+.1f} / шорт "
          f"{short_bp:+.1f} -> {'КАНДИДАТ' if alive else ''}", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ТЕЙК-ПО-ATR завершён", flush=True)
