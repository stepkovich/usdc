"""РЫЧАГИ ВЫХОДА СПРИНТЕРА (предрегистрация 07.10, офлайн на записях).

ВОПРОС: сейчас выход = будильник 300с: лимитка на дальнем краю с TTL
45с (мейкер 28% случаев), иначе рынок (68%) → средние издержки ~7бп.
Какие рычаги поднимают мейкер-долю и что дают в деньгах?

СЦЕНАРИИ (на ОДНИХ И ТЕХ ЖЕ сделках спринтера, отличается только выход):
  BASE    лимитка на дальнем краю, TTL 45с, иначе рынок — как в бою;
  MID     лимитка на середине спреда, TTL 45с, иначе рынок;
  MID120  середина спреда, TTL 120с, иначе рынок;
  TP2/TP4/TP6  тейк-профит лимиткой сразу после входа (+2/+4/+6 бп,
          весь холд 300с на заполнение), иначе рынок на 300с.
Издержки (USDC): вход мейкер 0; выход мейкер 0; выход рынок 10бп
(5 fee + 5 slip). Вход по миду, выход-прокси по миду с шагом 1с.
Валидация базы: мейкер-доля BASE должна выйти ~28% (как в живом журнале).
ВЕРДИКТ (записан до прогона): рычаг — кандидат, если он добавляет
>= +2 бп к BASE и мейкер-доля >= 60%. Ничего не деплоится.
"""
import glob, json, time
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
N_TEST_DAYS = 4
SP_GATE = 0.62
HOLD_MS = 300_000
BASE_TTL = 45_000
MID_TTL = 120_000
FEE_TAKER_OUT = 0.0010          # 5бп fee + 5бп slip
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
book["day"] = pd.to_datetime(book["ts"], unit="ms").dt.date.astype(str)
days = sorted(book["day"].unique())
test_days = days[-N_TEST_DAYS:]
print(f"книжных строк {len(book)}, дней {len(days)} "
      f"({days[0]}..{days[-1]}), тест {test_days} | {time.time()-t0:.0f}с",
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

def exit_price(mids_ts, mids_mid, e, side, fill_px, t0_ms, ttl_ms, fee):
    """Касание fill_px в [t0, t0+ttl] -> мейкер по fill_px; иначе рынок
    по последнему миду окна с полным тейкер-выходом."""
    m = (mids_ts >= t0_ms) & (mids_ts <= t0_ms + ttl_ms)
    seg = mids_mid[m]
    if len(seg) == 0:
        return None, None, None, None
    if side == 1:
        hit = seg >= fill_px
        exit_px = fill_px if hit.any() else seg[-1] * (1 - 0.0005)
    else:
        hit = seg <= fill_px
        exit_px = fill_px if hit.any() else seg[-1] * (1 + 0.0005)
    fee_used = 0.0 if hit.any() else fee
    return exit_px, bool(hit.any()), fee_used, t0_ms + ttl_ms

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
        p5[valid.index.values] = m5.predict_proba(
            valid[LOB_FEATS])[:, 1]
    d = pd.DataFrame({"ts": ts, "mid": mid, "spread": spread,
                      "p5": p5, "fwd5": fwd5,
                      "day": pd.to_datetime(ts, unit="ms").date}).dropna()
    ts = d["ts"].values
    mid = d["mid"].values
    spread = d["spread"].values
    p5 = d["p5"].values
    fwd5 = d["fwd5"].values
    n = len(d)
    i = 0
    while i < n:
        sig = 0
        if p5[i] >= SP_GATE:
            sig = 1
        elif p5[i] <= 1 - SP_GATE:
            sig = -1
        if sig == 0:
            i += 1
            continue
        e = mid[i]
        t_exit = ts[i] + HOLD_MS
        j_exit = int(np.searchsorted(ts, t_exit))
        if j_exit >= n:
            break
        half = spread[j_exit] / 2.0
        far_px = mid[j_exit] * ((1 + half) if sig == 1 else (1 - half))
        mid_px = mid[j_exit]
        m_ts = ts[j_exit: j_exit + 400]
        m_mid = mid[j_exit: j_exit + 400]
        ex_far, hit_far, fee_far, _ = exit_price(
            m_ts, m_mid, e, sig, far_px, t_exit, BASE_TTL,
            FEE_TAKER_OUT)
        ex_mid, hit_mid, fee_mid, _ = exit_price(
            m_ts, m_mid, e, sig, mid_px, t_exit, BASE_TTL,
            FEE_TAKER_OUT)
        ex_m120, hit_m120, fee_m120, _ = exit_price(
            m_ts, m_mid, e, sig, mid_px, t_exit, MID_TTL,
            FEE_TAKER_OUT)
        if ex_far is None or ex_mid is None or ex_m120 is None:
            i = j_exit + 300
            continue                          # стык дней — окна пусты
        pnl_base = sig * (ex_far / e - 1) - fee_far
        pnl_mid = sig * (ex_mid / e - 1) - fee_mid
        pnl_mid120 = sig * (ex_m120 / e - 1) - fee_m120
        tp_res = {}
        for tp in (0.0002, 0.0004, 0.0006):
            tp_px = e * ((1 + tp) if sig == 1 else (1 - tp))
            ex_tp, hit_tp, fee_tp, _ = exit_price(
                ts[i: j_exit + 400], mid[i: j_exit + 400], e, sig,
                tp_px, ts[i], HOLD_MS, FEE_TAKER_OUT)
            if ex_tp is None:
                continue
            tp_res[f"TP{int(tp*10000)}"] = sig * (ex_tp / e - 1) - fee_tp
        rec = {"sym": sym, "day": d["day"].iloc[i], "side": sig,
               "pnl_base": pnl_base, "pnl_mid": pnl_mid,
               "pnl_mid120": pnl_mid120, "hit_far": hit_far,
               "hit_mid": hit_mid, "hit_m120": hit_m120,
               **tp_res}
        ex_mkt = e * (1 + sig * fwd5[i])
        if "TP4" in tp_res:
            rec["pnl_split"] = 0.5 * (sig * (ex_mkt / e - 1)
                                      - FEE_TAKER_OUT) \
                + 0.5 * tp_res["TP4"]
        rows_out.append(rec)
        i = j_exit + 300

R = pd.DataFrame(rows_out)
print(f"сделок в симуляции: {len(R)} | {time.time()-t0:.0f}с", flush=True)

report = {"pre_reg": "кандидат: добавка >= +2бп к BASE и мейкер-доля>=60%",
          "levers": {}}
base_avg = float(R["pnl_base"].mean() * 10000)
for col, name in (("pnl_base", "BASE"), ("pnl_mid", "MID"),
                  ("pnl_mid120", "MID120"), ("TP2", "TP2"),
                  ("TP4", "TP4"), ("TP6", "TP6"), ("pnl_split", "SPLIT")):
    s = R.dropna(subset=[col])
    avg = float(s[col].mean() * 10000)
    mk_share = None
    if col == "pnl_base":
        mk_share = float(R["hit_far"].mean() * 100)
    elif col == "pnl_mid":
        mk_share = float(R["hit_mid"].mean() * 100)
    elif col == "pnl_mid120":
        mk_share = float(R["hit_m120"].mean() * 100)
    elif col.startswith("TP"):
        mk_share = 100.0
    delta = avg - base_avg
    by_day = {str(k): round(v * 10000, 1) for k, v in
              s.groupby("day")[col].mean().items()}
    long_bp = float(R[R["side"] == 1][col].mean() * 10000)
    short_bp = float(R[R["side"] == -1][col].mean() * 10000)
    report["levers"][name] = {
        "n": int(len(s)), "avg_bp": round(avg, 2),
        "delta_vs_base": round(delta, 2),
        "maker_share": None if mk_share is None else round(mk_share, 1),
        "long_bp": round(long_bp, 2), "short_bp": round(short_bp, 2),
        "by_day": by_day}
    print(f"{name:7s}: {len(s)} сд, avg {avg:+.2f} бп (Δ{delta:+.2f}), "
          f"мейкер {mk_share}%, лонг {long_bp:+.1f} / шорт {short_bp:+.1f}",
          flush=True)

best_name, best_v = max(
    ((k, v) for k, v in report["levers"].items() if k != "BASE"),
    key=lambda kv: kv[1]["avg_bp"])
report["best"] = best_name
report["candidate"] = bool(best_v["delta_vs_base"] >= 2.0
                           and (best_v["maker_share"] or 0) >= 60.0)
print("ЛУЧШИЙ:", best_name, "->",
      "КАНДИДАТ" if report["candidate"] else "нет", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("РЫЧАГИ ВЫХОДА завершены", flush=True)
