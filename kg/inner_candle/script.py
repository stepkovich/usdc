"""INNER-CANDLE v3 (09.10): A/B 13 против 13+6 внутрь-признаков."""
import glob, json, time
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
N_TEST_DAYS = 3
FEE_TAKER = 0.0010
GATE = 0.62
PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1)
SYMS37 = ["1000BONKUSDC", "1000PEPEUSDC", "1000SHIBUSDC", "AAVEUSDC",
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
INNER = ["inner_below", "inner_up_max", "inner_dn_max", "inner_drift",
         "inner_jitter", "inner_n"]

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
book = book.dropna(subset=["mid", "spread_bp"])
book = book[book["symbol"].isin(SYMS37)]
book["bin30"] = (book["ts"] // 1_800_000) * 1_800_000
book["day"] = pd.to_datetime(book["ts"], unit="ms").dt.date.astype(str)
days = sorted(book["day"].unique())
test_days = days[-N_TEST_DAYS:]
print(f"книга: {len(book)} строк, тест {test_days} "
      f"| {time.time()-t0:.0f}с", flush=True)

train_med = {}
for c in ("flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
          "ntr10"):
    med = book[book["day"] < test_days[0]][c].median()
    train_med[c] = float(med) if med and med > 0 else 1.0

rows = []
for (sym, bin30), g in book.groupby(["symbol", "bin30"]):
    if sym not in SYMS37 or len(g) < 30:
        continue
    g = g.sort_values("ts")
    last = g.iloc[-1]
    open_px = g["mid"].iloc[0]
    rel = (g["mid"] / open_px - 1) * 1e4
    thirds = np.array_split(rel.values, 3)
    steps = np.diff(g["mid"].values)
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    d30 = d120 = np.nan
    for lag, nm in ((30_000, "d30"), (120_000, "d120")):
        j0 = np.searchsorted(ts, ts[0] - lag)
        if j0 < len(ts):
            past = mid[0] / mid[min(j0, len(mid) - 1)] - 1
            if nm == "d30":
                d30 = float(past)
            else:
                d120 = float(past)
    rows.append({"sym": sym, "ot_close": int(bin30 + 1_800_000),
                 "day": str(g["day"].iloc[0]),
                 "close": float(g["mid"].iloc[-1]),
                 "spread_bp": float(last["spread_bp"]),
                 "microprice_rel": float((last["microprice"] / last["mid"]
                                          - 1) * 10000),
                 "d30": float(d30) if d30 == d30 else 0.0,
                 "d120": float(d120) if d120 == d120 else 0.0,
                 "imb5": float(last["imb5"]), "imb10": float(last["imb10"]),
                 "imb20": float(last["imb20"]),
                 "flow10_buy": float(last["flow10_buy"]
                                     / train_med["flow10_buy"]),
                 "flow10_sell": float(last["flow10_sell"]
                                      / train_med["flow10_sell"]),
                 "flow60_buy": float(last["flow60_buy"]
                                     / train_med["flow60_buy"]),
                 "flow60_sell": float(last["flow60_sell"]
                                      / train_med["flow60_sell"]),
                 "ntr10": float(last["ntr10"] / train_med["ntr10"]),
                 "vpin10": float(last["vpin10"]),
                 "inner_below": float((rel < 0).mean()),
                 "inner_up_max": float(rel.max()),
                 "inner_dn_max": float(rel.min()),
                 "inner_drift": float(thirds[2].mean()
                                      - thirds[0].mean()),
                 "inner_jitter": float(np.std(steps)) if len(steps) > 1
                 else 0.0,
                 "inner_n": float(len(g))})

B30 = pd.DataFrame(rows)
print(f"30м-циклов {len(B30)}", flush=True)
del book

B30 = B30.sort_values(["sym", "ot_close"]).reset_index(drop=True)
B30["fwd30"] = B30.groupby("sym")["close"].shift(-1) / B30["close"] - 1
B30 = B30.dropna(subset=["fwd30"]).reset_index(drop=True)
days = sorted(B30["day"].unique())
test_days = days[-N_TEST_DAYS:]
print(f"тест-дни: {test_days}, строк {len(B30)}", flush=True)

train = B30[B30["day"] < test_days[0]]
test = B30[B30["day"].isin(test_days)]

FEATS_A = LOB_FEATS
FEATS_B = LOB_FEATS + INNER

mA = lgb.LGBMClassifier(**PARAMS)
mA.fit(train[FEATS_A], (train["fwd30"] > 0).astype("int"))
test["pA"] = mA.predict_proba(test[FEATS_A])[:, 1]
mB = lgb.LGBMClassifier(**PARAMS)
mB.fit(train[FEATS_B], (train["fwd30"] > 0).astype("int"))
test["pB"] = mB.predict_proba(test[FEATS_B])[:, 1]
print("обе модели обучены", flush=True)

GATE = 0.62
for name, pcol in (("A_13", "pA"), ("B_19", "pB")):
    pnls = []
    for sym, g in test.groupby("sym"):
        g = g.sort_values("ot_close").reset_index(drop=True)
        last_bin = -10**18
        for r in g.itertuples():
            if r.ot_close - last_bin < 20:
                continue
            sig = 0
            pv = getattr(r, pcol)
            if pv >= GATE:
                sig = 1
            elif pv <= 1 - GATE:
                sig = -1
            if sig == 0:
                continue
            pnl = sig * r.fwd30 - FEE_TAKER
            pnls.append(pnl)
            last_bin = r.ot_close
    a = np.array(pnls)
    if len(a):
        print(f"{name}: {len(a)} сделок, avg {a.mean()*1e4:+.2f} бп, "
              f"win {float((a>0).mean()*100):.1f}%", flush=True)
    else:
        print(f"{name}: 0 сделок", flush=True)

json.dump({"done": True}, open("/kaggle/working/report.json", "w"),
          indent=1)
print("INNER-CANDLE v3 завершена", flush=True)
