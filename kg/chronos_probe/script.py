"""CHRONOS (фундаментальная модель) vs ДЕРЕВО: 5-мин направление
37 USDC-пар (предрегистрация 07.10/08.10, идея владельца «готовые
модели из сети»).

CHRONOS-BOLT-SMALL (Amazon): сеть, обученная на миллионах временных
рядов; zero-shot на наших данных, без обучения — лечит «мало данных»
прошлых NN-проб. Веса обучены до наших тестовых дней — утечки нет.

ДАННЫЕ: наша запись, 37 пар, минутные миды, тест = последние 4 полных
дня. ВХОДЫ: каждые 300с на символ. КОНТЕКСТ Chronos: 256 минутных
мидов. ПРОГНОЗ: медиана шага 5 (=5 мин) + разброс квантилей.
СИГНАЛ CHRONOS: прогноз-медиана > текущий мид -> лонг, иначе шорт.
ПЛАНКА: дерево-реплика спринтера (train-дни, LOB_FEATS) на тех же
моментах.
МЕТРИКИ: acc + квинтильный навык (по уверенности: Chronos — ширина
квантилей, дерево — p5) на ОДНИХ И ТЕХ ЖЕ строках.
ВЕРДИКТ (записан до прогона): Chronos ИНТЕРЕСЕН, если acc >= 53% И
навык >= +2 бп. Иначе фундаментальные модели не дают направления на
5-минутках — направление закрыто и для них.
"""
import glob, json, time
import numpy as np, pandas as pd
import lightgbm as lgb
import torch

t0 = time.time()
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

# ---------- дерево-планка: обучение на train-днях ----------
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
print("дерево обучено на", len(train), flush=True)
del train

# ---------- тест-дни: p5 дерева + fwd5 + минутные бары с sym ----------
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
    sub = g[["ts", "mid"] + LOB_FEATS + ["fwd5"]].copy()
    sub["sym"] = sym
    te_frames.append(sub)
minmids = (book.groupby(["symbol", (book["ts"] // 60_000) * 60_000])
           ["mid"].last().reset_index())
minmids.columns = ["symbol", "ot", "mid"]
minmids["ot"] = minmids["ot"].astype("int64")
del book
T = pd.concat(te_frames, ignore_index=True)
del te_frames
T = T.dropna(subset=LOB_FEATS + ["fwd5"]).reset_index(drop=True)
T["p5"] = m5.predict_proba(T[LOB_FEATS])[:, 1]
T["minute"] = (T["ts"] // 60_000) * 60_000
print(f"тест-строк {len(T)} | {time.time()-t0:.0f}с", flush=True)

# ---------- Chronos-Bolt ----------
import subprocess
r = subprocess.run(["pip", "install", "-q", "chronos-forecasting"],
                   capture_output=True, text=True)
if r.returncode != 0:
    print("pip chronos:", r.stderr[-200:], flush=True)
from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small", device_map="cuda",
    torch_dtype=torch.float32)
CTX = 256
PRED = 5                                     # шаги по 1м -> 5 мин

mm = minmids[minmids["symbol"].isin(SYMS)]
print(f"минутных баров {len(mm)} | {time.time()-t0:.0f}с", flush=True)

forecasts = {}                               # (sym, minute) -> (med, spread)
BATCH, META = [], []
def flush_batch():
    global BATCH, META
    if not BATCH:
        return
    with torch.no_grad():
        q = pipe.predict_quantiles(
            torch.tensor(np.array(BATCH, dtype="float32"), device="cuda"),
            prediction_length=PRED)
    qs = q[0].cpu().numpy() if isinstance(q, tuple) else \
        q.numpy() if hasattr(q, "numpy") else np.asarray(q)
    med = qs[:, PRED - 1, 4] if qs.ndim == 3 else qs[:, PRED - 1]
    lo = qs[:, PRED - 1, 0] if qs.ndim == 3 else qs[:, PRED - 1]
    hi = qs[:, PRED - 1, -1] if qs.ndim == 3 else qs[:, PRED - 1]
    for (s2, o2), mv, lv, hv in zip(META, med, lo, hi):
        forecasts[(s2, o2)] = (float(mv), float(hv - lv))
    BATCH, META = [], []

for sym, g in mm.groupby("symbol"):
    g = g.sort_values("ot").reset_index(drop=True)
    vals = g["mid"].values.astype("float32")
    ots = g["ot"].values
    for end in range(CTX, len(vals), 3):     # вход каждые 3 минуты
        ctx = vals[end - CTX:end]
        if np.isnan(ctx).any():
            continue
        BATCH.append(ctx)
        META.append((sym, int(ots[end - 1])))
        if len(BATCH) >= 1024:
            flush_batch()
flush_batch()
print(f"прогнозов Chronos {len(forecasts)} | {time.time()-t0:.0f}с",
      flush=True)

# ---------- склейка и метрики ----------
fmap_med = {k: v[0] for k, v in forecasts.items()}
fmap_spread = {k: v[1] for k, v in forecasts.items()}
T["chrono_med"] = [fmap_med.get((s2, int(mn)), np.nan)
                   for s2, mn in zip(T["sym"].values, T["minute"].values)]
T["chrono_spread"] = [fmap_spread.get((s2, int(mn)), np.nan)
                      for s2, mn in zip(T["sym"].values,
                                        T["minute"].values)]
D = T.dropna(subset=["chrono_med", "fwd5"]).reset_index(drop=True)
print(f"строк с прогнозом Chronos: {len(D)} | {time.time()-t0:.0f}с",
      flush=True)
if len(D) < 2000:
    report = {"error": f"мало строк с прогнозом: {len(D)}"}
    json.dump(report, open("/kaggle/working/report.json", "w"),
              indent=1)
    raise SystemExit(0)

D["chrono_dir"] = (D["chrono_med"] > D["mid"]).astype("float32")
D["chrono_conf"] = ((D["chrono_spread"] / D["mid"]) * 1e4).astype(
    "float32")
D["tree_dir"] = (D["p5"] > 0.5).astype("float32")

def quint_skill(d, conf_col):
    p = d[conf_col].values
    f = d["fwd5"].values.astype("float64")
    q20, q80 = np.quantile(p, [0.2, 0.8])
    return float((f[p >= q80].mean() - f[p <= q20].mean()) * 10000)

res = {}
for nm, dcol, ccol in (("CHRONOS", "chrono_dir", "chrono_conf"),
                       ("ДЕРЕВО", "tree_dir", "p5")):
    acc = float((D[dcol] == (D["fwd5"] > 0).astype("float32")).mean()
                * 100)
    sk = quint_skill(D, ccol)
    # квинтиль навыка по уверенности (верх vs низ уверенности)
    q20, q80 = np.quantile(D[ccol].values, [0.2, 0.8])
    conf_sk = float((D["fwd5"].values[D[ccol].values >= q80].mean()
                     - D["fwd5"].values[D[ccol].values <= q20].mean())
                    * 10000)
    res[nm] = {"acc": round(acc, 2), "quint_skill_bp": round(sk, 2),
               "conf_skill_bp": round(conf_sk, 2)}
    print(f"{nm}: acc {acc:.2f}%, квинтиль навык {sk:+.2f} бп, "
          f"по уверенности {conf_sk:+.2f} бп | {time.time()-t0:.0f}с",
          flush=True)

chrono = res["CHRONOS"]
verdict = bool(chrono["acc"] >= 53.0 and chrono["quint_skill_bp"] >= 2.0)
report = {"pre_reg": "Chronos интересен: acc>=53% и навык>=+2 бп",
          "results": res,
          "n": int(len(D)),
          "days": f"{test_days[0]}..{test_days[-1]}",
          "verdict": ("CHRONOS ИНТЕРЕСЕН" if verdict else
                      "фундаментальная модель направления не даёт")}
print("ВЕРДИКТ:", report["verdict"], flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("CHRONOS ПРОБА завершена", flush=True)
