"""CHRONOS НА СВЕЧАХ: ВОПРОС БУРИ (предрегистрация 09.10, идея владельца).

ПРОШЛАЯ ПРОБА: Chronos на 5-мин направлении книги — acc 55% но навык 0
(уровень цены не ранжирует). ИДЕЯ ВЛАДЕЛЬЦА: скормить свечи (мин/макс/
объём), может покажет себя лучше. ТЕХ-ФАКТ: Chronos ест ОДИН ряд
чисел; OHLCV напрямую нельзя. ЧТО ДЕЛАЕМ:
  (а) ряд = лог-доходности close (станд. режим Chronos — «сезонность
      нет, тренд нет» => разности);
  (б) виртуальный ряд «цена×признак»: цена, умноженная на
      нормированный объём (пиковый объём = выброс вверх) — канал
      объёма внутрь одного ряда.
ВОПРОС = как у Бури: |fwd за 10 баров 30м| > 0.5×ATR (буря есть/нет),
тест = 8 кварталов linspace (те же, что у боевой Бури), входы — каждые
12ч на символ. ПРОГНОЗ: |медиана шага 10 / текущий уровень − 1|.
СИГНАЛ: прогноз > 0.5×ATR → «буря предсказана».
ПЛАНКА: боевая модель Бури (LightGBM) на тех же строках.
МЕТРИКИ: acc шторм-детекции, полнота/точность, на пойманных барах —
квинтильный навык fwd по прогнозной величине (бп).
ВЕРДИКТ (до прогона): Chronos ИНТЕРЕСЕН, если acc >= acc(дерево)+2пп
ИЛИ навык на пойманных >= +3 бп. Ограничение честности: 30м-бары
30-минутные, контекст 256 баров = 5.3 суток.
"""
import glob, io, json, time, zipfile, zlib
import numpy as np, pandas as pd
import lightgbm as lgb
import torch

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000
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

FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]

# ---------- сбор панель + ярлык бури ----------
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
                      (np.abs(fwd) > thr).astype("float32"))
    X["fwd"] = fwd.astype("float32")
    X["atr_rel"] = atr_rel.astype("float32")
    X["ot"] = df["open_time"].values.astype(np.int64)
    X["sym"] = sym
    # ряды для Chronos: лог-рет и цена×объём-фактор
    lr = np.log(close.astype("float64"))
    X["lr"] = lr
    vf = df["volume"].values / (pd.Series(df["volume"].values)
                                .rolling(96, min_periods=20).mean()
                                .values + 1e-9)
    X["px_vol"] = (close * np.clip(vf, 0.5, 3.0)).astype("float32")
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
print(f"строк {len(data)}, фолды {test_q} | {time.time()-t0:.0f}с",
      flush=True)

# ---------- Chronos ----------
import subprocess
subprocess.run(["pip", "install", "-q", "chronos-forecasting"],
               capture_output=True)
from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small", device_map="cuda",
    torch_dtype=torch.float32)
CTX = 256
PRED = H                                       # 10 шагов 30м = 5ч

def chronos_batch(series_list, metas, out):
    with torch.no_grad():
        qnt = pipe.predict_quantiles(
            torch.tensor(np.array(series_list, dtype="float32"),
                         device="cuda"),
            prediction_length=PRED)
    qs = qnt[0].cpu().numpy() if isinstance(qnt, tuple) else \
        np.asarray(qnt)
    med = qs[:, PRED - 1, 4] if qs.ndim == 3 else qs[:, PRED - 1]
    lo = qs[:, PRED - 1, 0] if qs.ndim == 3 else qs[:, PRED - 1]
    hi = qs[:, PRED - 1, -1] if qs.ndim == 3 else qs[:, PRED - 1]
    for (sym, ot, lvl, atr_r), mv, lv, hv in zip(metas, med, lo, hi):
        out[(sym, ot)] = {"lvl": float(lvl), "med": float(mv),
                          "spread": float(hv - lv),
                          "atr": float(atr_r)}

results = {}
for variant, col in (("RET", "lr"), ("PXVOL", "px_vol")):
    fc = {}
    BATCH, META = [], []
    for sym, g in data[data["quarter"].isin(test_q)] \
            .groupby("sym"):
        g = g.sort_values("ot").reset_index(drop=True)
        # режим разностей для RET-варианта
        raw = g[col].values.astype("float64")
        for end in range(CTX, len(g), 12):    # каждые 6 часов
            ctx = raw[end - CTX:end]
            if np.isnan(ctx).any():
                continue
            if variant == "RET":
                d = np.diff(ctx)
                if np.isnan(d).any() or np.std(d) < 1e-12:
                    continue
                BATCH.append((ctx[-1] - ctx[0]) * 0 + d)
                lvl = ctx[-1]
            else:
                BATCH.append(ctx)
                lvl = ctx[-1]
            META.append((sym, int(g["ot"].values[end - 1]), lvl,
                         float(g["atr_rel"].values[end - 1])))
            if len(BATCH) >= 1024:
                chronos_batch(BATCH, META, fc)
                BATCH, META = [], []
    chronos_batch(BATCH, META, fc)
    print(f"{variant}: прогнозов {len(fc)} | {time.time()-t0:.0f}с",
          flush=True)
    results[variant] = fc

# ---------- планка: боевая Буря на тех же строках ----------
storm_pred_tree = {}
acc_rows_ch = {"RET": [], "PXVOL": []}
for qq in test_q:
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 5000 or len(trd) < 200_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    trd_ok = trd.dropna(subset=["y"])
    m.fit(trd_ok[FEATS], trd_ok["y"], categorical_feature=["sym_id"])
    te_ok = ted.dropna(subset=["y"])
    p_tree = m.predict_proba(te_ok[FEATS])[:, 1]
    for sym, ot, pt in zip(te_ok["sym"].values, te_ok["ot"].values,
                           p_tree):
        storm_pred_tree[(sym, int(ot))] = float(pt)
    print(f"фолд {qq}: дерево готово | {time.time()-t0:.0f}с", flush=True)

# ---------- метрики ----------
def storm_metrics(fc, name):
    rows = []
    for sym, g in data.groupby("sym"):
        g = g.sort_values("ot")
        last = {}
        for r in g.itertuples():
            key = (sym, int(r.ot))
            if key not in fc:
                continue
            if r.ot - last.get(sym, -10**18) < COOLDOWN_MS:
                continue
            last[sym] = r.ot
            f = fc[key]
            if name == "RET":
                # ряд был разностями: медиана = суммарный ход от lvl
                pred_move = f["med"] / f["lvl"] if f["lvl"] != 0 else 0
            else:
                pred_move = f["med"] / f["lvl"] - 1
            rows.append({"pred_storm": abs(pred_move) / max(f["atr"],
                                                           1e-9),
                         "spread": f["spread"] / max(f["lvl"], 1e-9),
                         "actual": 1.0 if r.y == 1 else 0.0,
                         "fwd": float(r.fwd)})
    D = pd.DataFrame(rows)
    if not len(D):
        return None
    D["pred_storm_flag"] = (D["pred_storm"] > 0.5).astype(int)
    acc = float((D["pred_storm_flag"] == D["actual"]).mean() * 100)
    caught = D[D["pred_storm_flag"] == 1]
    skill = None
    if len(caught) > 200:
        qs = np.quantile(caught["pred_storm"].values, [0.2, 0.8])
        skill = float((caught["fwd"][caught["pred_storm"] >= qs[1]]
                       .abs().mean()
                       - caught["fwd"][caught["pred_storm"] <= qs[0]]
                       .abs().mean()) * 10000)
    return {"n": int(len(D)), "acc": round(acc, 2),
            "caught": int(len(caught)),
            "actual_rate": round(float(D["actual"].mean() * 100), 1),
            "skill_bp": None if skill is None else round(skill, 2)}

report = {"pre_reg": "Chronos интересен: acc >= дерево+2пп ИЛИ навык "
                     ">= +3 бп", "variants": {}}
tree_rows = []
for sym, g in data.groupby("sym"):
    g = g.sort_values("ot")
    last = {}
    for r in g.itertuples():
        key = (sym, int(r.ot))
        if key not in storm_pred_tree:
            continue
        if r.ot - last.get(sym, -10**18) < COOLDOWN_MS:
            continue
        last[sym] = r.ot
        tree_rows.append({"pred": storm_pred_tree[key],
                          "actual": 1.0 if r.y == 1 else 0.0,
                          "fwd": float(r.fwd)})
TR = pd.DataFrame(tree_rows)
TR["flag"] = (TR["pred"] > 0.5).astype(int)
acc_tree = float((TR["flag"] == TR["actual"]).mean() * 100)
report["tree_acc"] = round(acc_tree, 2)
print(f"ДЕРЕВО (планка): acc {acc_tree:.2f}% на {len(TR)} строках",
      flush=True)

for variant in ("RET", "PXVOL"):
    st = storm_metrics(results[variant], variant)
    report["variants"][variant] = st
    if st:
        print(f"CHRONOS-{variant}: acc {st['acc']}%, поймано "
              f"{st['caught']}/{st['n']}, навык {st['skill_bp']} бп",
              flush=True)

interesting = False
for variant, st in report["variants"].items():
    if st and (st["acc"] >= acc_tree + 2
               or (st["skill_bp"] or -99) >= 3.0):
        interesting = True
report["verdict"] = ("CHRONOS ИНТЕРЕСЕН" if interesting else
                     "на вопросе бури тоже не даёт")
print("ВЕРДИКТ:", report["verdict"], flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("CHRONOS-БУРЯ завершена", flush=True)
