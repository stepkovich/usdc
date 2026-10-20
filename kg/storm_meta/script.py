"""МЕТА-ПОДТВЕРЖДЕНИЕ БУРИ (предрегистрировано 12.10, идея владельца
«модель с обучением подтверждением» = мета-разметка: второе дерево
учится на ИТОГАХ сигналов первого и подтверждает/ветирует новые).

МЕХАНИКА (walk-forward без подглядывания):
- фолды те же 8 кварталов; первичная модель-буря обучается на всём ДО
  фолда, сигналы фолда — её OOS-сигналы (gate 0.65, обе стороны,
  cooldown 10 баров, как в бою);
- С ФОЛДА 2: мета-модель (LightGBM, то же дерево) обучается на ВСЕХ
  OOS-сигналах ПРОШЛЫХ фолдов (признаки сигнала + уверенность p +
  atr_rel; метка = выиграл сигнал после 10бп издержек или нет) и
  предсказывает вероятность победы для сигналов текущего фолда;
  сигнал ПРИНЯТ при p_win >= 0.5, иначе вето;
- сигналы фолда после исхода добавляются в мета-обучение следующих.

ВЕРДИКТ (до прогона): КАНДИДАТ если на пуле фолдов 2..8:
avg(принятые) − avg(все сигналы) >= +3 бп, n(принятых) >= 1000,
avg(принятых) > 0. Отчёт по каждому фолду + доля принятых + винрейт.
Фолд 1 (без мета-истории) — контроль, в вердикт не идёт.

ЧЕСТНЫЕ ОГОВОРКИ: мета обучается на исторических OOS-сигналах (модель
первого уровня каждый раз ДО фолда — утечки нет); p сигнала и исход
механически связаны через волатильность монеты — часть работы мета
может дублировать «тихую треть» (это честно видеть цифрами); 8 фолдов
= мало перетреновочных просторов у мета-дерева (нарочно маленькое).
Ничего не деплоится; кандидат идёт на свежие дни.
"""
import glob, io, json, time, zipfile, zlib
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000
FEE = 0.0010
META_PARAMS = dict(n_estimators=120, learning_rate=0.05, max_depth=3,
                   num_leaves=7, subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=4, verbosity=-1)
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
                      np.where(fwd > thr, 1.0,
                               np.where(fwd < -thr, 0.0, np.nan)))
    X["fwd"] = fwd.astype("float32")
    X["atr_rel"] = atr_rel.astype("float32")
    X["close"] = close
    X["ot"] = df["open_time"].values.astype(np.int64)
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
FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
MFEATS = FEATS + ["p_sig", "atr_rel"]
print(f"строк {len(data)}, фолды {test_q} | {time.time()-t0:.0f}с",
      flush=True)

def collect(d):
    """OOS-сигналы фолда с полным вектором признаков для мета."""
    pools = {int(fi): gg.sort_values("ot") for fi, gg in d.groupby("sym_id")}
    sigs = []
    for side in (1, -1):
        sel = d[(d["p"] > GATE) if side == 1 else (d["p"] < 1 - GATE)] \
            .sort_values(["sym_id", "ot"])
        last = {}
        for row in sel.itertuples():
            if row.ot - last.get(row.sym_id, -10**18) < COOLDOWN_MS:
                continue
            last[row.sym_id] = row.ot
            pool = pools.get(int(row.sym_id))
            if pool is None:
                continue
            ots = pool["ot"].values
            j = int(np.searchsorted(ots, row.ot))
            if j >= len(ots) or ots[j] != row.ot or j + H >= len(ots):
                continue
            base = side * float(row.fwd) - FEE
            vec = [float(getattr(row, f)) for f in FEATS]
            vec.append(float(row.p))
            vec.append(float(row.atr_rel))
            sigs.append({"vec": vec, "base": base, "side": side})
    return sigs

hist = []                       # OOS-сигналы прошлых фолдов для мета
report = {"pre_reg": "КАНДИДАТ: пул фолдов 2..8: avg(принятые)−avg(все) "
                     ">= +3бп, n>=1000, avg(принятых)>0",
          "folds": {}}
acc_all, all_all, n_acc_pool, acc_pool = [], [], 0, 0.0
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 20000 or len(trd) < 500_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    trd_ok = trd.dropna(subset=["y"])
    m.fit(trd_ok[FEATS], trd_ok["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    sigs = collect(d)
    allbp = [s["base"] for s in sigs]
    if k == 0 or len(hist) < 2000:
        acc = list(range(len(sigs)))       # фолд 1: без истории, контроль
        meta_note = "контроль (без мета)"
    else:
        hm = pd.DataFrame([h["vec"] for h in hist], columns=MFEATS)
        hy = np.array([1.0 if h["base"] > 0 else 0.0 for h in hist])
        meta = lgb.LGBMClassifier(**META_PARAMS)
        meta.fit(hm, hy)
        Xf = pd.DataFrame([s["vec"] for s in sigs], columns=MFEATS)
        pw = meta.predict_proba(Xf)[:, 1]
        acc = [i for i in range(len(sigs)) if pw[i] >= 0.5]
        meta_note = (f"мета на {len(hist)} сигн., принята доля "
                     f"{len(acc)/max(1,len(sigs))*100:.0f}%")
    accbp = [sigs[i]["base"] for i in acc]
    f_avg_all = float(np.mean(allbp)) * 1e4 if allbp else float("nan")
    f_avg_acc = float(np.mean(accbp)) * 1e4 if accbp else float("nan")
    wr = float(np.mean([1.0 if b > 0 else 0.0 for b in accbp]) * 100) \
        if accbp else float("nan")
    report["folds"][str(qq)] = {
        "n_all": len(sigs), "n_acc": len(acc),
        "avg_all_bp": round(f_avg_all, 2),
        "avg_acc_bp": round(f_avg_acc, 2), "wr_acc": round(wr, 1),
        "note": meta_note}
    print(f"фолд {qq}: все {len(sigs)} ({f_avg_all:+.1f}бп) | "
          f"принято {len(acc)} ({f_avg_acc:+.1f}бп, WR {wr:.0f}%) | "
          f"{meta_note} | {time.time()-t0:.0f}с", flush=True)
    if k >= 1:
        acc_all.extend(accbp)
        all_all.extend(allbp)
        n_acc_pool += len(acc)
        acc_pool += sum(accbp)
    hist.extend(sigs)

if acc_all:
    a_acc = float(np.mean(acc_all)) * 1e4
    a_all = float(np.mean(all_all)) * 1e4
    cand = bool(n_acc_pool >= 1000 and a_acc - a_all >= 3.0 and a_acc > 0)
    report["verdict"] = {"pool_folds": "2..8", "n_acc": n_acc_pool,
                         "avg_acc_bp": round(a_acc, 2),
                         "avg_all_bp": round(a_all, 2),
                         "delta_bp": round(a_acc - a_all, 2),
                         "candidate": cand}
    print(f"ИТОГ (фолды 2..8): принятые {a_acc:+.2f}бп против всех "
          f"{a_all:+.2f}бп (Δ{a_acc-a_all:+.2f}), n={n_acc_pool} -> "
          f"{'КАНДИДАТ' if cand else 'НЕТ'}", flush=True)
# МЕТА-МОДЕЛЬ ВСЕГДА СОХРАНЯЕТСЯ: последняя обученная (на всей OOS-истории
# до последнего фолда) — это и есть свежая мета для деплоя. Вердикт
# «кандидат/нет» оценивает ПОЛЬЗУ замка, а не необходимость обновления.
try:
    meta.booster_.save_model("/kaggle/working/meta.txt")
    print(f"meta.txt сохранён: обучена на {len(hist)} сигналах истории",
          flush=True)
except NameError:
    print("мета не обучалась ни на одном фолде — сохранять нечего",
          flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("МЕТА-ПОДТВЕРЖДЕНИЕ ЗАВЕРШЕНО", flush=True)
