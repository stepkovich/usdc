"""МЕТА-ПОДТВЕРЖДЕНИЕ БУРИ (v10, 16.11): месячная схема (владелец).
Тетрадь сигналов строится ПОЛНЫМ месячным проходом по всей истории
(каждый месяц отвечает модель, обученная на всём до него), окно 5ч
НЕПРЕРЫВНО через границы месяцев. Для каждого сигнала тетрадь хранит
ПОЛНЫЙ набор исходов: движение 10 баров (цена->цена), биржевой стоп
-15% (дошёл/не дошёл, выход), максимум в плюс/минус (насколько
перешагнул), ФАНДИНГ за период удержания (ставки с зеркала, сторона),
флаг исполнения лимитки. Тетрадь целиком сохраняется в notebook.csv —
варианты обучения меты (метка «перешагнул без убытка», «насколько»,
«чистыми после фандинга») считаются офлайн без новых проходов.
Деплой-мета обучается как раньше (метка: выиграл 10 баров после
издержек) для сопоставимости с v9. Вердикт «кандидат/нет» — справка.
"""
import glob, io, json, time, zipfile, zlib
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
H = 10
STOP = 0.15
GATE = 0.65
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
    X["low"] = df["low"].values
    X["high"] = df["high"].values
    X["ot"] = df["open_time"].values.astype(np.int64)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=["atr_pct"]))
    if i % 100 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
# ФАНДИНГ: monthly-файлы с зеркала сложены панелью в funding/{sym}.csv
FUND = {}
for name in zf.namelist():
    if name.startswith("funding/") and name.endswith(".csv"):
        sym = name.split("/")[-1].replace("_funding.csv", "")
        pts = []
        for r in zf.read(name).decode().strip().split("\n")[1:]:
            c = r.split(",")
            try:
                ct = int(float(c[0]))
                if ct > 10 ** 14:
                    ct //= 1000
                pts.append((ct, float(c[1])))
            except Exception:
                pass
        pts.sort()
        FUND[zlib.crc32(sym.encode()) & 0xFFFF] = pts
print(f"фандинг: {len(FUND)} монет | {time.time()-t0:.0f}с", flush=True)
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="M")
data["month"] = q.astype(str)
months = sorted(data["month"].unique())
test_q = months[:]          # ПОЛНЫЙ проход: каждый месяц истории
# (схема владельца 16.11, как в репетиции usdc-monthly-replay: тетрадь
# меты наполняется сигналами ВСЕХ месяцев, каждый от своей модели)
test_q = [m for m in test_q if m >= months[3]]   # ранние месяцы отсеет
# guard len(trd)<500_000 внутри цикла, здесь не режем данные
FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
MFEATS = FEATS + ["p_sig", "atr_rel"]
print(f"строк {len(data)}, фолды {test_q} | {time.time()-t0:.0f}с",
      flush=True)

def collect(d, gate):
    """OOS-сигналы месяца с ПОЛНЫМ набором полей для тетради.
    ПОДТВЕРЖДЕНИЕ (синхронно с боем 23.11, правило владельца): сигнал
    учитывается только если у той же монеты на ПРЕДЫДУЩЕМ баре порог
    тоже был пройден в ту же сторону (два бара подряд)."""
    pools = {int(fi): gg.sort_values("ot") for fi, gg in d.groupby("sym_id")}
    ps_ = d["p"].values
    sy_ = d["sym_id"].values
    ot_ = d["ot"].values
    gate_ok = (ps_ > gate) | (ps_ < 1 - gate)
    sides_ = np.where(ps_ > gate, 1, -1)
    prev_side = {(int(sy), int(t)): int(sd)
                 for sy, t, sd in zip(sy_[gate_ok], ot_[gate_ok],
                                      sides_[gate_ok])}
    sigs = []
    for side in (1, -1):
        sel = d[(d["p"] > gate) if side == 1 else (d["p"] < 1 - gate)] \
            .sort_values(["sym_id", "ot"])
        for row in sel.itertuples():
            if prev_side.get((int(row.sym_id),
                              int(row.ot) - 1800_000)) != side:
                continue
            pool = pools.get(int(row.sym_id))
            if pool is None:
                continue
            ots = pool["ot"].values
            j = int(np.searchsorted(ots, row.ot))
            if j >= len(ots) or ots[j] != row.ot or j + H >= len(ots):
                continue
            c = float(row.close)
            seg_lo = pool["low"].values[j + 1: j + 1 + H]
            seg_hi = pool["high"].values[j + 1: j + 1 + H]
            filled = (seg_lo[0] <= c) if side == 1 else (seg_hi[0] >= c)
            stop = c * (1 - STOP) if side == 1 else c * (1 + STOP)
            hit = False
            kex = H
            for k in range(H):
                if side == 1 and seg_lo[k] <= stop:
                    hit, kex = True, k + 1
                    break
                if side == -1 and seg_hi[k] >= stop:
                    hit, kex = True, k + 1
                    break
            exit_px = stop if hit else float(pool["close"].values[j + H])
            exit_ret = side * (exit_px / c - 1) * 1e4
            hi_e = side * (seg_hi.max() / c - 1) * 1e4
            lo_e = side * (seg_lo.min() / c - 1) * 1e4
            mfe = max(hi_e, lo_e)
            mae = min(hi_e, lo_e)
            # фандинг за период удержания: события (ot[j], конец удержания]
            hold_end = int(pool["ot"].values[j + kex]) + 1_800_000
            paid = 0.0
            import bisect as _bs
            fpts = FUND.get(int(row.sym_id), [])
            if fpts:
                lo_i = _bs.bisect_right(fpts, (row.ot, 10))
                hi_i = _bs.bisect_right(fpts, (hold_end, 10))
                for _t, _r in fpts[lo_i:hi_i]:
                    paid += side * _r
            paid *= 1e4
            sigs.append({
                "month": str(row.month), "sym_id": int(row.sym_id),
                "side": side, "p": float(row.p),
                "atr_rel": float(row.atr_rel),
                "vec": [float(getattr(row, f)) for f in FEATS] +
                       [float(row.p), float(row.atr_rel)],
                "fwd_bp": side * float(row.fwd) * 1e4,
                "exit_ret_bp": exit_ret, "stop_hit": hit,
                "mfe_bp": float(mfe), "mae_bp": float(mae),
                "filled": bool(filled), "hold_end_ms": hold_end,
                "paid_bp": float(paid),
                "base": side * float(row.fwd) - FEE})
    return sigs


hist = []                       # OOS-сигналы прошлых фолдов для мета
report = {"pre_reg": "КАНДИДАТ: пул фолдов 2..8: avg(принятые)−avg(все) "
                     ">= +3бп, n>=1000, avg(принятых)>0",
          "folds": {}}
acc_all, all_all, n_acc_pool, acc_pool = [], [], 0, 0.0
for k, qq in enumerate(test_q):
    trd = data[data["month"] < qq]
    ted = data[data["month"] == qq]
    if len(ted) < 20000 or len(trd) < 500_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    trd_ok = trd.dropna(subset=["y"])
    m.fit(trd_ok[FEATS], trd_ok["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    sigs = collect(d, GATE)
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

# ТЕТРАДЬ ЦЕЛИКОМ: все сигналы прохода со всеми исходами — база
# офлайн-лаб обучения меты (варианты метки: «перешагнул без убытка»,
# «насколько», «чистыми после фандинга», балансировка, свежее окно)
try:
    rows_nb = []
    for h in hist:
        r = {"month": h["month"], "sym_id": h["sym_id"], "side": h["side"],
             "p": h["p"], "atr_rel": h["atr_rel"],
             "fwd_bp": round(h["fwd_bp"], 2),
             "exit_ret_bp": round(h["exit_ret_bp"], 2),
             "stop_hit": int(h["stop_hit"]),
             "mfe_bp": round(h["mfe_bp"], 1),
             "mae_bp": round(h["mae_bp"], 1),
             "filled": int(h["filled"]),
             "paid_bp": round(h["paid_bp"], 2),
             "base": round(h["base"], 4)}
        for f, vv in zip(MFEATS, h["vec"]):
            r[f] = vv
        rows_nb.append(r)
    pd.DataFrame(rows_nb).to_csv("/kaggle/working/notebook.csv",
                                 index=False)
    print(f"notebook.csv сохранён: {len(rows_nb)} сигналов "
          f"(все исходы + фандинг)", flush=True)
except Exception as e:
    print("тетрадь не сохранена:", str(e)[:120], flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("МЕТА-ПОДТВЕРЖДЕНИЕ ЗАВЕРШЕНО", flush=True)
