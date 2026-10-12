"""ВХОД БУРИ С ОТТЯЖКОЙ (предрегистрация 10.10, вариант 1 списка
«путь сделки», приказ владельца «надо что-то придумывать ещё»).

ВОПРОС: буря входит по цене сигнала (закрытие бара, мейкер GTX). Дип-
принцип — единственный класс, сдававший экзамены: вход ЛИМИТКОЙ на
откате k×ATR от цены сигнала, заявка живёт m баров, не наполнилась —
сделки нет.

ВАРИАНТЫ (на ОДНИХ сигналах, cooldown тот же):
  BASE      вход close[t], выход close[t+10] — как в бою/экзамене;
  LIM_same  лимит k×ATR, наполнилась в t+1..t+m -> выход ВСЁ РАВНО
            close[t+10] (чистый эффект цены входа, диагностика);
  LIM_hold  лимит k×ATR, наполнилась в баре f -> холд 10 баров ОТ
            НАПОЛНЕНИЯ, выход close[f+10] (как будет в бою — ГЛАВНЫЙ).

СЕТКА: k ∈ {0.25, 0.5, 1.0}×ATR14, m ∈ {2, 4}. Заполнение касанием
(лоу<=лимит для лонга), цена исполнения = лимит (гэп лучше лимита
не учитываем — консервативно). Издержки у всех одинаковые: вход
мейкер 5бп + выход тейкер 5бп = 10бп (вход лимиткой и есть мейкер).

ПРЕДРЕГИСТРАЦИЯ ВЕРДИКТА: главный вариант объявлен до прогона —
LIM_hold k=0.5, m=2. КАНДИДАТ, если avg(LIM_hold 0.5/2) − avg(BASE)
>= +3 бп, n >= 1000, наполнение >= 30%. Полная таблица сетки — для
прозрачности, НЕ для выбора лучшего постфактум. Оговорка честности:
дурной отбор заявки (лимитка наполняется на провалах) симуляцией по
касаниям учтён частично; кандидат перед боем всё равно идёт на
свежие дни. Ничего не деплоится.
"""
import glob, io, json, time, zipfile, zlib
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000
KS = [0.25, 0.5, 1.0]
MS = [2, 4]
SEG = 14                      # m_max + H: бары t+1..t+14
FEE_ALL = 0.0010              # вход мейкер 5бп + выход тейкер 5бп
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
    X["high"] = df["high"].values
    X["low"] = df["low"].values
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
print(f"строк {len(data)}, фолды {test_q} | {time.time()-t0:.0f}с",
      flush=True)

def collect(d):
    pools = {int(fi): gg.sort_values("ot") for fi, gg in d.groupby("sym_id")}
    trades = []
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
            # t+14 баров должны существовать (наполнение до t+4 + холд 10)
            if j >= len(ots) or ots[j] != row.ot or j + SEG >= len(ots):
                continue
            seg_h = pool["high"].values[j + 1: j + 1 + SEG].astype("float64")
            seg_l = pool["low"].values[j + 1: j + 1 + SEG].astype("float64")
            seg_c = pool["close"].values[j + 1: j + 1 + SEG].astype("float64")
            if len(seg_h) < SEG:
                continue
            trades.append({"side": side, "quarter": row.quarter,
                           "entry": float(row.close),
                           "atr_rel": float(row.atr_rel),
                           "fwd": float(row.fwd) if row.fwd == row.fwd
                           else np.nan,
                           "seg_h": seg_h, "seg_l": seg_l, "seg_c": seg_c})
    return trades

def lim_pnl(t, k, m, hold_from_fill):
    """Лимит на откате k×ATR, окно m баров. Не наполнилась -> None."""
    side, e = t["side"], t["entry"]
    lim = e * (1 - k * t["atr_rel"]) if side == 1 \
        else e * (1 + k * t["atr_rel"])
    seg_l, seg_h, seg_c = t["seg_l"], t["seg_h"], t["seg_c"]
    for i in range(m):
        touched = seg_l[i] <= lim if side == 1 else seg_h[i] >= lim
        if not touched:
            continue
        if hold_from_fill:
            exit_px = seg_c[i + H]          # холд 10 баров от наполнения
        else:
            exit_px = seg_c[H - 1]          # тот же t+10, что у BASE
        return side * (exit_px / lim - 1) - FEE_ALL, i
    return None, None

acc = {"BASE": [], "LIM_hold_05_2": [], **{f"LIM_same_k{k}_m{m}": []
       for k in KS for m in MS},
       **{f"LIM_hold_k{k}_m{m}": [] for k in KS for m in MS}}
fills = {f"LIM_hold_k{k}_m{m}": {"fill": 0, "no": 0}
         for k in KS for m in MS}
all_folds = []
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 20000 or len(trd) < 500_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    trd_ok = trd.dropna(subset=["y"])
    m.fit(trd_ok[FEATS], trd_ok["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    trades = collect(d)
    for t in trades:
        acc["BASE"].append(t["side"] * t["fwd"] - FEE_ALL)
        for kk in KS:
            for mm in MS:
                # диагностика: чистый эффект цены входа на том же выходе
                p_same, _ = lim_pnl(t, kk, mm, hold_from_fill=False)
                if p_same is not None:
                    acc[f"LIM_same_k{kk}_m{mm}"].append(p_same)
                # главный механик боя: холд от наполнения
                p_hold, fi = lim_pnl(t, kk, mm, hold_from_fill=True)
                key = f"LIM_hold_k{kk}_m{mm}"
                if p_hold is None:
                    fills[key]["no"] += 1
                else:
                    fills[key]["fill"] += 1
                    acc[key].append(p_hold)
                if kk == 0.5 and mm == 2:
                    if p_hold is not None:
                        acc["LIM_hold_05_2"].append(p_hold)
    all_folds.append(qq)
    print(f"фолд {qq}: сделок {len(trades)} | {time.time()-t0:.0f}с",
          flush=True)

report = {"folds": [str(f) for f in all_folds],
          "pre_reg": "ГЛАВНЫЙ: LIM_hold k=0.5 m=2. КАНДИДАТ если "
                     "avg(LIM_hold_05_2) − avg(BASE) >= +3 бп, n>=1000, "
                     "наполнение >=30%. Сетка — таблица, не выбор.",
          "variants": {}}
base_avg = float(np.mean(acc["BASE"]) * 10000)
nb = len(acc["BASE"])
report["variants"]["BASE"] = {"n": nb, "avg_bp": round(base_avg, 2)}
print(f"BASE: {nb} сделок, {base_avg:+.2f} бп", flush=True)
for name, lst in acc.items():
    if name in ("BASE", "LIM_hold_05_2"):
        continue
    if not lst:
        continue
    a = np.array(lst, dtype="float64")
    avg = float(a.mean() * 10000)
    f = fills.get(name)
    fr = (f["fill"] / max(1, f["fill"] + f["no"]) * 100) if f else None
    report["variants"][name] = {
        "n": len(lst), "avg_bp": round(avg, 2),
        "delta_vs_base": round(avg - base_avg, 2),
        "fill_pct": round(fr, 1) if fr is not None else None}
    fr_s = f"{fr:.0f}%" if fr is not None else "диагностика"
    print(f"{name}: n={len(lst)} avg {avg:+.2f} бп "
          f"(Δ{avg-base_avg:+.2f}) наполнение {fr_s}", flush=True)

# главный вариант по предрегистрации
g = acc["LIM_hold_05_2"]
f = fills["LIM_hold_k0.5_m2"]
fr = f["fill"] / max(1, f["fill"] + f["no"]) * 100
g_avg = float(np.mean(g) * 10000) if g else float("nan")
delta = g_avg - base_avg
report["variants"]["LIM_hold_05_2"] = {
    "n": len(g), "avg_bp": round(g_avg, 2), "delta_vs_base": round(delta, 2),
    "fill_pct": round(fr, 1),
    "candidate": bool(delta >= 3.0 and len(g) >= 1000 and fr >= 30)}
print(f"ГЛАВНЫЙ LIM_hold 0.5/2: n={len(g)} avg {g_avg:+.2f} бп "
      f"(Δ{delta:+.2f}), наполнение {fr:.0f}% -> "
      f"{'КАНДИДАТ' if report['variants']['LIM_hold_05_2']['candidate'] else 'НЕТ'}",
      flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ОТТЯЖКА завершена", flush=True)
