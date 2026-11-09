"""ПОЛНЫЙ СТЕК ФИЛЬТРОВ НА ЧЕСТНЫХ ФОЛДАХ (вопрос владельца 08.11:
сколько сделок даёт система со ВСЕМИ фильтрами и не голодает ли она).
Порядок — как в бою (ml5h/engine.py):
  1) сырой гейт 0.65/0.35, кулдаун 5ч/монету;
  2) МЕТА-вето: обучается ТОЛЬКО на сигналах прошлых фолдов (признаки
     сигнала + сырая p + atr), вето при p_win < 0.5; с 2000 исторических;
  3) тихая треть: atr >= живая 1/3-квантиль вселенной бара;
  4) калибровка: истинный шанс >= 0.60 (изотоническая карта обучается
     ТОЛЬКО на предсказаниях прошлых фолдов — leave-current-out);
  5) слоты: максимум 10 одновременных позиций (5ч каждая), по времени.
Симуляция сделки: вход по цене закрытия сигнального бара (заполнение
первым баром), выход t+10 баров по закрытию, стоп -15% внутри (стоп
раньше цели — худший случай), издержки 12 бп. Параметры рецепта
заморожены. Модель каждого фолда учится строго до своего квартала.
"""
import io, json, os, time, zipfile, zlib
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
GATE = 0.65
COOLDOWN_MS = 5 * 3600_000
SLIP, MAKER_FEE, TAKER_FEE = 0.0005, 0.0002, 0.0005
STOP = 0.15
H = 10
SLOTS = 10
GATE_TRUE = 0.60
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
META_PARAMS = dict(n_estimators=120, learning_rate=0.05, max_depth=3,
                   num_leaves=7, subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=4, verbosity=-1)
FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
SYM_IDX = FEATS.index("sym_id")
EXCLUDE = {zlib.crc32(b"BTCUSDT") & 0xFFFF,
           zlib.crc32(b"ETHUSDT") & 0xFFFF}


def log(m):
    print(f"{m} | {time.time()-t0:.0f}с", flush=True)


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
    X["zscore"] = ((close - sma) / std).replace([np.inf, -np.inf], np.nan) \
        .astype("float32")
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


def find_zip(name):
    for root, _d, fs in os.walk("/kaggle/input"):
        if name in fs:
            return os.path.join(root, name)
    return None


def build_arrays(zpath):
    zf = zipfile.ZipFile(zpath)
    members = sorted(n for n in zf.namelist() if n.endswith("_30m.csv"))
    log(f"панель: {len(members)} монет")
    btc = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
    btc["open_time"] = btc["open_time"].astype(np.int64)
    btc.index = pd.to_datetime(btc["open_time"], unit="ms")
    btc = btc["close"].astype("float32")
    Fs, ys, ots, cs, lo, hi, atrs = [], [], [], [], [], [], []
    bounds, syms = [], []      # границы блока каждой монеты в склейке
    off = 0
    for i, name in enumerate(members, 1):
        sym = name.replace("_30m.csv", "")
        df = pd.read_csv(io.BytesIO(zf.read(name)),
                         usecols=["open_time", "high", "low", "close",
                                  "volume", "quote_volume",
                                  "taker_buy_base"])
        for c in df.columns:
            if c != "open_time":
                df[c] = df[c].astype("float32")
        df["open_time"] = df["open_time"].astype(np.int64)
        df.index = pd.to_datetime(df["open_time"], unit="ms")
        X, tr = make_feats(df, btc, sym)
        close = df["close"].values
        atr = (tr.rolling(14).mean() / close).values.astype("float64")
        fut = np.full(len(df), np.nan, dtype="float32")
        fut[:-10] = close[10:]
        fwd = (fut / close - 1).astype("float64")
        thr = 0.5 * np.nan_to_num(atr, nan=np.nan)
        y = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                     np.where(fwd > thr, 1.0,
                              np.where(fwd < -thr, 0.0, np.nan)))
        X["y"] = y.astype("float32")
        X = X.replace([np.inf, -np.inf], np.nan)
        fcols = [c for c in FEATS if c in X.columns]
        keep = X[fcols + ["y"]].notna().all(axis=1).values
        if keep.any():
            Fs.append(X.loc[keep, fcols]
                      .reindex(columns=FEATS).to_numpy(dtype="float32"))
            ys.append(y[keep].astype("float32"))
            ots.append(df["open_time"].values[keep])
            cs.append(close[keep])
            lo.append(df["low"].values[keep])
            hi.append(df["high"].values[keep])
            atrs.append(atr[keep])
            bounds.append(off); syms.append(sym)
            off += int(keep.sum())
        del X, df
        if i % 150 == 0:
            log(f"{i}/{len(members)}")
    F = np.concatenate(Fs); del Fs
    bounds.append(off)     # правая граница последнего блока
    return {"F": F, "y": np.concatenate(ys),
            "ot": np.concatenate(ots),
            "close": np.concatenate(cs),
            "low": np.concatenate(lo), "high": np.concatenate(hi),
            "atr": np.concatenate(atrs),
            "bounds": bounds, "syms": syms}


zp = find_zip("panel.zip")
assert zp, "panel.zip не найден"
A = build_arrays(zp)
# Панель склеена по монетам: строки СВОЕЙ монеты идут подряд, у каждой
# монеты свой блок [bounds[i], bounds[i+1]). Проход выхода по сделке
# делает шаги по СВОЕМУ блоку — глобальный searchsorted по времени тут
# не нужен вовсе и был бы враньём: сосед по массиву = чужая монета
# (потерю сделок несортированным поиском поймал помощник 08.11).
RANGE = {zlib.crc32(s.encode()) & 0xFFFF: (A["bounds"][i], A["bounds"][i + 1])
         for i, s in enumerate(A["syms"])}
_yr = pd.DatetimeIndex(pd.to_datetime(A["ot"], unit="ms")).year
_qt = pd.DatetimeIndex(pd.to_datetime(A["ot"], unit="ms")).quarter
A["qc"] = ((_yr - 2020) * 4 + (_qt - 1)).astype(np.int16)
del _yr, _qt
codes = sorted(int(c) for c in np.unique(A["qc"]))
qq_all = [f"{2020 + c // 4}Q{c % 4 + 1}" for c in codes]
_ix = np.linspace(0, len(qq_all) - 1, 8).round().astype(int)
TEST_Q = [qq_all[i] for i in _ix]
log(f"экзамен: {TEST_Q}")


def qcode_from_str(qq):
    y, q = qq.split("Q")
    return (int(y) - 2020) * 4 + int(q) - 1


stats_by_fold = []
meta_hist = []          # сигналы прошлых фолдов для меты: (vec, win)
cal_hist = []           # (p, win_true) прошлых фолдов для изотоники
iso = None

for qq in TEST_Q:
    code = qcode_from_str(qq)
    mtr = np.flatnonzero(A["qc"] < code)
    mte = np.flatnonzero(A["qc"] == code)
    if len(mte) < 5000 or len(mtr) < 200_000:
        log(f"{qq}: пропущен")
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(A["F"][mtr], A["y"][mtr], categorical_feature=[SYM_IDX])
    p = m.predict_proba(A["F"][mte])[:, 1]
    del m
    # кандидаты сырого гейта + кулдаун 5ч
    idx = pd.DataFrame({"sid": A["F"][mte, SYM_IDX].astype(int),
                        "ot": A["ot"][mte], "p": p, "atr": A["atr"][mte],
                        "gi": mte})
    idx = idx.sort_values(["sid", "ot"])
    cand = []
    last = {}
    for row in idx.itertuples():
        if row.ot - last.get(row.sid, -10**18) < COOLDOWN_MS:
            continue
        last[row.sid] = row.ot
        if row.p > GATE or row.p < 1 - GATE:
            cand.append(row)
    n_raw = len(cand)

    # ---- 2) МЕТА (только на прошлых сигналах) ----
    cut_meta = 0
    if len(meta_hist) >= 2000:
        MX = np.array([m[0] for m in meta_hist])
        MY = np.array([m[1] for m in meta_hist])
        mm = lgb.LGBMClassifier(**META_PARAMS)
        mm.fit(MX, MY)
        kept = []
        for row in cand:
            fv = A["F"][row.gi]     # признаки сигнала — из своей строки
            v = np.concatenate([fv[:SYM_IDX], fv[SYM_IDX + 1:],
                                [row.p, row.atr]])
            pw = float(mm.predict(v.reshape(1, -1))[0])
            if pw >= 0.5:
                kept.append(row)
            else:
                cut_meta += 1
        cand = kept
        del mm
    n_meta = len(cand)

    # ---- 3) тихая треть ----
    kept = []
    dfq = pd.DataFrame({"ot": [r.ot for r in cand], "atr": [r.atr for r in cand]})
    q33 = dfq.groupby("ot")["atr"].quantile(1 / 3)
    qmap = q33.to_dict()
    cut_quiet = 0
    for row in cand:
        if row.atr >= qmap.get(row.ot, 0):
            kept.append(row)
        else:
            cut_quiet += 1
    cand = kept
    n_quiet = len(cand)

    # ---- 4) калибровка (изо на прошлых фолдах) ----
    cut_cal = 0
    iso = None
    if len(cal_hist) >= 500:
        ip = np.array([c[0] for c in cal_hist])
        iy = np.array([1.0 if c[1] else 0.0 for c in cal_hist])
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(ip, iy)
    if iso is not None:
        kept = []
        for row in cand:
            t = float(iso.predict([row.p])[0])
            if (row.p > GATE and t >= GATE_TRUE) or \
               (row.p < 1 - GATE and 1 - t >= GATE_TRUE):
                kept.append(row)
            else:
                cut_cal += 1
        cand = kept
    n_cal = len(cand)

    # ---- 5) слоты: 10 одновременных, по времени ----
    cand_sorted = sorted(cand, key=lambda r: r.ot)
    entries = []          # (sid, ot, p, atr, side)
    open_until = []
    cut_slots = 0
    for row in cand_sorted:
        while open_until and open_until[0] <= row.ot:
            open_until.pop(0)
        if len(open_until) >= SLOTS:
            cut_slots += 1
            continue
        side = 1 if row.p > GATE else -1
        entries.append((row.sid, row.ot, row.p, row.atr, side, row.gi))
        open_until.append(row.ot + H * 1800_000)

    # ---- исходы сделок (экзаменационная формула) ----
    ots = A["ot"]; lows = A["low"]; highs = A["high"]; cls = A["close"]
    t_res = []
    for sid, ot, p_, atr, side, gi in entries:
        s_, e_ = RANGE.get(sid, (-1, -1))
        if not (s_ <= gi < e_) or ots[gi] != ot or gi + 1 >= e_:
            continue          # сделка вне блока своей монеты — санити
        c0 = cls[gi]
        stop = c0 * (1 - STOP) if side == 1 else c0 * (1 + STOP)
        exit_px = None
        for k in range(gi + 1, min(e_, gi + 1 + H)):
            if side == 1 and lows[k] <= stop:
                exit_px = stop; break
            if side == -1 and highs[k] >= stop:
                exit_px = stop; break
        if exit_px is None and gi + H < e_:
            exit_px = cls[gi + H]
        if exit_px is None:
            continue
        pnl_bp = (side * (exit_px / c0 - 1) - MAKER_FEE - TAKER_FEE - SLIP) * 1e4
        t_res.append((sid, ot, side, pnl_bp, gi))
        win = pnl_bp > 0
        vec = np.concatenate([A["F"][gi][:SYM_IDX],
                              A["F"][gi][SYM_IDX + 1:],
                              [p_, atr]])
        meta_hist.append((vec, win))
        cal_hist.append((p_, win))

    wins = sum(1 for t in t_res if t[3] > 0)
    ndays = max(1, int((ots[mte].max() - ots[mte].min()) // 86400000) + 1)
    stats_by_fold.append(dict(
        fold=qq, n_raw=n_raw, cut_meta=cut_meta, cut_quiet=cut_quiet,
        cut_cal=cut_cal, cut_slots=cut_slots, n_entries=len(t_res),
        wins=wins, wr=round(wins / max(1, len(t_res)) * 100, 1),
        avg_bp=round(float(np.mean([t[3] for t in t_res])), 1) if t_res else 0,
        sum_bp=round(float(np.sum([t[3] for t in t_res])), 0),
        per_day=round(len(t_res) / ndays, 2), days=ndays))
    log(f"{qq}: сырых {n_raw} -> мета -{cut_meta} -> тихая -{cut_quiet} "
        f"-> калибр -{cut_cal} -> слоты -{cut_slots} -> сделок {len(t_res)} "
        f"(win {wins}/{len(t_res)})")
    # пополнить копилку калибровки исходами этого фолда — сделано выше
    # (cal_hist наполняется после каждого фолда)
    pass


import json
with open("/kaggle/working/full_stack_report.json", "w") as f:
    json.dump({"folds": stats_by_fold}, f, indent=1, default=float)

print("\nФОЛД | сырых | -мета | -тихая | -калибр | -слоты | сделок | win% | avg бп | в день", flush=True)
tot_entries = 0; tot_days = 0
for s_ in stats_by_fold:
    print(f"{s_['fold']} | {s_['n_raw']} | -{s_['cut_meta']} | -{s_['cut_quiet']} | "
          f"-{s_['cut_cal']} | -{s_['cut_slots']} | {s_['n_entries']} | {s_['wr']}% | "
          f"{s_['avg_bp']} | {s_['per_day']}/день", flush=True)
    tot_entries += s_["n_entries"]; tot_days += s_["days"]
print(f"\nВСЕГО сделок: {tot_entries} за {tot_days} дней = "
      f"{tot_entries/max(1,tot_days):.1f}/день", flush=True)
log("ПОЛНЫЙ СТЕК ГОТОВ")
