"""ПОЛНЫЙ БОЕВОЙ ЭКЗАМЕН 1ч-МОДЕЛИ (предрегистрация 30.09, до запуска).

Кандидаты с экрана лестницы: 1ч @ порог 0.65 (равномерный плюс 6/8) и
1ч + подтверждение дневным трендом. Экзамен повторяет механику ЖИВОГО
бота, а не лабораторную.

МЕХАНИКА (как в бою):
- бары 1ч (из 30м панели), те же 24 формулы в барах, горизонт 10 баров;
- ЛОНГ при p > 0.65, ШОРТ при p < 0.35, cooldown 10 баров на символ
  (отдельно для каждой стороны);
- подтверждение (конфиг 1h+1d): лонг только при дневном тренде ВВЕРХ,
  шорт только при дневном тренде ВНИЗ (последняя ЗАКРЫТАЯ дневная
  свеча против своей SMA50);
- аварийный стоп ±15% по экстремумам баров удержания, выход стопа
  тейкером;
- ДВА сценария издержек на одних и тех же сигналах:
    TAKER (справка): вход close*(1+-slip)+fee, выход тейкер = 0.2%/круг;
    MAKER (БОЕВОЙ, вердикт по нему): вход GTX по цене close сигнального
      бара, fee 0.02%; ИСПЫТАНИЕ НАПОЛНЕНИЯ: заявка исполняется, только
      если СЛЕДУЮЩИЙ бар коснулся цены входа (long: low <= close, short:
      high >= close) — иначе сделки нет (цена убежала); выход тейкер
      (0.05% fee + 0.05% slip) как в бою (фикс по времени — рынком).
- фолды: те же 8 кварталов (linspace), переобучение каждый фолд.

ЗАЯВЛЕННЫЙ КРИТЕРИЙ СДАЧИ (боевой, тот же что провалила широкая):
MAKER-сценарий, все сделки (лонг+шорт): средняя >= +0.10% И плюсовых
кварталов >= 5/8 И нуль-перцентиль >= 95 (50 случайных портфелей той же
плотности на двух последних фолдах, тейкер-лонг база). Отдельно
докладываются лонг-only и тейкер-справка. Сдал -> кандидат на демо-пилот
вместо текущего аналитика; не сдал -> остаёмся на боевом стеке.
"""
import glob, json, time, zipfile, io, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 3600_000
SLIP = 0.0005
TAKER_FEE = 0.0005
MAKER_FEE = 0.0002
STOP = 0.15
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
CONFIGS = [("1h", None), ("1h+1d", "1D")]

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
if not zips:
    json.dump({"error": "panel.zip not found"},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
print("монет в панели:", len(members), flush=True)

def resample(df, rule):
    return df[["open_time", "high", "low", "close", "volume",
               "quote_volume", "taker_buy_base"]] \
        .resample(rule).agg({"open_time": "min", "high": "max",
                             "low": "min", "close": "last",
                             "volume": "sum", "quote_volume": "sum",
                             "taker_buy_base": "sum"}).dropna()

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
    X["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).astype("float32")
    X["skew"] = rets.rolling(20, min_periods=10).skew().astype("float32")
    sma50, sma20 = close.rolling(50).mean(), close.rolling(20).mean()
    X["sma_cross"] = ((close > sma50).astype("float32")
                      - (close > sma20).astype("float32"))
    if sym != "BTCUSDT" and btc_close is not None:
        b = btc_close.reindex(df.index).ffill()
        X["btc_ret_10"] = b.pct_change(10).astype("float32")
        X["btc_ret_20"] = b.pct_change(20).astype("float32")
        bret = b.pct_change()
        X["btc_corr"] = rets.rolling(50, min_periods=20).corr(bret).astype("float32")
        X["rel_ret_20"] = (rets.rolling(20).sum() - bret.rolling(20).sum()).astype("float32")
        X["rel_ret_60"] = (rets.rolling(60).sum() - bret.rolling(60).sum()).astype("float32")
        cov = rets.rolling(480).cov(bret); var = bret.rolling(480).var()
        X["beta_20"] = (cov / var.replace(0, np.nan)).astype("float32")
        X["corr_20"] = rets.rolling(480, min_periods=100).corr(bret).astype("float32")
    X["h4_momentum"] = close.pct_change(48).astype("float32")
    X["hour"] = df.index.hour.astype("float32")
    X["dow"] = df.index.dayofweek.astype("float32")
    X["sym_id"] = np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)
    return X

btc_raw = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
btc_raw["open_time"] = btc_raw["open_time"].astype(np.int64)
btc_raw.index = pd.to_datetime(btc_raw["open_time"], unit="ms")
btc1h = btc_raw["close"].astype("float32").resample("1h").last().dropna()

DAY_MS = 24 * 3600_000
senior_maps = {}
frames = []
SYMS = {}
for i, name in enumerate(members, 1):
    sym = name.replace("_30m.csv", "")
    SYMS[i] = sym
    df = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["open_time", "high", "low", "close",
                              "volume", "quote_volume", "taker_buy_base"])
    for c in df.columns:
        if c != "open_time":
            df[c] = df[c].astype("float32")
    df["open_time"] = df["open_time"].astype(np.int64)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_index()
    sd = resample(df, "1D")
    sma = sd["close"].rolling(50, min_periods=50).mean()
    trend = (sd["close"] > sma).values          # True = вверх
    close_t = sd.index.values.astype("datetime64[ns]").astype(np.int64) \
        // 10**6 + DAY_MS
    senior_maps[sym] = (close_t, trend)
    rd = resample(df, "1h")
    X = make_feats(rd, btc1h, sym)
    close = rd["close"].values
    fut = np.full(len(rd), np.nan, dtype="float32")
    fut[:-H] = close[H:]
    X["y"] = np.where(np.isnan(fut), np.nan,
                      (fut / close > 1).astype("float32"))
    X["fwd"] = (fut / close - 1).astype("float32")
    X["close"] = close
    X["low"] = rd["low"].values
    X["high"] = rd["high"].values
    X["ot"] = rd["open_time"].values.astype(np.int64)
    X["fi"] = np.uint16(i)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=[c for c in X.columns
                                   if c not in ("fwd", "close")]))
    if i % 100 == 0:
        print(f"признаки {i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
print(f"строк {len(data)}, кварталы {test_q}", flush=True)

FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]

def senior_trend(sym, ot_values):
    ct, trend = senior_maps.get(sym, (None, None))
    if ct is None:
        return np.zeros(len(ot_values), dtype=bool)
    j = np.searchsorted(ct, ot_values, side="right") - 1
    ok = j >= 0
    out = np.zeros(len(ot_values), dtype=bool)
    out[ok] = trend[j[ok]]
    return out

def simulate(d, use_confirm):
    """Обе стороны, оба сценария издержек. Возврат DataFrame сделок."""
    pools = {int(fi): g.sort_values("ot")[["ot", "low", "high", "close"]]
             .reset_index(drop=True) for fi, g in d.groupby("fi")}
    conf_cache = {}
    rows = []
    for side, sign in (("L", 1), ("S", -1)):
        sel = d[(d["p"] > GATE) if side == "L" else (d["p"] < 1 - GATE)] \
            .sort_values(["sym_id", "ot"])
        last = {}
        for row in sel.itertuples():
            if row.ot - last.get(row.sym_id, -10**18) < COOLDOWN_MS:
                continue
            last[row.sym_id] = row.ot
            sym = SYMS[int(row.fi)]
            if use_confirm:
                if sym not in conf_cache:
                    conf_cache[sym] = {}
                if row.ot not in conf_cache[sym]:
                    conf_cache[sym][row.ot] = senior_trend(sym, [row.ot])[0]
                up = conf_cache[sym][row.ot]
                if (side == "L" and not up) or (side == "S" and up):
                    continue
            pool = pools.get(int(row.fi))
            if pool is None:
                continue
            ots = pool["ot"].values
            j = int(np.searchsorted(ots, row.ot))
            if j >= len(ots) or ots[j] != row.ot or j + H >= len(ots):
                continue
            seg_lo = pool["low"].values[j + 1: j + 1 + H]
            seg_hi = pool["high"].values[j + 1: j + 1 + H]
            exit_close = pool["close"].values[j + H]
            if np.isnan(exit_close):
                continue
            c = row.close
            if side == "L":
                stop_hit = seg_lo.min() <= c * (1 - STOP)
                stop_px = c * (1 - STOP) * (1 - SLIP)
                exit_taker_px = (stop_px if stop_hit
                                 else exit_close * (1 - SLIP))
                entry_taker = c * (1 + SLIP)
                pnl_taker = exit_taker_px / entry_taker - 1 - TAKER_FEE * 2
                filled = pool["low"].values[j + 1] <= c
                if filled:
                    pnl_maker = (exit_taker_px / c - 1
                                 - MAKER_FEE - TAKER_FEE - SLIP)
                else:
                    pnl_maker = np.nan
            else:
                stop_hit = seg_hi.max() >= c * (1 + STOP)
                stop_px = c * (1 + STOP) * (1 + SLIP)
                exit_taker_px = (stop_px if stop_hit
                                 else exit_close * (1 + SLIP))
                entry_taker = c * (1 - SLIP)
                pnl_taker = (entry_taker - exit_taker_px) / entry_taker \
                    - TAKER_FEE * 2
                filled = pool["high"].values[j + 1] >= c
                if filled:
                    pnl_maker = (c - exit_taker_px) / c \
                        - MAKER_FEE - TAKER_FEE - SLIP
                else:
                    pnl_maker = np.nan
            rows.append({"side": side, "quarter": row.quarter,
                         "pnl_taker": pnl_taker, "pnl_maker": pnl_maker,
                         "p": row.p, "sym": sym})
    return pd.DataFrame(rows)

def null_pct(te, n_real, seeds=50):
    rng = np.random.default_rng(7)
    sub = te[["close", "fwd"]].dropna()
    if n_real < 20 or len(sub) < n_real:
        return None
    out = []
    for _ in range(seeds):
        s = sub.sample(n=n_real, random_state=rng.integers(1 << 30))
        entry = s["close"].values * (1 + SLIP)
        exit_ = s["close"].values * (1 + s["fwd"].values) * (1 - SLIP)
        out.append((exit_ / entry - 1 - 2 * TAKER_FEE).mean() * 100)
    return out

report = {"pre_reg": "вердикт по MAKER (вход GTX+испытание наполнения, "
                     "выход тейкер), все сделки: avg>=+0.10%, 5/8 кварталов, "
                     "нуль>=95 (50 семян, тейкер-лонг плотность)",
          "folds": test_q, "configs": {}}
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 1000 or len(trd) < 100_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(trd[FEATS], trd["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    for cname, use_confirm in CONFIGS:
        T = simulate(d, use_confirm)
        T["quarter"] = qq
        report.setdefault("trades", {}).setdefault(cname, []).append(T)
        mk = T["pnl_maker"].dropna()
        tk = T["pnl_taker"]
        print(f"[{cname}] {qq}: лонг {len(T[T['side']=='L'])} / шорт "
              f"{len(T[T['side']=='S'])} | тейкер {tk.mean()*100:+.3f}% | "
              f"мейкер {mk.mean()*100 if len(mk) else float('nan'):+.3f}% "
              f"(наполнено {len(mk)}/{len(T)}) | {time.time()-t0:.0f}с",
              flush=True)
    print(f"фолд {qq} готов | {time.time()-t0:.0f}с", flush=True)

for cname, use_confirm in CONFIGS:
    T = pd.concat(report["trades"][cname], ignore_index=True)
    row = {}
    for scen, col in (("taker", "pnl_taker"), ("maker", "pnl_maker")):
        s = T.dropna(subset=[col])
        for scope, mask in (("all", s["side"].notna()),
                            ("long", s["side"] == "L"),
                            ("short", s["side"] == "S")):
            ss = s[mask]
            if not len(ss):
                continue
            avg = float(ss[col].mean() * 100)
            by_q = {kk: round(v * 100, 3) for kk, v in
                    ss.groupby("quarter")[col].mean().items()}
            pos_q = sum(1 for v in by_q.values() if v > 0)
            row[f"{scen}_{scope}"] = {
                "n": int(len(ss)), "avg": round(avg, 4),
                "pos_quarters": pos_q, "by_quarter": by_q}
    mk_all = row.get("maker_all", {})
    nulls = []
    last2_q = [qq for qq in test_q[-2:]]
    for qq in last2_q:
        tT = T[(T["quarter"] == qq)]
        n_real = int((tT["side"] == "L").sum())
        ted_q = data[data["quarter"] == qq]
        nl = null_pct(ted_q, n_real)
        mkq = tT["pnl_maker"].dropna()
        if nl and len(mkq):
            nulls.append(float(np.mean(np.array(nl) < mkq.mean() * 100) * 100))
    nl_pct = int(np.mean(nulls)) if nulls else None
    avg = mk_all.get("avg", -9)
    pos_q = mk_all.get("pos_quarters", 0)
    passed = bool(avg >= 0.10 and pos_q >= 5 and (nl_pct or 0) >= 95)
    row["null_pct"] = nl_pct
    row["passed"] = passed
    report["configs"][cname] = row
    T.to_csv(f"/kaggle/working/trades_{cname.replace('+','_')}.csv", index=False)
    print(f"\n[{cname}] ЭКЗАМЕН: мейкер-все {avg:+.4f}% (n={mk_all.get('n')}), "
          f"плюс-кв {pos_q}/8, нуль {nl_pct}% -> "
          f"{'СДАЛ' if passed else 'НЕ СДАЛ'}", flush=True)
    print(f"   лонг-мейкер {row.get('maker_long',{}).get('avg')}% | "
          f"шорт-мейкер {row.get('maker_short',{}).get('avg')}% | "
          f"тейкер-все {row.get('taker_all',{}).get('avg')}%", flush=True)
del report["trades"]
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("экзамен завершён", flush=True)
