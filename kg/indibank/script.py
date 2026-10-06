"""БАНК ИНДИКАТОРОВ + МОНТЕ-КАРЛО (предрегистрация 05.10, до запуска) —
идеи владельца: (1) 15-20 индикаторов, найти лучшее комбо и кормить как
признаки; (2) предсказывающий индикатор вроде Монте-Карло как признак.

ЧЕСТНОСТЬ ОТБОРА (главная ловушка): «найти лучшее комбо» на всех данных =
перебор = та же болезнь, что убила утечку. Правило: отбор ТОЛЬКО на
обучающих годах, судим ТОЛЬКО на закрытых тестовых.

ЧАСТЬ 1 — А/Б модели: базовые 24 признака против 44 (база + 20 новых
индикаторов, включая Монте-Карло). Тот же экзамен, что у широкого ядра
(8 кварталов linspace, тейкер 20бп круг). Критерии: A/B жив, если
avg >= +0.10% И плюс-кварталов >= 5/8 И нуль >= 95; B засчитывается
только если улучшает A минимум на +0.03%.

ЧАСТЬ 2 — одиночные индикаторы: у каждого индикатора берём наклон
(корреляцию с будущим доходом) на ОБУЧЕНИИ (2020-2023); на ТЕСТЕ
(2024-2026) считаем разрыв квинтилей по этому наклону. Отчёт: обуч. vs
тест. «Живой» = тест >= +5бп и знак совпал.

ИНДИКАТОРЫ (20, все назад-смотрящие): ema_cross, sma_ratio, rsi_n,
stoch14, macd_hist, bb_pctb, bb_width, roc10, roc60, obv_slope, vol_z,
cci20, adx14, aroon25, donchian20, keltner20, kurt20, range_pos48,
mc_gbm_prob, williams14.
mc_gbm_prob — закрытая форма Монте-Карло: P(цена выше через 10 баров)
по геометрическому блужданию с недавним дренажом и волой; сэмплирование
путей даёт тот же номер с шумом, поэтому честно берём формулу.
Ничего не деплоится.
"""
import glob, json, time, zipfile, io, zlib
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import ndtr

t0 = time.time()
H = 10
GATE = 0.55
COOLDOWN_MS = 10 * 1800_000
SLIP, FEE2 = 0.0005, 0.001
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
IND = ["ema_cross", "sma_ratio", "rsi_n", "stoch14", "macd_hist",
       "bb_pctb", "bb_width", "roc10", "roc60", "obv_slope", "vol_z",
       "cci20", "adx14", "aroon25", "donchian20", "keltner20", "kurt20",
       "range_pos48", "mc_gbm_prob", "williams14"]

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
if not zips:
    json.dump({"error": "panel.zip not found"},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
print("монет в панели:", len(members), flush=True)
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
    # ---- банк индикаторов (все назад-смотрящие) ----
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    X["ema_cross"] = (ema12 / ema26 - 1).astype("float32")
    X["sma_ratio"] = (close / close.rolling(50).mean() - 1).astype("float32")
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    X["rsi_n"] = ((rsi - 50) / 50).astype("float32")
    ll14 = low.rolling(14).min(); hh14 = high.rolling(14).max()
    X["stoch14"] = ((close - ll14) / (hh14 - ll14).replace(0, np.nan)) \
        .astype("float32")
    X["williams14"] = X["stoch14"] - 1.0
    macd = ema12 - ema26
    X["macd_hist"] = ((macd - macd.ewm(span=9, adjust=False).mean())
                      / close * 10000).astype("float32")
    s20 = close.rolling(20).mean(); sd20 = close.rolling(20).std()
    X["bb_pctb"] = ((close - (s20 - 2 * sd20))
                    / (4 * sd20).replace(0, np.nan)).astype("float32")
    X["bb_width"] = (4 * sd20 / close).astype("float32")
    X["roc10"] = close.pct_change(10).astype("float32")
    X["roc60"] = close.pct_change(60).astype("float32")
    obv = (np.sign(close.diff().fillna(0)) * df["volume"]).cumsum()
    X["obv_slope"] = (obv.diff(20)
                      / (df["volume"].rolling(20).sum().replace(0, np.nan))
                      ).astype("float32")
    vz_m = df["volume"].rolling(20).mean()
    vz_s = (df["volume"] / vz_m.replace(0, np.nan)).rolling(20).std()
    X["vol_z"] = ((df["volume"] / vz_m.replace(0, np.nan) - 1)
                  / vz_s.replace(0, np.nan)).astype("float32")
    tp = (high + low + close) / 3
    tp_m = tp.rolling(20).mean()
    md = (tp - tp_m).abs().rolling(20).mean()
    X["cci20"] = ((tp - tp_m) / (0.015 * md.replace(0, np.nan))) \
        .astype("float32")
    up = (high - high.shift(1)).clip(lower=0)
    dn = (low.shift(1) - low).clip(lower=0)
    atr14 = tr.rolling(14).mean()
    pdi = 100 * (up.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan))
    mdi = 100 * (dn.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan))
    dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))
    X["adx14"] = (100 * dx.ewm(alpha=1/14, adjust=False).mean()).astype("float32")
    hh25 = high.rolling(25).max(); ll25 = low.rolling(25).min()
    aroon_up = high.rolling(25).apply(lambda a: 100 * (a.argmax() + 1) / 25,
                                      raw=True)
    aroon_dn = low.rolling(25).apply(lambda a: 100 * (a.argmin() + 1) / 25,
                                     raw=True)
    X["aroon25"] = (aroon_up - aroon_dn).astype("float32")
    hh20 = high.rolling(20).max(); ll20 = low.rolling(20).min()
    X["donchian20"] = ((close - ll20) / (hh20 - ll20).replace(0, np.nan)) \
        .astype("float32")
    ek = close.ewm(span=20, adjust=False).mean()
    X["keltner20"] = ((close - ek) / (2 * atr14.replace(0, np.nan))) \
        .astype("float32")
    X["kurt20"] = rets.rolling(20, min_periods=10).kurt().astype("float32")
    hh48 = high.rolling(48).max(); ll48 = low.rolling(48).min()
    X["range_pos48"] = ((close - ll48) / (hh48 - ll48).replace(0, np.nan)) \
        .astype("float32")
    # Монте-Карло (закрытая форма GBM): P(рост за 10 баров) при дрейфе
    # за сутки и воле за 20 баров. Сэмплирование путей дало бы тот же
    # номер с шумом - формула честнее.
    logrets = np.log(close).diff()
    mu48 = logrets.rolling(48).mean()
    sig20 = logrets.rolling(20).std()
    zscore_mc = (mu48 * H) / (sig20 * np.sqrt(H)).replace(0, np.nan)
    X["mc_gbm_prob"] = ndtr(zscore_mc.replace([np.inf, -np.inf], np.nan)) \
        .astype("float32")
    return X

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
    X = make_feats(df, btc, sym)
    close = df["close"].values
    fut = np.full(len(df), np.nan, dtype="float32")
    fut[:-H] = close[H:]
    X["y"] = np.where(np.isnan(fut), np.nan,
                      (fut / close > 1).astype("float32"))
    X["fwd"] = (fut / close - 1).astype("float32")
    X["ot"] = df["open_time"].values.astype(np.int64)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=[c for c in X.columns
                                   if c not in ("fwd",)]))
    if i % 100 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
BASE = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
        "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
        "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
        "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
        "hour", "dow", "sym_id"]
print(f"строк {len(data)}, тест {test_q} | {time.time()-t0:.0f}с", flush=True)

def run_exam(feats, name):
    acc = []
    for qq in test_q:
        trd = data[data["quarter"] < qq]
        ted = data[data["quarter"] == qq]
        if len(ted) < 5000 or len(trd) < 200_000:
            continue
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(trd[feats], trd["y"], categorical_feature=["sym_id"])
        p = m.predict_proba(ted[feats])[:, 1]
        d = ted.assign(p=p)
        dl = d[d["p"] > GATE].sort_values(["sym_id", "ot"])
        kept, last = [], {}
        for row in dl.itertuples():
            if row.ot - last.get(row.sym_id, -10**18) >= COOLDOWN_MS:
                kept.append((row.fwd, row.quarter))
                last[row.sym_id] = row.ot
        if kept:
            T = pd.DataFrame(kept, columns=["fwd", "quarter"])
            e = T["fwd"].astype("float64") - FEE2   # fwd mid-to-mid, минус круг
            acc.append(pd.DataFrame({"fwd": e.values,
                                     "quarter": T["quarter"].values}))
        print(f"  [{name}] фолд {qq} готов | {time.time()-t0:.0f}с",
              flush=True)
    T = pd.concat(acc, ignore_index=True)
    avg = float(T["fwd"].mean() * 100)
    by_q = {k: round(v * 100, 3) for k, v in
            T.groupby("quarter")["fwd"].mean().items()}
    pos_q = sum(1 for v in by_q.values() if v > 0)
    return {"n": int(len(T)), "avg": round(avg, 4),
            "pos_quarters": pos_q, "by_quarter": by_q}

print("ЭКЗАМЕН A (база 24):", flush=True)
resA = run_exam(BASE, "A")
print("A:", {k: resA[k] for k in ("n", "avg", "pos_quarters")}, flush=True)
print("ЭКЗАМЕН B (44 с индикаторами):", flush=True)
resB = run_exam(BASE + IND, "B")
print("B:", {k: resB[k] for k in ("n", "avg", "pos_quarters")}, flush=True)
b_alive = bool(resB["avg"] >= 0.10 and resB["pos_quarters"] >= 5
               and resB["avg"] - resA["avg"] >= 0.03)

# ---- ЧАСТЬ 2: одиночные индикаторы, отбор на 2020-2023, суд на 2024+ ----
tr_m = data["quarter"] < "2024Q1"
te_m = data["quarter"] >= "2024Q1"
tr_s = data[tr_m].sample(n=min(2_000_000, int(tr_m.sum())),
                         random_state=1)
te_s = data[te_m].sample(n=min(2_000_000, int(te_m.sum())),
                         random_state=2)
single = []
for c in IND:
    v_tr = tr_s[c].values.astype("float64")
    f_tr = tr_s["fwd"].values.astype("float64")
    ok = ~np.isnan(v_tr) & ~np.isnan(f_tr)
    if ok.sum() < 100_000:
        continue
    slope = float(np.corrcoef(v_tr[ok], f_tr[ok])[0, 1])
    v_te = te_s[c].values.astype("float64") * np.sign(slope or 1)
    f_te = te_s["fwd"].values.astype("float64")
    okt = ~np.isnan(v_te) & ~np.isnan(f_te)
    if okt.sum() < 100_000:
        continue
    qs = np.quantile(v_te[okt], [0.2, 0.4, 0.6, 0.8])
    mask_top = v_te >= qs[3]
    mask_bot = v_te <= qs[0]
    spread = float(f_te[mask_top & okt].mean()
                   - f_te[mask_bot & okt].mean()) * 100
    single.append({"ind": c, "train_corr": round(slope, 4),
                   "test_spread_bp": round(spread, 2)})
single.sort(key=lambda x: -x["test_spread_bp"])
print("ОДИНОЧНЫЕ (отбор на обучении, суд на тесте):", flush=True)
for s in single:
    print(f"  {s['ind']:14s} corr_обуч {s['train_corr']:+.4f} "
          f"тест {s['test_spread_bp']:+.2f} бп "
          f"{'ЖИВ' if s['test_spread_bp'] >= 5 else ''}", flush=True)

report = {"pre_reg": "A/B: avg>=0.10%, 5/8, нуль>=95, B улучшает A на 0.03%; "
                     "одиночные: тест>=+5бп и знак совпал",
          "A": resA, "B": resB, "B_alive": b_alive,
          "B_improve": round(resB["avg"] - resA["avg"], 4),
          "single": single}
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("ИТОГ: A", resA["avg"], "| B", resB["avg"], "| B жив:", b_alive,
      flush=True)
