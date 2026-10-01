"""ФАНДИНГ-ГИПОТЕЗА (предрегистрация 30.09, до запуска).

ВОПРОС: добавляет ли ставка фандинга край модели на ЧЕСТНЫХ данных?
Шаг-2 выбросил фандинг на утечной базе; честно его никто не проверял.

ДИЗАЙН: та же панель, признаки, ярлыки, параметры и 8 тестовых
кварталов, что у боевого ядра v3. Два конфига на каждом фолде:
база (24 признака) и база+фандинг (29). Пороги 0.55/0.60/0.65
оцениваются на общих предсказаниях каждого конфига. Лонг p>gate,
cooldown 10 баров, pnl = вход close*(1+slip), выход close[t+10]*(1-slip),
− taker x2; стопы не моделируем.

ФАНДИНГ-ПРИЗНАКИ (события 8ч переносились на 30м бары ffill):
  funding = последняя ставка; funding_7d = среднее 336 баров (7 дней);
  funding_30d = среднее 1440 баров (30 дней); funding_z = отклонение
  от 30д среднего в сигмах; has_funding = флаг наличия данных.
Нет данных (до 2024) -> нули + флаг 0. Покрытие: 2024 год + янв 2025
.. авг 2026.

ЗАЯВЛЕННЫЕ КРИТЕРИИ (до прогона):
  ПЕРВИЧНЫЙ: на фолдах с покрытием (2024Q2, 2025Q1, 2025Q4, 2026Q3),
    порог 0.60: Δ = avg(fund) − avg(base) >= +0.05% -> ЖИВА;
    Δ <= 0 -> ВРЕДИТ; иначе -> НЕ ДАЛА.
  ПЛАЦЕБО: на фолдах без фандинга (2021Q3..2023Q4) |Δ| <= 0.03%;
    если больше — подозрение на баг реализации, вердикт аннулируется.
  ВТОРИЧНЫЕ (отчёт, не решение): пороги 0.55 и 0.65.
Экран ничего не деплоит.
"""
import glob, json, os, time, zipfile, io, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
H = 10
COOLDOWN_MS = 10 * 1800_000
SLIP, FEE2 = 0.0005, 0.001
GATES = [0.55, 0.60, 0.65]
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
COVERED_Q = {"2024Q2", "2025Q1", "2025Q4", "2026Q3"}

def sym_id(sym): return np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)

# --- панель ---
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

# --- фандинг: словарь символ -> Series(ставка) ---
# Кегл может отдать папки и распакованными (f2024/X.npz), и zip-архивами
# (f2024.zip) — обрабатываем оба случая, плюс диагностика входов.
fund_series = {}
def _add_npz(sym, buf_or_path):
    try:
        z = np.load(buf_or_path, allow_pickle=True)
        if "t" not in z.files or "rate" not in z.files:
            return
        idx = pd.to_datetime(z["t"].astype(np.int64), unit="ms")
        fund_series.setdefault(sym, []).append(
            pd.Series(z["rate"].astype(float), index=idx))
    except Exception:
        pass

for f in glob.glob("/kaggle/input/**/*.npz", recursive=True):
    _add_npz(f.split("/")[-1].replace(".npz", ""), f)
if not fund_series:                       # папки приехали zip-ами
    for zp in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        if "panel" in zp:
            continue
        try:
            zf2 = zipfile.ZipFile(zp)
            for n in zf2.namelist():
                if n.endswith(".npz"):
                    sym = n.split("/")[-1].replace(".npz", "")
                    with zf2.open(n) as fh:
                        _add_npz(sym, io.BytesIO(fh.read()))
        except Exception:
            continue
fund_series = {s: pd.concat(v).sort_index() for s, v in fund_series.items()}
fund_series = {s: ser[~ser.index.duplicated(keep="last")]
               for s, ser in fund_series.items()}
print("фандинг-символов:", len(fund_series), flush=True)
if not fund_series:
    tree = {}
    for root, dirs, fs in os.walk("/kaggle/input"):
        rel = os.path.relpath(root, "/kaggle/input")
        if rel.count(os.sep) <= 1:
            tree[rel] = (len(dirs), len(fs), fs[:5])
    print("ВХОДЫ:", tree, flush=True)
    json.dump({"error": "funding npz not found", "input_tree": str(tree)},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)

def funding_feats(df, sym):
    out = pd.DataFrame(index=df.index)
    fr = fund_series.get(sym)
    if fr is None:
        for c in ("funding", "funding_7d", "funding_30d", "funding_z"):
            out[c] = np.float32(0)
        out["has_funding"] = np.float32(0)
        return out
    last = fr.reindex(df.index, method="ffill")
    m7 = last.rolling(336, min_periods=48).mean()
    m30 = last.rolling(1440, min_periods=240).mean()
    s30 = last.rolling(1440, min_periods=240).std()
    out["funding"] = last.astype("float32")
    out["funding_7d"] = m7.astype("float32")
    out["funding_30d"] = m30.astype("float32")
    out["funding_z"] = ((last - m30) / s30.replace(0, np.nan)).astype("float32")
    out["has_funding"] = last.notna().astype("float32")
    return out.fillna(np.float32(0))

FEATS_BASE = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
              "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
              "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
              "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
              "hour", "dow", "sym_id"]
FEATS_FUND = FEATS_BASE + ["funding", "funding_7d", "funding_30d",
                           "funding_z", "has_funding"]

parts = []
for i, name in enumerate(members, 1):
    sym = name.replace("_30m.csv", "")
    df = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["open_time", "high", "low", "close",
                              "volume", "quote_volume", "taker_buy_base"])
    for c in df.columns:
        if c != "open_time":          # 13-значные метки не влезают в float32
            df[c] = df[c].astype("float32")
    df["open_time"] = df["open_time"].astype(np.int64)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    close = df["close"]
    rets = close.pct_change()
    X = pd.DataFrame(index=df.index)
    for h in (1, 4, 10, 20):
        X[f"ret_{h}"] = close.pct_change(h).astype("float32")
    tr = pd.concat([df["high"] - df["low"], (df["high"] - close.shift(1)).abs(),
                    (df["low"] - close.shift(1)).abs()], axis=1).max(axis=1)
    X["atr_pct"] = (tr.rolling(14).mean() / close).astype("float32")
    X["std_20"] = rets.rolling(20).std().astype("float32")
    X["vol_ratio"] = (df["volume"] / df["volume"].rolling(20).mean()
                      .replace(0, np.nan)).astype("float32")
    tb_ratio = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    X["taker_buy"] = tb_ratio.rolling(20, min_periods=10).mean().astype("float32")
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    X["clv"] = ((close - df["low"]) / rng - 0.5).rolling(20, min_periods=10) \
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
    if sym != "BTCUSDT" and btc is not None:
        b = btc.reindex(df.index).ffill()
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
    X["sym_id"] = sym_id(sym)
    F = funding_feats(df, sym)
    for c in F.columns:
        X[c] = F[c]
    fut_close = np.full(len(df), np.nan, dtype="float32")
    fut_close[:-H] = close.values[H:]
    X["y"] = np.where(np.isnan(fut_close), np.nan,
                      (fut_close / close.values > 1).astype("float32"))
    X["fwd"] = (fut_close / close.values - 1).astype("float32")
    X["close"] = close.values
    X["ot"] = df["open_time"].values
    X = X.replace([np.inf, -np.inf], np.nan)
    parts.append(X.dropna(subset=[c for c in X.columns
                                  if c not in ("fwd", "close")]))
    if i % 100 == 0:
        print(f"признаки {i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(parts, ignore_index=True)
del parts
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
cov = data[data["quarter"].isin(COVERED_Q)]["has_funding"].mean()
print(f"строк {len(data)}, покрытие фандингом на тест-фолдах: {cov*100:.0f}% "
      f"| кварталы: {test_q}", flush=True)

def gate_trades(d, gate):
    dl = d[d["p"] > gate].sort_values(["sym_id", "ot"])
    kept, last = [], {}
    for row in dl.itertuples():
        if row.ot - last.get(row.sym_id, -10**18) >= COOLDOWN_MS:
            kept.append((row.ot, row.sym_id, row.close, row.p,
                         row.quarter, row.fwd))
            last[row.sym_id] = row.ot
    if not kept:
        return pd.DataFrame(columns=["pnl", "p", "quarter"])
    T = pd.DataFrame(kept, columns=["ot", "sym_id", "close", "p",
                                    "quarter", "fwd"])
    T = T[~np.isnan(T["fwd"].values)].reset_index(drop=True)
    entry = T["close"].values * (1 + SLIP)
    exit_ = T["close"].values * (1 + T["fwd"].values) * (1 - SLIP)
    T["pnl"] = exit_ / entry - 1 - FEE2
    return T

acc = {c: {g: [] for g in GATES} for c in ("base", "fund")}
for qq in test_q:
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 5000 or len(trd) < 200_000:
        continue
    line = f"фолд {qq}:"
    for cname, feats in (("base", FEATS_BASE), ("fund", FEATS_FUND)):
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(trd[feats], trd["y"], categorical_feature=["sym_id"])
        d = ted.assign(p=m.predict_proba(ted[feats])[:, 1])
        for g in GATES:
            T = gate_trades(d, g)
            acc[cname][g].append(T)
            avg = T["pnl"].mean() * 100 if len(T) else float("nan")
            line += f" | {cname} g{g}: {len(T)}шт {avg:+.3f}%"
    print(line + f" | {time.time()-t0:.0f}с", flush=True)

report = {"pre_reg": "первичный: покрытые фолды, gate 0.60, Δ>=+0.05% ЖИВА; "
                     "плацебо |Δ|<=0.03%; покрытие " + str(sorted(COVERED_Q)),
          "gates": {}, "folds": test_q}
D10 = {}
for g in GATES:
    B = pd.concat(acc["base"][g], ignore_index=True)
    F = pd.concat(acc["fund"][g], ignore_index=True)
    row = {}
    for cname, T in (("base", B), ("fund", F)):
        row[cname] = {"n": int(len(T)),
                      "avg": round(float(T["pnl"].mean() * 100), 4),
                      "by_q": {k: round(v * 100, 3) for k, v in
                               T.groupby("quarter")["pnl"].mean().items()}}
    bc = B[B["quarter"].isin(COVERED_Q)]
    fc = F[F["quarter"].isin(COVERED_Q)]
    bp = B[~B["quarter"].isin(COVERED_Q)]
    fp = F[~F["quarter"].isin(COVERED_Q)]
    d_cov = (fc["pnl"].mean() - bc["pnl"].mean()) * 100
    d_pla = (fp["pnl"].mean() - bp["pnl"].mean()) * 100
    row["delta_covered"] = round(float(d_cov), 4)
    row["delta_placebo"] = round(float(d_pla), 4)
    if g == 0.60:
        placebo_ok = abs(d_pla) <= 0.03
        if (row["base"]["n"] == row["fund"]["n"]
                and abs(d_cov) < 1e-6 and abs(d_pla) < 1e-6):
            verdict = "НЕТ ЭФФЕКТА (предсказания идентичны — данные не дошли)"
        else:
            verdict = ("ЖИВА" if d_cov >= 0.05 else
                       "ВРЕДИТ" if d_cov <= 0 else "НЕ ДАЛА")
        if not placebo_ok:
            verdict += " (плацебо-тревога: |Δ|=" + \
                f"{abs(d_pla):.3f}% > 0.03% — проверить реализацию)"
        report["verdict"] = verdict
    report["gates"][str(g)] = row
    print(f"gate {g}: base {row['base']['avg']:+.4f}% ({row['base']['n']}) | "
          f"fund {row['fund']['avg']:+.4f}% ({row['fund']['n']}) | "
          f"Δпокрытые {row['delta_covered']:+.4f}% Δплацебо {row['delta_placebo']:+.4f}%",
          flush=True)
if "verdict" not in report:
    report["verdict"] = "нет данных"
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("ВЕРДИКТ:", report.get("verdict"), flush=True)
