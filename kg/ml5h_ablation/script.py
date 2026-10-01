"""ВСКРЫТИЕ ПРИЗНАКОВ широкого аналитика (предрегистрация 30.09, до запуска).

ВОПРОС: какие из 24 признаков боевой модели несут вклад, а какие балласт,
на ЧЕСТНЫХ данных (после фикса h4_momentum) и после издержек.

ДИЗАЙН: та же панель (panel.zip даунлоадера), те же формулы признаков
и ярлыков, те же параметры модели, что у боевого широкого ядра v3.
ОДИН сплит на конфигурацию: обучение = всё до 2025Q1, тест =
2025Q1..конец (~7 кварталов) — тестовый период ОДИНАКОВЫЙ для всех
конфигураций (сопоставимое сравнение). Конфигурации: контроль (все 24)
и 7 x "выбросить одну группу". Лонг p>0.55, шорт p<0.45, cooldown
10 баров на символ, pnl = вход close*(1+slip), выход close[t+10]*(1-slip),
минус комиссии taker x2. Стопы НЕ моделируем: по измерению MAE задевают
0.27% сделок — на сравнение групп не влияют.

ГРУППЫ (ровно боевые 24 признака):
  G-mom   моментумы: ret_1, ret_4, ret_10, ret_20, h4_momentum
  G-vol   волатильность: atr_pct, std_20, skew
  G-flow  объём/поток: vol_ratio, taker_buy, clv
  G-tech  техника: zscore, rsi, sma_cross
  G-btc   биткоин-контекст: btc_ret_10, btc_ret_20, btc_corr
  G-cross кросс (победитель шага-2): rel_ret_20, rel_ret_60, beta_20, corr_20
  G-cal   календарь+идентификация: hour, dow, sym_id

ПРАВИЛА ЧТЕНИЯ (записаны до прогона; A = avg лонга контроля):
  БАЛАСТ    : без группы avg >= A - 0.01  (группа не нужна)
  ВРЕДНА    : без группы avg >  A + 0.02  (группа вредит — выкинуть)
  ВКЛАД     : без группы avg <  A - 0.03  (группа реально работает)
  иначе     : НЕОДНОЗНАЧНО
Вторичные сигналы: навык (топ-минус-боттом квинтиль уверенности), шорт
avg, поквартальная стабильность. Ничего не деплоим: цель — отбор групп
для следующего ПОЛНОГО экзамена (walk-forward по 8 кварталам).
"""
import glob, json, time, zipfile, io, zlib
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.55
COOLDOWN_MS = 10 * 1800_000
SLIP, FEE2 = 0.0005, 0.001
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
SPLIT_Q = "2025Q1"          # тест = этот квартал и позже

GROUPS = {
    "G-mom":   ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum"],
    "G-vol":   ["atr_pct", "std_20", "skew"],
    "G-flow":  ["vol_ratio", "taker_buy", "clv"],
    "G-tech":  ["zscore", "rsi", "sma_cross"],
    "G-btc":   ["btc_ret_10", "btc_ret_20", "btc_corr"],
    "G-cross": ["rel_ret_20", "rel_ret_60", "beta_20", "corr_20"],
    "G-cal":   ["hour", "dow", "sym_id"],
}

def sym_id(sym): return np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
print("panel.zip:", zips[:1], flush=True)
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
    X["fi"] = np.uint16(i)
    X["sym_id"] = sym_id(sym)
    fut_close = np.full(len(df), np.nan, dtype="float32")
    fut_close[:-H] = close.values[H:]
    X["y"] = np.where(np.isnan(fut_close), np.nan,
                      (fut_close / close.values > 1).astype("float32"))
    X["fwd"] = (fut_close / close.values - 1).astype("float32")
    X["close"] = close.values
    X["ot"] = df["open_time"].values
    X = X.replace([np.inf, -np.inf], np.nan)
    kept = X.dropna(subset=[c for c in X.columns if c not in ("fwd", "close")])
    parts.append(kept)
    if i % 100 == 0:
        print(f"признаки {i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(parts, ignore_index=True)
del parts
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
feats_all = (["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
              "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
              "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
              "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
              "hour", "dow", "sym_id"])
assert set(sum(GROUPS.values(), [])) == set(feats_all), "группы != 24 признакам"
trd = data[data["quarter"] < SPLIT_Q]
ted = data[data["quarter"] >= SPLIT_Q]
print(f"строк всего {len(data)}, train {len(trd)}, test {len(ted)}, "
      f"тест {SPLIT_Q}..{sorted(data['quarter'].unique())[-1]} | "
      f"{time.time()-t0:.0f}с", flush=True)

def run_config(name, feats, save_trades=False):
    m = lgb.LGBMClassifier(**PARAMS)
    cat = ["sym_id"] if "sym_id" in feats else None
    if cat:
        m.fit(trd[feats], trd["y"], categorical_feature=cat)
    else:
        m.fit(trd[feats], trd["y"])
    p = m.predict_proba(ted[feats])[:, 1]
    d = ted.assign(p=p)
    out = {"name": name, "feats_n": len(feats)}
    # ЛОНГИ (fwd = будущая доходность уже в строке; merge не нужен — у
    # хешей sym_id бывают коллизии, слияние по ним размножает строки)
    dl = d[d["p"] > GATE].sort_values(["sym_id", "ot"])
    kept_l, last = [], {}
    for row in dl.itertuples():
        if row.ot - last.get(row.sym_id, -10**18) >= COOLDOWN_MS:
            kept_l.append((row.ot, row.sym_id, row.close, row.p,
                           row.quarter, row.fwd))
            last[row.sym_id] = row.ot
    if kept_l:
        L = pd.DataFrame(kept_l, columns=["ot", "sym_id", "close", "p",
                                          "quarter", "fwd"])
        L = L[~np.isnan(L["fwd"].values)].reset_index(drop=True)
        entry = L["close"].values * (1 + SLIP)
        exit_ = L["close"].values * (1 + L["fwd"].values) * (1 - SLIP)
        L["pnl"] = exit_ / entry - 1 - FEE2
        out["long_n"] = int(len(L))
        out["long_avg"] = round(float(L["pnl"].mean() * 100), 4)
        out["long_wr"] = round(float((L["pnl"] > 0).mean() * 100), 1)
        qq = L.groupby("quarter")["pnl"].mean() * 100
        out["long_by_q"] = {k: round(v, 3) for k, v in qq.items()}
        L2 = L.sort_values("p")
        n5 = max(1, len(L2) // 5)
        out["long_skill"] = round(float(L2["pnl"].tail(n5).mean() * 100
                                         - L2["pnl"].head(n5).mean() * 100), 3)
        if save_trades:
            L[["ot", "sym_id", "p", "pnl", "quarter"]].to_csv(
                "/kaggle/working/control_long_trades.csv", index=False)
    # ШОРТЫ
    ds = d[d["p"] < 1 - GATE].sort_values(["sym_id", "ot"])
    kept_s, last = [], {}
    for row in ds.itertuples():
        if row.ot - last.get(row.sym_id, -10**18) >= COOLDOWN_MS:
            kept_s.append((row.ot, row.sym_id, row.close, row.p, row.fwd))
            last[row.sym_id] = row.ot
    if kept_s:
        S = pd.DataFrame(kept_s, columns=["ot", "sym_id", "close", "p", "fwd"])
        S = S[~np.isnan(S["fwd"].values)].reset_index(drop=True)
        entry = S["close"].values * (1 - SLIP)
        exit_ = S["close"].values * (1 + S["fwd"].values) * (1 + SLIP)
        S["pnl"] = (entry - exit_) / entry - FEE2
        out["short_n"] = int(len(S))
        out["short_avg"] = round(float(S["pnl"].mean() * 100), 4)
        S2 = S.sort_values("p")
        n5 = max(1, len(S2) // 5)
        out["short_skill"] = round(float(S2["pnl"].head(n5).mean() * 100
                                          - S2["pnl"].tail(n5).mean() * 100), 3)
    imp = sorted(zip(feats, m.booster_.feature_importance("gain")),
                 key=lambda x: -x[1])[:12]
    out["gain_top12"] = [[f, round(float(v), 1)] for f, v in imp]
    print(f"[{name}] лонг {out.get('long_n')} шт avg {out.get('long_avg')}% "
          f"навык {out.get('long_skill')} | шорт {out.get('short_n')} шт "
          f"avg {out.get('short_avg')}% | {time.time()-t0:.0f}с", flush=True)
    return out

report = {"pre_reg": "сплит <2025Q1 / >=2025Q1; пороги: балласт>=A-0.01, "
                     "вредна>A+0.02, вклад<A-0.03",
          "split": SPLIT_Q, "configs": {}, "verdict_rules": "см. pre_reg"}
res = run_config("control", feats_all, save_trades=True)
report["configs"]["control"] = res
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
for gname, gcols in GROUPS.items():
    feats = [f for f in feats_all if f not in gcols]
    r = run_config(f"drop_{gname}", feats)
    A = report["configs"]["control"]["long_avg"]
    a = r["long_avg"]
    r["reading"] = ("БАЛАСТ" if a >= A - 0.01 else
                    "ВРЕДНА" if a > A + 0.02 else
                    "ВКЛАД" if a < A - 0.03 else "НЕОДНОЗНАЧНО")
    print(f"   -> {gname}: {r['reading']}", flush=True)
    report["configs"][f"drop_{gname}"] = r
    json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("вскрытие завершено:", json.dumps(
    {k.replace("drop_", ""): v["reading"]
     for k, v in report["configs"].items() if k != "control"}), flush=True)
