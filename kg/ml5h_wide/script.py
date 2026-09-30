"""ИССЛЕДОВАНИЕ (предрегистрация 29.09): шорты аналитика на тех же данных.
Модель БЕЗ ИЗМЕНЕНИЙ (ценовые ярлыки, как у боевой). Симуляция сделок
в обе стороны: лонг p>0.55, шорт p<0.45. Исполнение честное:
вход ±slip, аварийный стоп ±15% (для шорта нырок = ВЗЛЁТ: max high),
выход close[t+10], комиссии taker×2. Отдельно лонги/шорты + вместе.
Чистое измерение — деплойных решений в кернеле нет."""
import glob, json, zlib
import numpy as np, pandas as pd, lightgbm as lgb

H = 10          # горизонт 10 баров 30м = 5 часов
GATE = 0.55
COOLDOWN_MS = 10 * 1800_000
SLIP, FEE2 = 0.0005, 0.001
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")

def sym_id(sym): return np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)

import os, zipfile, io
zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
print("panel.zip найден:", zips[:2], flush=True)
if not zips:
    for root, dirs, fs in os.walk("/kaggle/input"):
        print(" ", root, len(fs), "файлов", flush=True)
    json.dump({"verdict": False, "error": "panel.zip not found",
               "input_tree": str(os.listdir("/kaggle/input"))},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
btc_pre = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
btc_pre["open_time"] = btc_pre["open_time"].astype(np.int64)
btc_pre.index = pd.to_datetime(btc_pre["open_time"], unit="ms")
btc = btc_pre["close"].astype("float32")
print("монет в панели:", len(members), flush=True)
if len(members) < 500:
    json.dump({"verdict": False, "error": f"panel members {len(members)}"},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
parts = []
for i, name in enumerate(members, 1):
    sym = name.replace("_30m.csv", "")
    df = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["open_time", "high", "low", "close",
                              "volume", "quote_volume", "taker_buy_base"])
    for c in df.columns:
        if c != "open_time":          # 13-значные метки не влезают в float32!
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
        bret = b.pct_change()
        X["btc_ret_10"] = bret.rolling(10).sum().astype("float32") * 0 + b.pct_change(10).astype("float32")
        X["btc_ret_20"] = b.pct_change(20).astype("float32")
        X["btc_corr"] = rets.rolling(50, min_periods=20).corr(bret).astype("float32")
    X["h4_momentum"] = close.pct_change(48).astype("float32")
    tsdt = df.index
    X["hour"] = tsdt.hour.astype("float32")
    X["dow"] = tsdt.dayofweek.astype("float32")
    # G-cross
    if sym != "BTCUSDT" and btc is not None:
        bret = btc.reindex(df.index).ffill().pct_change()
        X["rel_ret_20"] = (rets.rolling(20).sum() - bret.rolling(20).sum()).astype("float32")
        X["rel_ret_60"] = (rets.rolling(60).sum() - bret.rolling(60).sum()).astype("float32")
        cov = rets.rolling(480).cov(bret); var = bret.rolling(480).var()
        X["beta_20"] = (cov / var.replace(0, np.nan)).astype("float32")
        X["corr_20"] = rets.rolling(480, min_periods=100).corr(bret).astype("float32")
    else:
        for c in ("rel_ret_20", "rel_ret_60", "beta_20", "corr_20"):
            X[c] = np.float32(0)
    X["high"] = df["high"].values.astype("float32")
    X["low"] = df["low"].values.astype("float32")
    X["fi"] = np.uint16(i)
    X["sym_id"] = sym_id(sym)
    # реализованная сделка: вход close×(1+slip), стоп -15%, выход close[t+10]
    lows = df["low"].values.astype("float32")
    entry = close.values * (1 + SLIP)
    stop_px = entry * (1 - 0.15)
    fut_low = np.full(len(df), np.inf, dtype="float32")
    for k in range(1, H + 1):
        fut_low[:-k] = np.minimum(fut_low[:-k], lows[k:])
    fut_close = np.full(len(df), np.nan, dtype="float32")
    fut_close[:-H] = close.values[H:]
    stop_hit = fut_low <= stop_px
    exit_p = np.where(stop_hit, stop_px * (1 - SLIP),
                      fut_close * (1 - SLIP))
    pnl_lbl = exit_p / entry - 1 - FEE2
    X["y"] = np.where(np.isnan(fut_close), np.nan,
                      (fut_close / close.values > 1).astype("float32"))
    X["fwd"] = (fut_close / close.values - 1).astype("float32")
    X["close"] = close.values
    X["ot"] = df["open_time"].values
    X = X.replace([np.inf, -np.inf], np.nan)
    kept = X.dropna(subset=[c for c in X.columns if c not in ("fwd", "close")])
    if i <= 3:
        nans = X.isna().sum()
        bad = nans[nans > 0].sort_values(ascending=False)
        print(f"ДИАГНОСТИКА {sym}: было {len(X)}, осталось {len(kept)} "
              f"({len(kept)/max(1,len(X))*100:.1f}%), NaN по колонкам: "
              f"{dict(bad.head(6))}", flush=True)
    parts.append(kept)
    if i % 100 == 0:
        print(f"признаки {i}/{len(members)}", flush=True)
data = pd.concat(parts, ignore_index=True)
del parts
print("строк:", len(data), flush=True)
feats = [c for c in data.columns if c not in ("y", "fwd", "close", "ot", "low", "high", "fi")]
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
# ПРЕДРЕГИСТРАЦИЯ v3 (до просмотра результата): 8 кварталов, индексы
# np.linspace по ВСЕЙ истории 2021-2026 — медведь 2021-2022 включён.
import numpy as _np
idx = _np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
nan_counts = data.isna().sum()
print("NaN по колонкам (только ненулевые):", flush=True)
for c in data.columns:
    if nan_counts[c] > 0:
        print(f"  {c}: {nan_counts[c]}", flush=True)
print("тестовые кварталы:", test_q, flush=True)
all_trades, folds, nulls = [], [], []
for k, qq in enumerate(test_q):
    tr = data[data["quarter"] < qq]
    te = data[data["quarter"] == qq]
    if len(te) < 5000 or len(tr) < 200_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(tr[feats], tr["y"], categorical_feature=["sym_id"])
    p = m.predict_proba(te[feats])[:, 1]
    d = te.assign(p=p)

    # --- ЛОНГИ: p > 0.55, нырок = min low, стоп -15% ---
    dl = d[d["p"] > GATE].sort_values(["sym_id", "ot"])
    kept_l, last_l = [], {}
    for row in dl.itertuples():
        if row.ot - last_l.get(row.sym_id, -10**18) >= COOLDOWN_MS:
            kept_l.append(row); last_l[row.sym_id] = row.ot
    L = pd.DataFrame(kept_l)
    if len(L):
        L = L.reset_index(drop=True)
        entry = L["close"].values * (1 + SLIP)
        stop_px = entry * (1 - 0.15)
        # худший нырок за холд
        fut_low = np.full(len(L), np.inf)
        for kk in range(1, H + 1):
            shift = L["ot"].values
        # посчитаем по пулу котировок символа (нужны low после входа)
        mae_l = np.full(len(L), np.nan)
        exit_l = np.full(len(L), np.nan)
        pools = {fi: g for fi, g in te.groupby("fi")}
        for fi, g in L.groupby("fi"):
            pool = pools.get(fi)
            if pool is None: continue
            ots = pool["ot"].values
            lows = pool["low"].values
            closes10 = pool["close"].values
            for idx, row in g.iterrows():
                j = int(np.searchsorted(ots, row["ot"]))
                seg = lows[j+1: j+1+H]
                fut_c = closes10[j+H] if j+H < len(closes10) else np.nan
                if len(seg):
                    mae_l[idx] = seg.min() / entry[idx] - 1
                    if seg.min() <= stop_px[idx]:
                        exit_l[idx] = stop_px[idx] * (1 - SLIP)
                    else:
                        exit_l[idx] = fut_c * (1 - SLIP) if not np.isnan(fut_c) else np.nan
        ok = ~np.isnan(exit_l)
        L = L[ok].reset_index(drop=True)
        L["pnl"] = exit_l[ok] / entry[ok] - 1 - FEE2
        L["side"] = "LONG"
        L["mae"] = mae_l[ok]
    else:
        L = pd.DataFrame(columns=["pnl", "side", "mae", "p", "quarter"])
        L["quarter"] = None

    # --- ШОРТЫ: p < 0.45, "нырок" = ВЗЛЁТ max high, стоп +15% ---
    ds = d[d["p"] < 1 - GATE].sort_values(["sym_id", "ot"])
    kept_s, last_s = [], {}
    for row in ds.itertuples():
        if row.ot - last_s.get(row.sym_id, -10**18) >= COOLDOWN_MS:
            kept_s.append(row); last_s[row.sym_id] = row.ot
    S = pd.DataFrame(kept_s)
    if len(S):
        S = S.reset_index(drop=True)
        entry_s = S["close"].values * (1 - SLIP)
        stop_px_s = entry_s * (1 + 0.15)
        mae_s = np.full(len(S), np.nan)
        exit_s = np.full(len(S), np.nan)
        for fi, g in S.groupby("fi"):
            pool = pools.get(fi)
            if pool is None: continue
            ots = pool["ot"].values
            highs = pool["high"].values
            closes10 = pool["close"].values
            for idx, row in g.iterrows():
                j = int(np.searchsorted(ots, row["ot"]))
                seg = highs[j+1: j+1+H]
                fut_c = closes10[j+H] if j+H < len(closes10) else np.nan
                if len(seg):
                    mae_s[idx] = seg.max() / entry_s[idx] - 1
                    if seg.max() >= stop_px_s[idx]:
                        exit_s[idx] = stop_px_s[idx] * (1 + SLIP)
                    else:
                        exit_s[idx] = fut_c * (1 + SLIP) if not np.isnan(fut_c) else np.nan
        ok = ~np.isnan(exit_s)
        S = S[ok].reset_index(drop=True)
        S["pnl"] = (entry_s[ok] - exit_s[ok]) / entry_s[ok] - FEE2
        S["side"] = "SHORT"
        S["mae"] = mae_s[ok]
    else:
        S = pd.DataFrame(columns=["pnl", "side", "mae", "p", "quarter"])
        S["quarter"] = None

    for dfx, nm in ((L, "LONG"), (S, "SHORT")):
        if len(dfx):
            dfx["quarter"] = qq

    def side_stats(dfw, nm):
        if dfw is None or len(dfw) == 0:
            return {"trades": 0, "avg": None, "skill": None}
        e = dfw["pnl"].mean() * 100
        dfw2 = dfw.sort_values("p")
        n5 = max(1, len(dfw2) // 5)
        if nm == "LONG":
            bot, top = dfw2["pnl"].head(n5).mean()*100, dfw2["pnl"].tail(n5).mean()*100
        else:  # для шортов уверенность = НИЗКИЙ p
            top, bot = dfw2["pnl"].head(n5).mean()*100, dfw2["pnl"].tail(n5).mean()*100
        return {"trades": len(dfw), "avg": round(e, 4),
                "skill": round(top - bot, 4)}

    ls, ss = side_stats(L, "LONG"), side_stats(S, "SHORT")
    print(f"фолд {qq}: ЛОНГ {ls['trades']} шт {ls['avg']}% (навык {ls['skill']}%) | "
          f"ШОРТ {ss['trades']} шт {ss['avg']}% (навык {ss['skill']}%)", flush=True)
    folds.append({"quarter": qq, "long": ls, "short": ss})
    if len(L): all_trades.append(L.assign(kind="L"))
    if len(S): all_trades.append(S.assign(kind="S"))

res = pd.concat([t for t in all_trades if len(t)], ignore_index=True)
res["quarter"] = test_res_q = res["quarter"]
L_all = res[res["kind"] == "L"]; S_all = res[res["kind"] == "S"]
base_l = round(float(L_all["pnl"].mean() * 100), 4) if len(L_all) else None
base_s = round(float(S_all["pnl"].mean() * 100), 4) if len(S_all) else None
base_c = round(float(res["pnl"].mean() * 100), 4) if len(res) else None
skill_l = round(float(np.mean([f["long"]["skill"] for f in folds if f["long"]["skill"] is not None])), 3) if any(f["long"]["skill"] is not None for f in folds) else None
skill_s = round(float(np.mean([f["short"]["skill"] for f in folds if f["short"]["skill"] is not None])), 3) if any(f["short"]["skill"] is not None for f in folds) else None
print(f"ИТОГ ЛОНГ: {base_l}% | навык {skill_l}", flush=True)
print(f"ИТОГ ШОРТ: {base_s}% | навык {skill_s}", flush=True)
print(f"ИТОГ ВМЕСТЕ: {base_c}%", flush=True)
json.dump({"long_avg": base_l, "long_skill": skill_l,
           "short_avg": base_s, "short_skill": skill_s,
           "combined_avg": base_c, "folds": folds},
          open("/kaggle/working/report.json", "w"), indent=1)
res.to_csv("/kaggle/working/all_trades.csv", index=False)
print("измерение завершено", flush=True)

# ФИНАЛЬНАЯ МОДЕЛЬ (предрегистрация 30.09): та же формула что у боевой
# (ценовые ярлыки), но обучение на ПОЛНЫХ данных (20.5М строк — после
# фикса float32-каста меток). Критерии деплоя те же: навык (топ-квинтиль
# минус боттом по уверенности) > 0 минимум в 6/8 кварталов И средний
# >= +0.20%. Иначе модель не сохраняется как деплой-кандидат.
skill_pos = sum(1 for f in folds
                if f["long"]["skill"] is not None and f["long"]["skill"] > 0)
verdict = bool(skill_pos >= 6 and (skill_l or 0) >= 0.20)
final = lgb.LGBMClassifier(**PARAMS)
final.fit(data[feats], data["y"], categorical_feature=["sym_id"])
final.booster_.save_model("/kaggle/working/ml5h.txt")
syms = sorted({m.split("/")[-1].replace("_30m.csv", "") for m in members})
json.dump({"features": feats, "symbols": syms, "gate": 0.55,
           "hold_bars": H, "bar_minutes": 30,
           "universe": "wide-full-data", "train_rows": int(len(data)),
           "verdict": verdict, "long_avg": base_l, "long_skill": skill_l,
           "folds": folds},
          open("/kaggle/working/ml5h_meta.json", "w"), indent=1)
# вердикт ОБЯЗАТЕЛЬНО и в report.json — его читает синхронизатор на сервере
json.dump({"verdict": verdict, "avg": base_l, "skill": skill_l,
           "skill_pos_quarters": skill_pos,
           "folds": folds},
          open("/kaggle/working/report.json", "w"), indent=1)
print("финальная модель сохранена | вердикт:", verdict, flush=True)
