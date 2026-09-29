"""Кегл-кернел: обучение модели стакана на данных с сервера.
Вход: датасет sewerted/usdc-lob-features (parquet-архивы строк стакана).
Выход: lob.txt (модель) + lob_report.json (walk-forward по дням).
Правило честности: >= 4 дней данных, >= 3 тестовых дней, порог деплоя
записан в отчёт — решение принимает серверная обвязка по цифрам."""
import glob, json, os
import numpy as np, pandas as pd, lightgbm as lgb

HORIZON_S = 300
LAG_TOL_MS = 2500
SEED = 42
MIN_TEST_DAYS = 3
FEATS = ["spread_bp", "microprice_rel", "imb5", "imb10", "imb20",
         "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
         "ntr10", "vpin10", "d30", "d120"]

frames = []
for f in sorted(glob.glob("/kaggle/input/usdc-lob-features/**/*.parquet",
                          recursive=True)):
    try:
        frames.append(pd.read_parquet(f))
    except Exception:
        print("битый файл пропущен:", f, flush=True)
df = pd.concat(frames, ignore_index=True).drop_duplicates(["ts", "symbol"]) \
       .sort_values(["symbol", "ts"]).reset_index(drop=True)
print(f"строк {len(df)}, символов {df['symbol'].nunique()}", flush=True)

parts = []
for sym, g in df.groupby("symbol"):
    g = g.sort_values("ts").reset_index(drop=True)
    ts, mid = g["ts"].values, g["mid"].values
    idx = np.clip(ts.searchsorted(ts + HORIZON_S * 1000), 0, len(g) - 1)
    ok = ts[idx] >= ts + HORIZON_S * 1000 - 1500
    g["mid_fut"] = np.where(ok, mid[idx], np.nan)
    g["y"] = (g["mid_fut"] / mid > 1).astype(float)
    g.loc[g["mid_fut"].isna(), "y"] = np.nan
    for lag_ms, name in ((30_000, "d30"), (120_000, "d120")):
        j = np.clip(ts.searchsorted(ts - lag_ms), 0, len(g) - 1)
        past = np.where(np.abs(ts[j] - (ts - lag_ms)) <= LAG_TOL_MS,
                        mid[j], np.nan)
        g[name] = mid / past - 1
    g["microprice_rel"] = (g["microprice"] / g["mid"] - 1) * 10000
    for c in ("flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell", "ntr10"):
        med = g[c].median()
        g[c] = g[c] / (med if med and med > 0 else 1.0)
    parts.append(g)
ds = pd.concat(parts, ignore_index=True).dropna(subset=FEATS + ["y"])
ds["day"] = pd.to_datetime(ds["ts"], unit="ms").dt.date
days = sorted(ds["day"].unique())
print(f"дней {len(days)}: {days[0]}..{days[-1]}, строк {len(ds)}", flush=True)

report = {"days": str(len(days)), "ok": False}
if len(days) < MIN_TEST_DAYS + 1:
    report["reason"] = f"мало дней ({len(days)}) — нужно >= {MIN_TEST_DAYS+1}"
    json.dump(report, open("/kaggle/working/lob_report.json", "w"))
else:
    test_days = days[-MIN_TEST_DAYS:]
    rows = []
    for day in test_days:
        tr, te = ds[ds["day"] < day], ds[ds["day"] == day]
        if len(te) < 5000 or len(tr) < 20000:
            continue
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=4,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=SEED, n_jobs=4, verbosity=-1)
        m.fit(tr[FEATS], tr["y"])
        p = m.predict_proba(te[FEATS])[:, 1]
        fwd = te["mid_fut"] / te["mid"] - 1
        q75, q25 = np.quantile(p, 0.75), np.quantile(p, 0.25)
        spread = (fwd[p >= q75].mean() - fwd[p <= q25].mean()) * 10000
        acc = ((p > 0.5) == (te["y"] == 1)).mean() * 100
        rows.append({"day": str(day), "acc": round(acc, 2),
                     "spread_bp": round(spread, 2)})
        print(f"{day}: acc {acc:.1f}%, спред {spread:+.1f} бп", flush=True)
    ok_days = sum(1 for r in rows if r["spread_bp"] > 0)
    report.update({"test": rows, "plus_days": ok_days, "n_test": len(rows),
                   "ok": ok_days >= max(2, (len(rows) + 1) // 2)})
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=4,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=SEED, n_jobs=4, verbosity=-1)
    m.fit(ds[FEATS], ds["y"])
    m.booster_.save_model("/kaggle/working/lob.txt")
    report["rows"] = int(len(ds))
    json.dump(report, open("/kaggle/working/lob_report.json", "w"), indent=1)
    print("модель сохранена; ок дней:", ok_days, "/", len(rows), flush=True)
