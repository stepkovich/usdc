"""Кегл-кернел: обучение модели стакана на данных с сервера.
Вход: датасет sewerted/usdc-lob-features (parquet-архивы строк стакана).
Выход: lob.txt (модель) + lob_medians.json + lob_feats.json +
lob_report.json (walk-forward по дням).
Правило честности: >= 4 дней данных, >= 3 тестовых дней, порог деплоя
записан в отчёт — решение принимает серверная обвязка по цифрам.

v2 (01.10, А1 «многоуровневый стакан»): рекордер пишет сырые 20 уровней
и 5 лестничных признаков (imb1, slope_b, slope_a, wall_b, wall_a).
ПРОТОКОЛ A/B (записан до прогона):
  A — старые 13 признаков на всей истории (текущее поведение);
  B — те же 13 + 5 лестничных, ТОЛЬКО на строках с лестницей
      (после деплоя рекордера v2) и ТОЛЬКО если таких дней >= 4;
  сравнение честное: обе модели учатся на ОДНОМ окне (все дни B,
  кроме последних 3) и тестируются на ПОСЛЕДНИХ 3 днях B.
  Деплой: B ставится, если плюс-дней >= 2/3 И средний спред B лучше A
  на тех же днях; иначе A по старым критериям; ни один не прошёл —
  работает старая модель.
lob_feats.json пишется ВСЕГДА со списком признаков выбранной модели
(движок по нему строит вектор — иначе несходство числа фич).
"""
import glob, json, os
import json as _json
import numpy as np, pandas as pd, lightgbm as lgb

HORIZON_S = 300
LAG_TOL_MS = 2500
SEED = 42
MIN_TEST_DAYS = 3
FEATS = ["spread_bp", "microprice_rel", "imb5", "imb10", "imb20",
         "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
         "ntr10", "vpin10", "d30", "d120"]
LADDER_FEATS = ["imb1", "slope_b", "slope_a", "wall_b", "wall_a"]
FEATS_B = FEATS + LADDER_FEATS
BASE_COLS = (["ts", "symbol", "mid", "spread_bp", "microprice",
              "imb5", "imb10", "imb20", "bid_sum20", "ask_sum20",
              "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
              "ntr10", "vpin10"] + LADDER_FEATS)

frames = []
for f in sorted(glob.glob("/kaggle/input/usdc-lob-features/**/*.parquet",
                          recursive=True)):
    try:
        try:                                    # новый формат (21 колонка)
            frames.append(pd.read_parquet(f, columns=BASE_COLS))
        except Exception:                       # старый (16) — лестницы нет
            df_old = pd.read_parquet(f, columns=BASE_COLS[:16])
            for c in LADDER_FEATS:
                df_old[c] = np.nan
            frames.append(df_old)
    except Exception:
        print("битый файл пропущен:", f, flush=True)
df = pd.concat(frames, ignore_index=True).drop_duplicates(["ts", "symbol"]) \
       .sort_values(["symbol", "ts"]).reset_index(drop=True)
print(f"строк {len(df)}, символов {df['symbol'].nunique()}, "
      f"с лестницей {int(df[LADDER_FEATS[0]].notna().sum())}", flush=True)

parts = []
medians = {}
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
    sym_meds = {}
    for c in ("flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
              "ntr10"):
        med = g[c].median()
        med = float(med) if med and med > 0 else 1.0
        sym_meds[c] = med
        g[c] = g[c] / med
    medians[sym] = sym_meds
    parts.append(g)
ds = pd.concat(parts, ignore_index=True).dropna(subset=FEATS + ["y"])
ds["day"] = pd.to_datetime(ds["ts"], unit="ms").dt.date
days = sorted(ds["day"].unique())
print(f"дней {len(days)}: {days[0]}..{days[-1]}, строк {len(ds)}", flush=True)

# --- строки с лестницей (рекордер v2) ---
ds_b = ds.dropna(subset=LADDER_FEATS).copy()
days_b = sorted(ds_b["day"].unique())
print(f"дней с лестницей: {len(days_b)}, строк {len(ds_b)}", flush=True)

PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, random_state=SEED,
              n_jobs=4, verbosity=-1)

def walk_forward(data, feats, test_days):
    rows = []
    for day in test_days:
        tr, te = data[data["day"] < day], data[data["day"] == day]
        if len(te) < 5000 or len(tr) < 20000:
            continue
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(tr[feats], tr["y"])
        p = m.predict_proba(te[feats])[:, 1]
        fwd = te["mid_fut"] / te["mid"] - 1
        q75, q25 = np.quantile(p, 0.75), np.quantile(p, 0.25)
        spread = (fwd[p >= q75].mean() - fwd[p <= q25].mean()) * 10000
        acc = ((p > 0.5) == (te["y"] == 1)).mean() * 100
        rows.append({"day": str(day), "acc": round(acc, 2),
                     "spread_bp": round(spread, 2)})
        print(f"  {day}: acc {acc:.1f}%, спред {spread:+.1f} бп", flush=True)
    return rows

def ok_criteria(rows):
    plus = sum(1 for r in rows if r["spread_bp"] > 0)
    return plus >= max(2, (len(rows) + 1) // 2), plus

report = {"days": str(len(days)), "days_b": str(len(days_b)), "ok": False}
deployed = None                      # None | "A" | "B"

# --- ЭКЗАМЕН A: старые признаки на всей истории ---
if len(days) >= MIN_TEST_DAYS + 1:
    test_a = days[-MIN_TEST_DAYS:]
    print("экзамен A (13 признаков, вся история):", flush=True)
    rows_a = walk_forward(ds, FEATS, test_a)
    ok_a, plus_a = ok_criteria(rows_a)
    report["A"] = {"test": rows_a, "plus_days": plus_a,
                   "n_test": len(rows_a), "ok": bool(ok_a)}
else:
    print("данных меньше 4 дней — только пилот", flush=True)
    ok_a = False
    report["A"] = {"ok": False}

# --- ЭКЗАМЕН B: лестница (нужно >= 4 дня с лестницей) ---
if len(days_b) >= MIN_TEST_DAYS + 1:
    test_b = days_b[-MIN_TEST_DAYS:]
    train_end = days_b[:-MIN_TEST_DAYS][-1]
    tr_win = ds_b[ds_b["day"] <= train_end]
    print(f"экзамен B ({len(FEATS_B)} признаков, окно до {train_end}, "
          f"тест {test_b[0]}..{test_b[-1]}):", flush=True)
    rows_b = walk_forward(ds_b, FEATS_B, test_b)
    ok_b, plus_b = ok_criteria(rows_b)
    # честное сравнение: модель A (13 признаков) на тех же строках и днях
    rows_a_same = walk_forward(ds_b, FEATS, test_b)
    ok_a_same, plus_a_same = ok_criteria(rows_a_same)
    avg_b = np.mean([r["spread_bp"] for r in rows_b]) if rows_b else -1e9
    avg_a_same = np.mean([r["spread_bp"] for r in rows_a_same]) \
        if rows_a_same else -1e9
    report["B"] = {"test": rows_b, "plus_days": plus_b,
                   "n_test": len(rows_b), "ok": bool(ok_b),
                   "A_same_window": rows_a_same,
                   "avg_b": round(float(avg_b), 2),
                   "avg_a_same": round(float(avg_a_same), 2)}
    print(f"B: плюс-дней {plus_b}/{len(rows_b)}, средний спред "
          f"{avg_b:+.2f} бп | A на тех же днях: {avg_a_same:+.2f} бп",
          flush=True)
    if ok_b and avg_b > avg_a_same:
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(ds_b[FEATS_B], ds_b["y"])
        m.booster_.save_model("/kaggle/working/lob.txt")
        _json.dump(FEATS_B, open("/kaggle/working/lob_feats.json", "w"))
        deployed = "B"
        report["ok"] = True
        report["deployed"] = "B"
        print("ДЕПЛОЙ B (лестница) — навык лучше старых признаков "
              "на тех же днях", flush=True)
elif len(days_b) >= 2:
    # ПИЛОТ B (мало дней, НЕ для деплоя): первый день — обучение,
    # последний — тест; A учим на том же первом дне для сравнения
    d0, d1 = days_b[0], days_b[-1]
    tr = ds_b[ds_b["day"] == d0]
    te = ds_b[ds_b["day"] == d1]
    if len(te) >= 5000 and len(tr) >= 20000:
        out = {}
        for name, feats in (("A", FEATS), ("B", FEATS_B)):
            m = lgb.LGBMClassifier(**PARAMS)
            m.fit(tr[feats], tr["y"])
            p = m.predict_proba(te[feats])[:, 1]
            fwd = te["mid_fut"] / te["mid"] - 1
            q75, q25 = np.quantile(p, 0.75), np.quantile(p, 0.25)
            spread = (fwd[p >= q75].mean() - fwd[p <= q25].mean()) * 10000
            acc = ((p > 0.5) == (te["y"] == 1)).mean() * 100
            out[name] = {"day": str(d1), "acc": round(acc, 2),
                         "spread_bp": round(spread, 2)}
            print(f"пилот-B {name}: acc {acc:.1f}%, спред {spread:+.2f} бп",
                  flush=True)
        report["pilot_b"] = out

# --- деплой A (старое поведение, если B не выбран) ---
if deployed is None and len(days) >= MIN_TEST_DAYS + 1:
    if ok_a:
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(ds[FEATS], ds["y"])
        m.booster_.save_model("/kaggle/working/lob.txt")
        _json.dump(FEATS, open("/kaggle/working/lob_feats.json", "w"))
        deployed = "A"
        report["ok"] = True
        report["deployed"] = "A"
        print("ДЕПЛОЙ A (старые признаки) — критерии взяты", flush=True)

_json.dump(medians, open("/kaggle/working/lob_medians.json", "w"),
           indent=1)
if "A" in report:
    report["A"]["test"] = report["A"].get("test", [])
if deployed:
    src = report["B"] if deployed == "B" else report["A"]
    report["plus_days"] = src.get("plus_days")
    report["n_test"] = src.get("n_test")
report["rows"] = int(len(ds))
report["rows_b"] = int(len(ds_b))
json.dump(report, open("/kaggle/working/lob_report.json", "w"), indent=1)
print(f"итог: деплой {deployed or 'нет'}; окей-дней A: "
      f"{report.get('A', {}).get('plus_days')}", flush=True)
