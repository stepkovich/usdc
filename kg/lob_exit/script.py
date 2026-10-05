"""А2 «АДАПТИВНЫЙ ВЫХОД СПРИНТЕРА» (предрегистрация 05.10, до запуска).

ПОВОД: сейчас сделка закрывается по будильнику — ровно через 300с, что
бы ни происходило. Если сигнал умер на 60-й секунде — мы держим убыточную
позицию ещё 4 минуты; если сигнал крепчает — выбрасываем прибыль.
ГИПОТЕЗА: выход, следящий за сигналом, добавляет кромку.

ДИЗАЙН: модель (те же 13 признаков, те же параметры) учится на окне до
тестовых дней; на тестовых днях предсказание p для КАЖДОЙ 500мс-строки.
Симуляция сделок по символам: лонг p>0.62, шорт p<0.38; вход по mid;
позиция одна на символ; после закрытия пауза 300с. Издержки ОДИНАКОВЫЕ
в обоих сценариях (вход мейкер 2бп, выход тейкер 10бп) — сравниваем
ТОЛЬКО момент выхода.

СЦЕНАРИИ (записаны до прогона, без перебора):
  БАЗА: выход ровно через 300с.
  АДАПТ: каждые 30с после входа смотрим p строки: если |сигнал| умер
    (лонг: p<0.52; шорт: p>0.48) — выходим немедленно; если сигнал
    крепче порога входа весь путь — держим до 600с максимум; иначе
    обычный выход на 300с.
Оценка: средний PnL/сделку, всего сделок, плюс-дни. Вывод: адапт жив,
если улучшает среднее >= +1 бп на тех же сделках и не режет число сделок
более чем вдвое. Ничего не деплоится — решение за владельцем.
"""
import glob, json, os, re
import numpy as np, pandas as pd, lightgbm as lgb

HORIZON_S = 300
LAG_TOL_MS = 2500
SEED = 42
WINDOW_DAYS = 7
FEATS = ["spread_bp", "microprice_rel", "imb5", "imb10", "imb20",
         "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
         "ntr10", "vpin10", "d30", "d120"]
LADDER = ["imb1", "slope_b", "slope_a", "wall_b", "wall_a"]
BASE_COLS = (["ts", "symbol", "mid", "spread_bp", "microprice",
              "imb5", "imb10", "imb20", "bid_sum20", "ask_sum20",
              "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
              "ntr10", "vpin10"] + LADDER)
FLOAT_COLS = [c for c in BASE_COLS if c not in ("ts", "symbol")]
KEEP = (["ts", "symbol", "mid", "spread_bp", "imb5", "imb10", "imb20",
         "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
         "ntr10", "vpin10", "microprice_rel", "d30", "d120", "y"] + LADDER)
GATE_L, GATE_S = 0.62, 0.38
SLIDE_MS = 30_000            # шаг проверки сигнала в адаптиве
FEE_MAKER, FEE_TAKER = 0.0002, 0.001     # вход мейкер, выход тейкер

files = sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True))
print("всё, что смонтировано:", files[:3], "... всего", len(files), flush=True)
if not files:
    for root, dirs, fs in os.walk("/kaggle/input"):
        print(" ", root, len(fs), fs[:3], flush=True)
    json.dump({"error": "нет parquet во входе"},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
def _fday(p):
    m = re.search(r"feat_(\d{8})_", p)
    return m.group(1) if m else ""
days_all = sorted({_fday(f) for f in files if _fday(f)})
if len(days_all) > WINDOW_DAYS:
    cut = days_all[-WINDOW_DAYS]
    files = [f for f in files if _fday(f) >= cut]
print(f"файлов {len(files)}", flush=True)

frames = []
for f in files:
    try:
        try:
            df_f = pd.read_parquet(f, columns=BASE_COLS)
        except Exception:
            df_f = pd.read_parquet(f, columns=BASE_COLS[:16])
            for c in LADDER:
                df_f[c] = np.float32(np.nan)
        df_f[FLOAT_COLS] = df_f[FLOAT_COLS].astype("float32")
        frames.append(df_f)
    except Exception:
        print("битый файл:", f, flush=True)
df = pd.concat(frames, ignore_index=True).drop_duplicates(["ts", "symbol"])
del frames
df.sort_values(["symbol", "ts"], inplace=True)
df.reset_index(drop=True, inplace=True)

parts = []
medians = {}
for sym, g in df.groupby("symbol", observed=True):
    g = g.sort_values("ts").reset_index(drop=True)
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    idx = np.clip(ts.searchsorted(ts + HORIZON_S * 1000), 0, len(g) - 1)
    ok = ts[idx] >= ts + HORIZON_S * 1000 - 1500
    mid_fut = np.where(ok, mid[idx], np.nan)
    g["y"] = np.where(np.isnan(mid_fut), np.nan,
                      (mid_fut / mid > 1).astype("float32"))
    for lag_ms, name in ((30_000, "d30"), (120_000, "d120")):
        j = np.clip(ts.searchsorted(ts - lag_ms), 0, len(g) - 1)
        past = np.where(np.abs(ts[j] - (ts - lag_ms)) <= LAG_TOL_MS,
                        mid[j], np.nan)
        g[name] = (mid / past - 1).astype("float32")
    g["microprice_rel"] = (g["microprice"].astype("float64") / mid - 1) \
        .astype("float32")
    sym_meds = {}
    for c in ("flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
              "ntr10"):
        med = g[c].median()
        med = float(med) if med and med > 0 else 1.0
        sym_meds[c] = med
        g[c] = (g[c].astype("float64") / med).astype("float32")
    medians[sym] = sym_meds
    parts.append(g[KEEP])
    del g
del df
ds = pd.concat(parts, ignore_index=True)
del parts
ds = ds.dropna(subset=FEATS + ["y"]).reset_index(drop=True)
ds["day"] = pd.to_datetime(ds["ts"], unit="ms").dt.date
days = sorted(ds["day"].unique())
print(f"дней {len(days)}: {days[0]}..{days[-1]}, строк {len(ds)}", flush=True)

PARAMS = dict(n_estimators=150, learning_rate=0.05, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, random_state=SEED,
              n_jobs=4, verbosity=-1)

test_days = days[-3:]
train = ds[ds["day"] < test_days[0]]
m = lgb.LGBMClassifier(**PARAMS)
m.fit(train[FEATS], train["y"])
print("модель обучена на", len(train), "строках", flush=True)

# предсказание p на тестовых днях (батчами по символу для скорости)
te = ds[ds["day"].isin(test_days)].copy()
te["p"] = np.nan
for sym, g in te.groupby("symbol", observed=True):
    te.loc[g.index, "p"] = m.predict_proba(g[FEATS])[:, 1]
te = te.dropna(subset=["p"])
print(f"тест-дни {test_days}, строк с p: {len(te)}", flush=True)

def simulate(g, mode):
    """Один символ: сделки с cooldown; возвращает список pnl (в долях).
    mode: 'base' = выход через 300с; 'adapt' = следящий выход."""
    g = g.sort_values("ts").reset_index(drop=True)
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    p = g["p"].values
    day = g["day"].values
    trades = []
    last_close_ts = -10**18
    i = 0
    n = len(g)
    while i < n:
        side = 0
        if p[i] > GATE_L:
            side = 1
        elif p[i] < GATE_S:
            side = -1
        if side == 0 or ts[i] < last_close_ts + HORIZON_S * 1000:
            i += 1
            continue
        entry_ts = ts[i]
        entry_px = mid[i]
        # максимум 600с горизонта строк
        end_i = min(int(np.searchsorted(ts, entry_ts + 610_000)), n)
        if end_i - i < 10:            # нет данных вперёд
            i += 1
            continue
        exit_px = None
        if mode == "base":
            j = int(np.searchsorted(ts, entry_ts + HORIZON_S * 1000)) - 1
            exit_px = mid[j] if j > i else None
        else:
            t_check = entry_ts + SLIDE_MS
            while t_check <= entry_ts + 600_000:
                j = int(np.searchsorted(ts, t_check)) - 1
                if j <= i:
                    t_check += SLIDE_MS
                    continue
                pj = p[j]
                dead = (side == 1 and pj < 0.52) or \
                       (side == -1 and pj > 0.48)
                strong = (side == 1 and pj > GATE_L) or \
                         (side == -1 and pj < GATE_S)
                if dead or (t_check >= entry_ts + 300_000 and not strong):
                    exit_px = mid[j]
                    break
                t_check += SLIDE_MS
            if exit_px is None:
                j = int(np.searchsorted(ts, entry_ts + 600_000)) - 1
                exit_px = mid[j] if j > i else None
        if exit_px is None:
            i += 1
            continue
        if side == 1:
            pnl = exit_px / (entry_px * (1 + FEE_MAKER)) - 1 - FEE_TAKER
        else:
            pnl = 1 - exit_px / (entry_px * (1 - FEE_MAKER)) - FEE_TAKER
        trades.append({"pnl": pnl, "day": day[i], "hold_s":
                       int((ts[j] - entry_ts) / 1000)})
        last_close_ts = ts[j]
        i = int(np.searchsorted(ts, last_close_ts + HORIZON_S * 1000))
    return trades

res = {"base": [], "adapt": []}
for sym, g in te.groupby("symbol", observed=True):
    for mode in ("base", "adapt"):
        res[mode].extend(simulate(g, mode))

report = {"test_days": [str(d) for d in test_days], "scenarios": {}}
for mode in ("base", "adapt"):
    T = pd.DataFrame(res[mode])
    if not len(T):
        report["scenarios"][mode] = {"n": 0}
        continue
    avg = float(T["pnl"].mean() * 10000)
    by_day = {str(k): round(float(v["pnl"].mean() * 10000), 2)
              for k, v in T.groupby("day")}
    plus_days = sum(1 for v in by_day.values() if v > 0)
    hold = float(T["hold_s"].mean())
    report["scenarios"][mode] = {
        "n": int(len(T)), "avg_bp": round(avg, 2),
        "plus_days": plus_days, "by_day_bp": by_day,
        "avg_hold_s": round(hold)}
    print(f"{mode}: {len(T)} сделок, средняя {avg:+.2f} бп, "
          f"плюс-дней {plus_days}/{len(by_day)}, средний холд {hold:.0f}с",
          flush=True)

nb = report["scenarios"]["base"].get("n", 0)
na = report["scenarios"]["adapt"].get("n", 0)
ab = report["scenarios"]["base"].get("avg_bp", -99)
aa = report["scenarios"]["adapt"].get("avg_bp", -99)
verdict = (na > 0 and nb > 0 and aa - ab >= 1.0 and na >= nb * 0.5)
report["verdict"] = bool(verdict)
print("ВЕРДИКТ:", "АДАПТ ЖИВ (+%.2f бп к базе)" % (aa - ab) if verdict
      else "адапт не даёт преимущества (%+.2f бп)" % (aa - ab), flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
