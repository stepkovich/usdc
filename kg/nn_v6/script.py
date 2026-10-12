"""ПРОБА НЕЙРОСЕТИ НА СТАКАНЕ v6 (10.12, приказ владельца «данных
стало значительно больше — только смотри чтоб кегл не сломался»).

ЧТО ИЗМЕНИЛОСЬ против v5 (11 дней): запись с 25.09 — теперь ~17 дней;
тест = ПОСЛЕДНИЕ 4 ПОЛНЫХ ДНЯ (было 2), валидация 1 день, обучение —
всё до них. Архитектуры те же (честное сравнение с прошлыми пробами):
  TREE   — LightGBM на 13 признаках (реплика спринтера) — планка;
  NN-SEQ — LSTM на 60с истории признаков (окна каждые 16с вместо 10с —
           память, информация та же);
  NN-RAW — CNN на сырой лестнице 20 уровней × 4 канала (5с каденс,
           окно ~15 дней с жёстким капом строк).

ПРЕДРЕГИСТРАЦИЯ ВЕРДИКТА (до запуска): НАМЁК = квинтильный навык сети
превышает дерево на >= +1 бп В ПУЛЕ 4 тест-дней И минимум на 3 из 4
дней по отдельности. Иначе — «намёка нет». Проба, не решение о бое.

ЗАЩИТА ОТ ПАДЕНИЯ (урок OOM v5):
  - капы строк: таблица признаков <= 30М, сырьё <= 12М (режем каденс);
  - капы тензоров: LSTM-train <= 2.5М окон, CNN-train <= 3М строк
    (случайное прореживание с фиксированным сидом);
  - каждая ветка сети в try/except: упала — в отчёт идёт ошибка,
    дерево и вторая ветка выживают, report.json пишется ВСЕГДА.
"""
import glob, json, re, time, traceback
import numpy as np, pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn

t0 = time.time()
HORIZON_MS = 300_000
LAG_TOL = 2500
SEQ_LEN = 30
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("устройство:", DEV, flush=True)

FEATS = ["spread_bp", "microprice_rel", "imb5", "imb10", "imb20",
         "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
         "ntr10", "vpin10", "d30", "d120"]
LADDER = ["imb1", "slope_b", "slope_a", "wall_b", "wall_a"]
FEAT_ROWS_CAP = 30_000_000
RAW_ROWS_CAP = 12_000_000
SEQ_TRAIN_CAP = 2_500_000
RAW_TRAIN_CAP = 3_000_000

report = {"это_проба": True, "версия": 6,
          "предрегистрация": "НАМЁК = навык сети >= дерево+1бп в пуле "
                             "4 тест-дней И на >= 3 из 4 дней",
          "tree": {}, "nn_seq": None, "nn_raw": None, "errors": {}}

def fday(p):
    m = re.search(r"feat_(\d{8})_", p)
    return m.group(1) if m else ""

files = sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True))
feat_files, raw_files = [], []
for f in files:
    try:
        pd.read_parquet(f, columns=["bid_p_20"])
        raw_files.append(f)
    except Exception:
        feat_files.append(f)
print(f"файлов признаков {len(feat_files)}, с сырьём {len(raw_files)}",
      flush=True)

# ---------- таблица признаков (TREE и NN-SEQ) ----------
def load_feat(f):
    try:
        try:
            df = pd.read_parquet(f, columns=["ts", "symbol", "mid",
                                             "microprice"] + FEATS[:10])
        except Exception:
            df = pd.read_parquet(f, columns=["ts", "symbol", "mid",
                                             "microprice", "spread_bp",
                                             "imb5", "imb10", "imb20",
                                             "flow10_buy", "flow10_sell",
                                             "flow60_buy", "flow60_sell",
                                             "ntr10", "vpin10"])
        for c in LADDER:
            df[c] = np.float32(np.nan)
        return df
    except Exception:
        return None

frames = []
for f in files:
    df = load_feat(f)
    if df is not None:
        frames.append(df.iloc[::4])          # каденс 2с
df = pd.concat(frames, ignore_index=True)
del frames
if len(df) > FEAT_ROWS_CAP:                  # страховка RAM
    skip = int(np.ceil(len(df) / FEAT_ROWS_CAP))
    print(f"таблица {len(df)} строк > капа — прореживаю ::{skip}",
          flush=True)
    df = df.iloc[::skip]
df = df.drop_duplicates(["ts", "symbol"]).sort_values(["symbol", "ts"]) \
    .reset_index(drop=True)
print(f"строк признаков {len(df)} | {time.time()-t0:.0f}с", flush=True)

def add_labels_medians(g):
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    idx = np.clip(ts.searchsorted(ts + HORIZON_MS), 0, len(g) - 1)
    ok = ts[idx] >= ts + HORIZON_MS - 1500
    mid_fut = np.where(ok, mid[idx], np.nan)
    g["y"] = np.where(np.isnan(mid_fut), np.nan,
                      (mid_fut > mid).astype("float32"))
    g["fwd"] = np.where(ok, mid_fut / mid - 1, np.nan).astype("float32")
    g["microprice_rel"] = (g["microprice"] / mid - 1) * 10000
    for lag, name in ((30_000, "d30"), (120_000, "d120")):
        j = np.clip(ts.searchsorted(ts - lag), 0, len(g) - 1)
        past = np.where(np.abs(ts[j] - (ts - lag)) <= LAG_TOL, mid[j],
                        np.nan)
        g[name] = mid / past - 1
    for c in ("flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
              "ntr10"):
        med = g[c].median()
        g[c] = g[c] / (med if med and med > 0 else 1.0)
    return g

parts = []
for sym, g in df.groupby("symbol", observed=True):
    parts.append(add_labels_medians(g))
df = pd.concat(parts, ignore_index=True)
del parts
df = df.dropna(subset=FEATS + ["y"]).reset_index(drop=True)
df["day"] = pd.to_datetime(df["ts"], unit="ms").dt.date
all_days = sorted(df["day"].unique())
full_days = all_days[:-1]                    # последний день = сегодня,
                                             # неполный — не трогаем
test_days = full_days[-4:]
val_day = full_days[-5]
print(f"дней всего {len(all_days)}, тест {test_days}, вал {val_day}",
      flush=True)
report["days"] = {"all": len(all_days), "test": [str(d) for d in test_days]}

tr = df[df["day"] < val_day]
te = df[df["day"].isin(test_days)]

def day_skill(te_, p, label):
    """Квинтильный навык в пуле и по дням (свой пул строк у каждой
    модели — сравнение сквозное)."""
    d = te_.assign(p=p)
    res = {}
    fwd = d["fwd"].values
    ok = ~np.isnan(fwd)
    q = np.quantile(d["p"].values[ok], [0.2, 0.8])
    res["pool"] = float((fwd[ok][d["p"].values[ok] >= q[1]].mean()
                         - fwd[ok][d["p"].values[ok] <= q[0]].mean())
                        * 10000)
    per = {}
    for day, g in d.groupby("day"):
        f_ = g["fwd"].values
        ok_ = ~np.isnan(f_)
        if ok_.sum() < 1000:
            continue
        q_ = np.quantile(g["p"].values[ok_], [0.2, 0.8])
        per[str(day)] = float((f_[ok_][g["p"].values[ok_] >= q_[1]].mean()
                               - f_[ok_][g["p"].values[ok_] <= q_[0]]
                               .mean()) * 10000)
    res["per_day"] = per
    acc = float(((d["p"].values > 0.5) == (d["y"].values > 0.5)).mean()
                * 100)
    res["acc"] = acc
    print(f"{label}: acc {acc:.2f}%, навык пул {res['pool']:+.2f} бп, "
          f"по дням " + " ".join(f"{k}:{v:+.1f}" for k, v in per.items()),
          flush=True)
    return res

# ---------- TREE бейзлайн ----------
m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=4,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=42, n_jobs=4, verbosity=-1)
m.fit(tr[FEATS], tr["y"])
p_tree = m.predict_proba(te[FEATS])[:, 1]
report["tree"] = {k: v for k, v in day_skill(te, p_tree, "TREE").items()}
report["tree"]["per_day"] = {k: round(v, 2)
                             for k, v in report["tree"]["per_day"].items()}
report["tree"]["pool"] = round(report["tree"]["pool"], 2)

# ---------- NN-SEQ: LSTM на 60с истории ----------
try:
    print("NN-SEQ: сборка последовательностей...", flush=True)
    def build_seq(g):
        g = g.sort_values("ts").reset_index(drop=True)
        X = g[FEATS].values.astype("float32")
        y = g["y"].values.astype("float32")
        f_ = g["fwd"].values.astype("float32")
        ts = g["ts"].values
        n = len(g) - SEQ_LEN
        if n <= 0:
            return None
        idx = np.arange(SEQ_LEN, len(g))
        seq = np.lib.stride_tricks.sliding_window_view(X, SEQ_LEN,
                                                       axis=0) \
            .transpose(0, 2, 1)[:n]
        seq = seq[::8]                       # окно каждые 16с (память)
        return seq, np.stack([y[SEQ_LEN:][:n][::8][:len(seq)],
                              f_[SEQ_LEN:][:n][::8][:len(seq)],
                              ts[SEQ_LEN:][:n][::8][:len(seq)]])

    seq_parts = []
    for sym, g in df.groupby("symbol", observed=True):
        b = build_seq(g)
        if b is not None:
            seq_parts.append((sym, b[0], b[1]))
    train_end_ms = pd.Timestamp(str(val_day)).timestamp() * 1000
    test0_ms = pd.Timestamp(str(test_days[0])).timestamp() * 1000
    tr_lists = [(s[(t[2] < train_end_ms)], t[0][t[2] < train_end_ms],
                 t[1][t[2] < train_end_ms]) for _, s, t in seq_parts]
    all_train = np.concatenate([a for a, _, _ in tr_lists])
    mu = all_train.reshape(-1, len(FEATS)).mean(0)
    sd = all_train.reshape(-1, len(FEATS)).std(0) + 1e-6
    del all_train
    Xtr = np.concatenate([a for a, _, _ in tr_lists])
    ytr = np.concatenate([b for _, b, _ in tr_lists])
    del tr_lists
    if len(Xtr) > SEQ_TRAIN_CAP:             # страховка GPU
        rng = np.random.default_rng(42)
        keep = rng.choice(len(Xtr), SEQ_TRAIN_CAP, replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]
    print(f"NN-SEQ: train {Xtr.shape} | {time.time()-t0:.0f}с", flush=True)

    class LSTMNet(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lstm = nn.LSTM(d, 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                      nn.Linear(32, 2))
        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1])

    model = LSTMNet(len(FEATS)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(ytr.astype("int64"), device=DEV)
    del Xtr, ytr
    for ep in range(4):
        model.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), 8192):
            b = perm[i:i + 8192]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xt[:50000]).argmax(1)
                         == yt[:50000]).float().mean())
        print(f"  seq эп.{ep}: loss {loss.item():.4f}, "
              f"acc {acc*100:.2f}%", flush=True)
    del Xt, yt
    model.eval()
    prs, fwd_all, day_all = [], [], []
    with torch.no_grad():
        for sym, s, t3 in seq_parts:
            m_ = t3[2] >= test0_ms
            if not m_.any():
                continue
            Xb = torch.tensor((s[m_] - mu) / sd, device=DEV)
            prs.append(model(Xb).softmax(1)[:, 1].cpu().numpy())
            fwd_all.append(t3[1][m_])
            day_all.append(pd.to_datetime(t3[2][m_], unit="ms").date)
    pr = np.concatenate(prs)
    fwdv = np.concatenate(fwd_all)
    dayv = np.concatenate(day_all)
    ok = ~np.isnan(fwdv)
    acc_seq = float(((pr[ok] > 0.5) == (fwdv[ok] > 0)).mean() * 100)
    q = np.quantile(pr[ok], [0.2, 0.8])
    pool = float((fwdv[ok][pr[ok] >= q[1]].mean()
                  - fwdv[ok][pr[ok] <= q[0]].mean()) * 10000)
    per = {}
    for dd in sorted(set(dayv[ok])):
        mm = dayv == dd
        o_ = ok & mm
        if o_.sum() < 1000:
            continue
        q_ = np.quantile(pr[o_], [0.2, 0.8])
        per[str(dd)] = float((fwdv[o_][pr[o_] >= q_[1]].mean()
                              - fwdv[o_][pr[o_] <= q_[0]].mean()) * 10000)
    report["nn_seq"] = {"acc": round(acc_seq, 2), "pool_bp": round(pool, 2),
                        "per_day": {k: round(v, 2)
                                    for k, v in per.items()}}
    print(f"NN-SEQ: acc {acc_seq:.2f}%, навык пул {pool:+.2f} бп",
          flush=True)
    del seq_parts
except Exception:
    report["errors"]["nn_seq"] = traceback.format_exc()[-1500:]
    print("NN-SEQ УПАЛ:", traceback.format_exc()[-500:], flush=True)

# ---------- NN-RAW: CNN на сырой лестнице ----------
try:
    print("NN-RAW: загрузка сырых уровней...", flush=True)
    raw_cols = (["ts", "symbol", "mid"]
                + [f"bid_p_{i}" for i in range(1, 21)]
                + [f"bid_q_{i}" for i in range(1, 21)]
                + [f"ask_p_{i}" for i in range(1, 21)]
                + [f"ask_q_{i}" for i in range(1, 21)])
    rframes = []
    use = raw_files[-130:]                   # ~13 дней (память)
    for f in use:
        try:
            rf = pd.read_parquet(f, columns=raw_cols)
            for c in raw_cols:               # float32: вдвое меньше RAM
                if c not in ("ts", "symbol"):
                    rf[c] = rf[c].astype("float32")
            rframes.append(rf.iloc[::10])    # 5с каденс (память GPU)
        except Exception:
            pass
    rdf = pd.concat(rframes, ignore_index=True)
    del rframes
    if len(rdf) > RAW_ROWS_CAP:              # страховка RAM
        skip = int(np.ceil(len(rdf) / RAW_ROWS_CAP))
        print(f"сырьё {len(rdf)} > капа — режу ::{skip}", flush=True)
        rdf = rdf.iloc[::skip]
    rdf = rdf.dropna().sort_values(["symbol", "ts"]).reset_index(drop=True)
    print(f"сырых строк {len(rdf)} | {time.time()-t0:.0f}с", flush=True)

    def raw_xy(g):
        ts = g["ts"].values
        mid = g["mid"].values.astype("float64")
        idx = np.clip(ts.searchsorted(ts + HORIZON_MS), 0, len(g) - 1)
        ok = ts[idx] >= ts + HORIZON_MS - 2500
        y = np.where(ok, mid[idx] > mid, np.nan)
        bp = g[[f"bid_p_{i}" for i in range(1, 21)]].values.astype("float64")
        bq = g[[f"bid_q_{i}" for i in range(1, 21)]].values.astype("float64")
        ap = g[[f"ask_p_{i}" for i in range(1, 21)]].values.astype("float64")
        aq = g[[f"ask_q_{i}" for i in range(1, 21)]].values.astype("float64")
        tot = bq.sum(1) + aq.sum(1) + 1e-12
        X = np.stack([(mid[:, None] - bp) / mid[:, None] * 10000,
                      (ap - mid[:, None]) / mid[:, None] * 10000,
                      bq / tot[:, None], aq / tot[:, None]], axis=2)
        fwd = np.where(ok, mid[idx] / mid - 1, np.nan)
        return X.astype("float32"), y.astype("float32"), fwd, ts

    rparts = []
    for sym, g in rdf.groupby("symbol", observed=True):
        X, y, fwd, ts = raw_xy(g)
        okm = ~np.isnan(y)
        rparts.append((X[okm], y[okm], fwd[okm], ts[okm].astype("int64")))
    del rdf
    Xr = np.concatenate([a for a, b, c, d in rparts])
    yr = np.concatenate([b for a, b, c, d in rparts])
    fr = np.concatenate([c for a, b, c, d in rparts])
    tr_ = np.concatenate([d for a, b, c, d in rparts])
    del rparts
    print(f"NN-RAW: {Xr.shape} | {time.time()-t0:.0f}с", flush=True)

    raw_day = pd.to_datetime(tr_, unit="ms").date
    uniq = sorted(set(raw_day))[:-1]         # без сегодняшнего
    raw_test_start = pd.Timestamp(str(uniq[-4])).timestamp() * 1000
    mte = tr_ >= raw_test_start
    mtr = ~mte
    print(f"NN-RAW дней {len(uniq)+1}, тест с {uniq[-4]}", flush=True)

    class CNNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(4, 32, 3, padding=1), nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool1d(4), nn.Flatten(),
                nn.Linear(64 * 4, 64), nn.ReLU(), nn.Linear(64, 2))
        def forward(self, x):
            return self.net(x.permute(0, 2, 1))

    if mtr.sum() > 100000 and mte.sum() > 10000:
        Xi, yi = Xr[mtr], yr[mtr].astype("int64")
        if len(Xi) > RAW_TRAIN_CAP:          # страховка GPU
            rng = np.random.default_rng(42)
            keep = rng.choice(len(Xi), RAW_TRAIN_CAP, replace=False)
            Xi, yi = Xi[keep], yi[keep]
        model2 = CNNNet().to(DEV)
        Xt2 = torch.tensor(Xi, device=DEV)
        yt2 = torch.tensor(yi, device=DEV)
        del Xi, yi
        opt = torch.optim.Adam(model2.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        for ep in range(4):
            model2.train()
            perm = torch.randperm(len(Xt2), device=DEV)
            for i in range(0, len(Xt2), 8192):
                b = perm[i:i + 8192]
                opt.zero_grad()
                loss = lossf(model2(Xt2[b]), yt2[b])
                loss.backward()
                opt.step()
            print(f"  raw эп.{ep}: loss {loss.item():.4f}", flush=True)
        del Xt2, yt2
        model2.eval()
        prs = []
        Xte = Xr[mte]
        with torch.no_grad():
            for i in range(0, len(Xte), 65536):
                xb = torch.tensor(Xte[i:i + 65536], device=DEV)
                prs.append(model2(xb).softmax(1)[:, 1].cpu().numpy())
        pr = np.concatenate(prs)
        print(f"    CNN диагностика: std(p)={pr.std():.5f}", flush=True)
        fwd_te = fr[mte]
        yr_te = yr[mte]
        day_te = pd.to_datetime(tr_[mte], unit="ms").date
        ok = ~np.isnan(fwd_te)
        acc_raw = float(((pr[ok] > 0.5) == (yr_te[ok] > 0.5)).mean() * 100)
        q = np.quantile(pr[ok], [0.2, 0.8])
        pool = float((fwd_te[ok][pr[ok] >= q[1]].mean()
                      - fwd_te[ok][pr[ok] <= q[0]].mean()) * 10000)
        per = {}
        for dd in sorted(set(day_te[ok])):
            mm = day_te == dd
            o_ = ok & mm
            if o_.sum() < 1000:
                continue
            q_ = np.quantile(pr[o_], [0.2, 0.8])
            per[str(dd)] = float((fwd_te[o_][pr[o_] >= q_[1]].mean()
                                  - fwd_te[o_][pr[o_] <= q_[0]].mean())
                                 * 10000)
        report["nn_raw"] = {"acc": round(acc_raw, 2),
                            "pool_bp": round(pool, 2),
                            "per_day": {k: round(v, 2)
                                        for k, v in per.items()}}
        print(f"NN-RAW: acc {acc_raw:.2f}%, навык пул {pool:+.2f} бп",
              flush=True)
    else:
        report["errors"]["nn_raw"] = f"мало строк: tr={mtr.sum()}, " \
                                     f"te={mte.sum()}"
except Exception:
    report["errors"]["nn_raw"] = traceback.format_exc()[-1500:]
    print("NN-RAW УПАЛ:", traceback.format_exc()[-500:], flush=True)

# ---------- вердикт по предрегистрации ----------
try:
    verdict = {}
    for name, r in (("nn_seq", report["nn_seq"]),
                    ("nn_raw", report["nn_raw"])):
        if not r:
            verdict[name] = "нет результата"
            continue
        delta_pool = r["pool_bp"] - report["tree"]["pool"]
        wins = sum(1 for d, v in r["per_day"].items()
                   if report["tree"]["per_day"].get(d) is not None
                   and v - report["tree"]["per_day"][d] >= 1.0)
        need_days = max(1, int(np.ceil(0.75 * len(r["per_day"]))))
        ok_ = delta_pool >= 1.0 and wins >= min(3, need_days)
        verdict[name] = {"delta_pool_bp": round(delta_pool, 2),
                         "days_beating_tree": wins,
                         "hint": bool(ok_)}
        print(f"ВЕРДИКТ {name}: Δ пула {delta_pool:+.2f} бп, дней "
              f"против дерева {wins} -> "
              f"{'НАМЁК ЕСТЬ' if ok_ else 'намёка нет'}", flush=True)
    report["verdict"] = verdict
except Exception:
    report["errors"]["verdict"] = traceback.format_exc()[-800:]

json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ПРОБА v6 ЗАВЕРШЕНА", flush=True)
