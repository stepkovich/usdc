"""РЕТРАЙ ПРОБЫ НЕЙРОСЕТИ (08.10): данных стало вдвое больше (~11
дней записи), эпох 4 вместо 2. Правила прежние: намёк = навык сети >
дерева на +1 бп на обоих тест-днях. Планка: дерево-реплика спринтера.
Проба, не решение о бое.

СТАРЫЙ ЗАГОЛОВОК:"""
"""(архив) ПРОБА НЕЙРОСЕТИ НА СТАКАНЕ (предрегистрация 07.10, тест механизма).

ЗАЧЕНО ДО ЗАПУСКА: это ТЕСТ МЕХАНИЗМА по приказу владельца («сделай на
том что есть»), НЕ решение о бое. Вопрос один: (а) собирается ли ГПУ-
конвейер端-в-端 и (б) есть ли ХОТЬ НАМЁК, что сеть видит больше деревьев.
Данных мало (сырая лестница с 01.10 ~6 дней; признаки с 25.09 ~12 дней)
— статистически это разведка.

ЗАДАЧА: как у спринтера — направление середины через 300с (y = mid_fut
> mid). Метрика: точность + квинтильный навык (топ-минус-боттом по
уверенности, бп, мид-к-миду без издержек) НА ОДНИХ И ТЕХ ЖЕ тест-днях:
  TREE  — бейзлайн: LightGBM на одной строке признаков (реплика
          спринтера, 13 признаков) — планка;
  NN-SEQ— сеть на ПОСЛЕДОВАТЕЛЬНОСТИ: последние 60 строк (30с) × 13
          признаков -> LSTM(64)+dense. Деревья времени не видят —
          проверяем, есть ли в динамике признаков добавка;
  NN-RAW— сеть на СЫРОЙ ЛЕСТНИЦЕ: снапшот 20 уровней × 4 канала
          (доли объёмов bid/ask + дистанции в бп) -> CNN(2 conv1d)
          + dense. Проверяем, есть ли в форме то, чего нет в числах.
РАЗДЕЛЫ: обучение = все дни до тестовых (минус последний — валидация),
тест = последние 2 полных дня. Стоп-критерий: нет — это проба.
ВЫВОД (форма): «есть намёк», если квинтильный навык сети > дерева
минимум на +1 бп на обоих тест-днях; иначе «намёка нет, ждём данных».
"""
import glob, json, re, time
import numpy as np, pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn

t0 = time.time()
HORIZON_MS = 300_000
LAG_TOL = 2500
SEQ_LEN = 30                   # 30 кадров по 2с = 60с истории
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("устройство:", DEV, flush=True)

FEATS = ["spread_bp", "microprice_rel", "imb5", "imb10", "imb20",
         "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
         "ntr10", "vpin10", "d30", "d120"]
LADDER = ["imb1", "slope_b", "slope_a", "wall_b", "wall_a"]

def fday(p):
    m = re.search(r"feat_(\d{8})_", p)
    return m.group(1) if m else ""

files = sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True))
# сырьё определяем по наличию колонки bid_p_20 (файлы с 01.10)
feat_files, raw_files = [], []
for f in files:
    try:
        pd.read_parquet(f, columns=["bid_p_20"])
        raw_files.append(f)
    except Exception:
        feat_files.append(f)
print(f"файлов признаков {len(feat_files)}, с сырьём {len(raw_files)}",
      flush=True)

# ---------- таблица признаков (для TREE и NN-SEQ) ----------
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
for f in files:                          # ВСЕ файлы: в сырых есть те же
    df = load_feat(f)                    # признаковые колонки (21 колонка)
    if df is not None:
        frames.append(df.iloc[::4])      # каденс 2с: память конечна
df = pd.concat(frames, ignore_index=True)
del frames
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
days = sorted(df["day"].unique())
test_days = days[-2:]
val_day = days[-3]
print(f"дней {len(days)}, тест {test_days}, вал {val_day}", flush=True)

tr = df[df["day"] < val_day]
va = df[df["day"] == val_day]
te = df[df["day"].isin(test_days)]

# ---------- TREE бейзлайн ----------
m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=4,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=42, n_jobs=4, verbosity=-1)
m.fit(tr[FEATS], tr["y"])
p_tree = m.predict_proba(te[FEATS])[:, 1]

def skill(te_, p):
    y = te_["y"].values
    fwd = te_["fwd"].values if "fwd" in te_ else None
    acc = float(((p > 0.5) == (y > 0.5)).mean() * 100)
    return acc

acc_tree = skill(te, p_tree)
print(f"TREE: acc {acc_tree:.2f}%", flush=True)

# квинтильный навык нужен по fwd — пересчитаем fwd на тесте
def add_fwd(te_):
    out = []
    for sym, g in te_.groupby("symbol", observed=True):
        g = g.sort_values("ts")
        ts = g["ts"].values
        mid = g["mid"].values.astype("float64")
        idx = np.clip(ts.searchsorted(ts + HORIZON_MS), 0, len(g) - 1)
        ok = ts[idx] >= ts + HORIZON_MS - 1500
        g["fwd"] = np.where(ok, mid[idx] / mid - 1, np.nan)
        out.append(g)
    return pd.concat(out, ignore_index=True)
te = add_fwd(te)

def qskill(p):
    ok = ~np.isnan(te["fwd"].values)
    q = np.quantile(p[ok], [0.2, 0.8])
    fwd = te["fwd"].values[ok]
    return float((fwd[p[ok] >= q[1]].mean()
                  - fwd[p[ok] <= q[0]].mean()) * 10000)

qs_tree = qskill(p_tree)
print(f"TREE: квинтильный навык {qs_tree:+.2f} бп", flush=True)

# ---------- NN-SEQ: LSTM на 60 строках ----------
print("NN-SEQ: сборка последовательностей...", flush=True)
def build_seq(g):
    g = g.sort_values("ts").reset_index(drop=True)
    X = g[FEATS].values.astype("float32")
    y = g["y"].values.astype("float32")
    ts = g["ts"].values
    n = len(g) - SEQ_LEN
    if n <= 0:
        return None, None
    idx = np.arange(SEQ_LEN, len(g))
    seq = np.lib.stride_tricks.sliding_window_view(X, SEQ_LEN, axis=0) \
        .transpose(0, 2, 1)[:n]
    seq = seq[::5]                         # окно каждые 10с
    y_sel = y[SEQ_LEN:][:n][::5][:len(seq)]
    ts_sel = ts[SEQ_LEN:][:n][::5][:len(seq)]
    return seq, np.stack([y_sel, ts_sel])

seq_parts, seen = [], set()
for sym, g in df.groupby("symbol", observed=True):
    s, yt = build_seq(g)
    if s is None:
        continue
    seq_parts.append((sym, s, yt[0], yt[1]))
# нормализация по обучающим дням
train_end_ms = pd.Timestamp(str(val_day)).timestamp() * 1000
all_seq_train = np.concatenate([s[(t < train_end_ms)] for _, s, y, t
                                in seq_parts]) if seq_parts else None
mu = all_seq_train.reshape(-1, len(FEATS)).mean(0)
sd = all_seq_train.reshape(-1, len(FEATS)).std(0) + 1e-6
del all_seq_train

def seq_xy(day_mask):
    Xs, ys = [], []
    for sym, s, y, t in seq_parts:
        m_ = day_mask(t.astype("int64"))
        if m_.any():
            Xs.append((s[m_] - mu) / sd)
            ys.append(y[m_])
    if not Xs:
        return None, None
    return np.concatenate(Xs), np.concatenate(ys)

Xtr, ytr = seq_xy(lambda t: t < train_end_ms)
Xva, yva = seq_xy(lambda t: (t >= train_end_ms)
                  & (t < pd.Timestamp(str(test_days[0])).timestamp() * 1000))
print(f"NN-SEQ: train {Xtr.shape}, val {Xva.shape} | {time.time()-t0:.0f}с",
      flush=True)

class LSTMNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lstm = nn.LSTM(d, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                  nn.Linear(32, 2))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1])

def train_torch(model, Xtr, ytr, Xva, yva, epochs=4, bs=8192):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(ytr.astype("int64"), device=DEV)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xt[b])
            loss = lossf(out, yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xt[:50000]).argmax(1)
                         == yt[:50000]).float().mean())
        print(f"  эп.{ep}: loss {loss.item():.4f}, acc {acc*100:.2f}%",
              flush=True)
    return model

if Xtr is not None and len(Xtr) > 100000:
    model = LSTMNet(len(FEATS)).to(DEV)
    model = train_torch(model, Xtr, ytr, Xva, yva)
    torch.save(model.state_dict(), "/kaggle/working/lstm.pt")
    # тест-предсказание по символам, те же строки что у дерева —
    # квинтиль по своей выборке (честно: свой пул тест-строк)
    p_seq_parts = []
    with torch.no_grad():
        for sym, s, y, t in seq_parts:
            m_ = t >= pd.Timestamp(str(test_days[0])).timestamp() * 1000
            if not m_.any():
                continue
            Xb = torch.tensor(((s[m_] - mu) / sd), device=DEV)
            pr = model(Xb).softmax(1)[:, 1].cpu().numpy()
            p_seq_parts.append((pr, y[m_], sym))
    pr_all = np.concatenate([a for a, b, c in p_seq_parts])
    y_all = np.concatenate([b for a, b, c in p_seq_parts])
    acc_seq = float(((pr_all > 0.5) == (y_all > 0.5)).mean() * 100)
    q20, q80 = np.quantile(pr_all, [0.2, 0.8])
    print(f"NN-SEQ: acc {acc_seq:.2f}% (свой пул тест-строк)", flush=True)
else:
    acc_seq = None
    print("NN-SEQ: не хватает строк", flush=True)

# ---------- NN-RAW: CNN на сырой лестнице ----------
print("NN-RAW: загрузка сырых уровней...", flush=True)
raw_cols = (["ts", "symbol", "mid"]
            + [f"bid_p_{i}" for i in range(1, 21)]
            + [f"bid_q_{i}" for i in range(1, 21)]
            + [f"ask_p_{i}" for i in range(1, 21)]
            + [f"ask_q_{i}" for i in range(1, 21)])
rframes = []
raw_files = raw_files[-90:]               # последние ~9 дней (память)
for f in raw_files:
    try:
        rf = pd.read_parquet(f, columns=raw_cols)
        rf = rf.iloc[::8]                  # 4с каденс (память GPU-хоста)
        rframes.append(rf)
    except Exception:
        pass
rdf = pd.concat(rframes, ignore_index=True)
del rframes
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
    X = np.stack([(mid[:, None] - bp) / mid[:, None] * 10000,   # дист бп
                  (ap - mid[:, None]) / mid[:, None] * 10000,
                  bq / tot[:, None], aq / tot[:, None]], axis=2)
    fwd = np.where(ok, mid[idx] / mid - 1, np.nan)
    return X.astype("float32"), y.astype("float32"), fwd, ts

rparts = []
for sym, g in rdf.groupby("symbol", observed=True):
    X, y, fwd, ts = raw_xy(g)
    ok = ~np.isnan(y)
    rparts.append((X[ok], y[ok], fwd[ok], ts[ok].astype("int64")))
del rdf
Xr = np.concatenate([a for a, b, c, d in rparts])
yr = np.concatenate([b for a, b, c, d in rparts])
fr = np.concatenate([c for a, b, c, d in rparts])
tr_ = np.concatenate([d for a, b, c, d in rparts])
print(f"NN-RAW: {Xr.shape} | {time.time()-t0:.0f}с", flush=True)

raw_day = pd.to_datetime(tr_, unit="ms").date
uniq = sorted(set(raw_day))
raw_test_start = pd.Timestamp(str(uniq[-2])).timestamp() * 1000
mtr = tr_ < raw_test_start
mte = ~mtr
print(f"NN-RAW дни: {len(uniq)}, тест с {uniq[-2]}", flush=True)
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
    model2 = CNNNet().to(DEV)
    Xt2 = torch.tensor(Xr[mtr], device=DEV)
    yt2 = torch.tensor(yr[mtr].astype("int64"), device=DEV)
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
    torch.save(model2.state_dict(), "/kaggle/working/cnn.pt")
    model2.eval()
    prs = []
    with torch.no_grad():
        for i in range(0, int(mte.sum()), 65536):
            xb = torch.tensor(Xr[mte][i:i + 65536], device=DEV)
            prs.append(model2(xb).softmax(1)[:, 1].cpu().numpy())
    pr = np.concatenate(prs)
    print(f"    CNN диагностика: std(p)={pr.std():.5f} min={pr.min():.4f} "
          f"max={pr.max():.4f}", flush=True)
    acc_raw = float(((pr > 0.5) == (yr[mte] > 0.5)).mean() * 100)
    q20, q80 = np.quantile(pr, [0.2, 0.8])
    fwd_te = fr[mte]
    qs_raw = float((fwd_te[pr >= q80].mean()
                    - fwd_te[pr <= q20].mean()) * 10000)
    print(f"NN-RAW: acc {acc_raw:.2f}%, квинтильный навык {qs_raw:+.2f} бп",
          flush=True)
else:
    acc_raw = qs_raw = None
    print("NN-RAW: мало строк", flush=True)

report = {"это_тест_механизма": True,
          "tree": {"acc": round(acc_tree, 2), "qskill_bp": round(qs_tree, 2)},
          "nn_seq_acc": None if acc_seq is None else round(acc_seq, 2),
          "nn_raw": None if acc_raw is None else {
              "acc": round(acc_raw, 2), "qskill_bp": round(qs_raw, 2)},
          "note": "данных 6-12 дней: разведка, не решение"}
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ПРОБА НЕЙРОСЕТИ ЗАВЕРШЕНА", flush=True)
