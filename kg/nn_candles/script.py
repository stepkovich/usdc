"""ПРОБА НЕЙРОСЕТИ НА СВЕЧАХ (предрегистрация 07.10, тест механизма).

ЗАЧЕНО ДО ЗАПУСКА: разведка на ГПУ. Вопрос: видит ли сеть в СЫРЫХ
свечах (последние 48 баров, 6 каналов) то, чего нет в 24 рукописных
признаках? Планка — то же дерево, что в боевой модели-буре.

ДАННЫЕ: панель 520 USDT-пар, 30м, 2020-2026 (~20.5М строк, panel.zip).
МЕТКА (как у боевой): y=1 fwd>0.5×ATR14, y=0 fwd<−0.5×ATR, болото NaN.
Фолды: 8 кварталов linspace (как в боевом экзамене); в фолде:
  TREE  — LightGBM на 24 признаках (планка);
  NN    — CNN по оси времени: окно 48 баров × 6 каналов
          (log-ret, (high-low)/close, (close-open)/close,
          vol/ma20, taker-доля, позиция в диапазоне 48),
          per-window z-норма; Conv1d(6→32→64)+pool+head.
МЕТРИКИ на тест-квартале: acc и квинтильный навык fwd (бп, мид-мид,
без издержек) — сравнение на ОДНИХ И ТЕХ ЖЕ строках.
ВЫВОД (форма): «намёк есть», если pooled навык NN > TREE на +1 бп ИЛИ
acc NN выше дерева в >=5 фолдах из 8; иначе «свечи для сети тоже
пусты на этом объёме данных». Разведка, не решение.
"""
import glob, io, json, time, zipfile, zlib, datetime
import numpy as np, pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn

t0 = time.time()
H = 10
WIN = 48
STRIDE = 4
GATE = 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("устройство:", DEV, flush=True)
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
if not zips:
    json.dump({"error": "panel.zip not found"},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
print("монет:", len(members), flush=True)
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
    X["zscore"] = ((close - sma) / std).replace([np.inf, -np.inf],
                                                np.nan).astype("float32")
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
    return X, tr

# каналы сырых окон (6): считаются на лету из массивов
def raw_windows(sym_arr, start, stop, stride, max_row=None):
    """sym_arr: dict с close/high/low/vol/tb/atr (float64 np).
    max_row: последний допустимый row (окна не залезают метками за него).
    Возврат X [n,48,6] float32, y, fwd, ot — строки [start,stop) шаг stride."""
    c = sym_arr["close"]; o = sym_arr["open"]
    hi = sym_arr["high"]; lo = sym_arr["low"]
    v = sym_arr["vol"]; tb = sym_arr["tb"]; atr = sym_arr["atr"]
    ot = sym_arr["ot"]
    rows = np.arange(start, min(stop, (max_row or len(c) - H - 1)) + 1,
                     stride)
    rows = rows[rows >= WIN]
    if len(rows) == 0:
        return None, None, None, None
    # окно в хронологическом порядке: col0 = самый старый бар
    idx = (rows[:, None] - (WIN - 1)) + np.arange(WIN)[None, :]
    idx = idx.ravel()
    cl = c[idx]; op = o[idx]; hh = hi[idx]; ll = lo[idx]
    vv = v[idx]; tbb = tb[idx]
    n = len(rows)
    cl = cl.reshape(n, WIN); op = op.reshape(n, WIN)
    hh = hh.reshape(n, WIN); ll = ll.reshape(n, WIN)
    vv = vv.reshape(n, WIN); tbb = tbb.reshape(n, WIN)
    logret = np.zeros((n, WIN), dtype="float64")
    logret[:, 1:] = np.log(cl[:, 1:] / cl[:, :-1])
    hl = (hh - ll) / cl
    co = (cl - op) / cl
    vma = sym_arr["vol_ma"][idx].reshape(n, WIN)
    vr = np.minimum(vv / (vma + 1e-12), 10.0)
    trc = np.clip(tbb / (vv + 1e-12), 0, 1)
    w_hi = hh.max(1); w_lo = ll.min(1)
    pos = np.repeat(((cl[:, -1] - w_lo) / (w_hi - w_lo + 1e-12))[:, None],
                    WIN, axis=1)
    X = np.stack([logret, hl, co, vr, trc, pos], axis=2).astype("float32")
    # per-window z-норма по каждому каналу
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-6
    X = (X - m) / s
    fwd = c[rows + H] / c[rows] - 1
    atr_r = atr[rows]
    y = np.where(np.isnan(fwd) | np.isnan(atr_r), np.nan,
                 np.where(fwd > 0.5 * atr_r, 1.0,
                          np.where(fwd < -0.5 * atr_r, 0.0, np.nan)))
    ok = (~np.isnan(y)) & (~np.isnan(fwd)) & (~np.isnan(X).any(axis=(1, 2)))
    return X[ok], y[ok], fwd[ok], ot[rows][ok]

FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]

syms_arr = {}
feat_parts = []
for i, name in enumerate(members, 1):
    sym = name.replace("_30m.csv", "")
    df = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["open_time", "open", "high", "low", "close",
                              "volume", "quote_volume", "taker_buy_base"])
    for c in df.columns:
        if c != "open_time":
            df[c] = df[c].astype("float32")
    df["open_time"] = df["open_time"].astype(np.int64)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    X, tr = make_feats(df, btc, sym)
    close = df["close"].values
    atr_rel = (tr.rolling(14).mean() / close).values.astype("float64")
    fut = np.full(len(df), np.nan, dtype="float64")
    fut[:-H] = close[H:]
    fwd = fut / close - 1
    thr = 0.5 * np.nan_to_num(atr_rel, nan=np.nan)
    X["y"] = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                      np.where(fwd > thr, 1.0,
                               np.where(fwd < -thr, 0.0, np.nan)))
    X["fwd"] = fwd.astype("float32")
    X["ot"] = df["open_time"].values.astype(np.int64)
    X["sym"] = sym
    X = X.replace([np.inf, -np.inf], np.nan)
    g = X.dropna(subset=[c for c in X.columns if c not in ("fwd",)])
    feat_parts.append(g)
    # сырьё для окон (только строки, вошедшие в g, чтобы деревья и сеть
    # видели одни и те же моменты: бин по ot)
    keep_ot = set(g["ot"].values.tolist())
    atr_abs = (tr.rolling(14).mean()).values.astype("float64")
    syms_arr[sym] = {
        "vol_ma": (df["volume"].rolling(20).mean()).values.astype("float64"),
        "close": close.astype("float64"),
        "open": df["open"].values.astype("float64"),
        "high": df["high"].values.astype("float64"),
        "low": df["low"].values.astype("float64"),
        "vol": df["volume"].values.astype("float64"),
        "tb": df["taker_buy_base"].values.astype("float64"),
        "atr": atr_rel, "atr_abs": atr_abs, "ot": df["open_time"].values,
        "keep_ot": keep_ot}
    if i % 100 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(feat_parts, ignore_index=True)
del feat_parts
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
FEATS_ALL = FEATS
print(f"строк {len(data)}, фолды {test_q} | {time.time()-t0:.0f}с",
      flush=True)

# карта ot -> строка для каждого символа (для окон по тем же моментам)
ot_index = {}
for sym, arr in syms_arr.items():
    ot_index[sym] = {ot: k for k, ot in enumerate(arr["ot"].tolist())}

class CNNNet(nn.Module):
    def __init__(self, ch=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveMaxPool1d(8), nn.Flatten(),
            nn.Linear(64 * 8, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2))
    def forward(self, x):
        return self.net(x.permute(0, 2, 1))

def train_cnn(Xtr, ytr, epochs=6, bs=8192):
    model = CNNNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(ytr.astype("int64"), device=DEV)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=DEV)
        tot_loss = 0.0
        nb = 0
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xt[b])
            loss = lossf(out, yt[b])
            loss.backward()
            opt.step()
            tot_loss += loss.item(); nb += 1
        print(f"    эп.{ep}: loss {tot_loss/max(1,nb):.4f}", flush=True)
    return model

report = {"это_проба": True, "folds": {}, "tree": [], "nn": []}
pooled = {"tree_acc": [], "tree_sk": [], "nn_acc": [], "nn_sk": [],
          "tree_acc_folds": 0, "n_folds": 0}
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 20000 or len(trd) < 500_000:
        continue
    # TREE
    mt = lgb.LGBMClassifier(**PARAMS)
    mt.fit(trd[FEATS_ALL], trd["y"], categorical_feature=["sym_id"])
    pt = mt.predict_proba(ted[FEATS_ALL])[:, 1]
    acc_t = float(((pt > 0.5) == (ted["y"] > 0.5)).mean() * 100)
    f_te = ted["fwd"].values.astype("float64")
    q20, q80 = np.quantile(pt, [0.2, 0.8])
    sk_t = float((f_te[pt >= q80].mean()
                  - f_te[pt <= q20].mean()) * 10000)
    # NN: окна обучения (те же моменты trd, stride 4) и теста (все)
    Xtr_l, ytr_l = [], []
    tr_syms = trd[["sym", "ot"]].drop_duplicates()
    q_start_ms = int(ted["ot"].min())
    ot_by_sym = {s: g["ot"].values for s, g in trd.groupby("sym")}
    for sym, ots in ot_by_sym.items():
        arr = syms_arr.get(sym)
        if arr is None:
            continue
        keep = arr["keep_ot"]
        pos = [ot_index[sym][int(o)] for o in ots if int(o) in keep]
        if not pos:
            continue
        # окна train: их горизонт (row+H) не должен заходить в тест
        first_test_row = int(np.searchsorted(arr["ot"], q_start_ms))
        max_row = first_test_row - H - 1
        Xw, yw, _, _ = raw_windows(arr, min(pos), max(pos) + 1, STRIDE,
                                   max_row=max_row)
        if Xw is not None:
            Xtr_l.append(Xw); ytr_l.append(yw)
    Xtr = np.concatenate(Xtr_l); ytr = np.concatenate(ytr_l)
    del Xtr_l, ytr_l
    # убрать утечку: окна, чей горизонт заходит в тест-квартал
    q_start_ms = ted["ot"].min()
    # (упрощение: окна обучаются только на барах до начала теста)
    Xte_l, yte_l, fte_l, symte_l = [], [], [], []
    for sym, g in ted.groupby("sym"):
        arr = syms_arr.get(sym)
        if arr is None:
            continue
        keep = arr["keep_ot"]
        pos = [ot_index[sym][int(o)] for o in g["ot"].values
               if int(o) in keep]
        if not pos:
            continue
        Xw, yw, fw, ots = raw_windows(arr, min(pos), max(pos) + 1, 1,
                                      max_row=len(arr["close"]) - H - 1)
        if Xw is not None:
            Xte_l.append(Xw); yte_l.append(yw); fte_l.append(fw)
            symte_l.append(np.full(len(yw), sym))
    Xte = np.concatenate(Xte_l); yte = np.concatenate(yte_l)
    fte = np.concatenate(fte_l)
    symte = np.concatenate(symte_l)
    del Xte_l, yte_l, fte_l
    model = train_cnn(Xtr.astype("float32"), ytr, epochs=2)
    del Xtr
    model.eval()
    prs = []
    with torch.no_grad():
        for i in range(0, len(Xte), 65536):
            xb = torch.tensor(Xte[i:i + 65536], device=DEV)
            prs.append(model(xb).softmax(1)[:, 1].cpu().numpy())
    pn = np.concatenate(prs)
    del Xte
    print(f"    диагностика сети: std(p)={pn.std():.5f}, "
          f"min={pn.min():.4f}, max={pn.max():.4f}", flush=True)
    acc_n = float(((pn > 0.5) == (yte > 0.5)).mean() * 100)
    q20n, q80n = np.quantile(pn, [0.2, 0.8])
    sk_n = float((fte[pn >= q80n].mean()
                  - fte[pn <= q20n].mean()) * 10000)
    pooled["tree_acc"].append(acc_t); pooled["tree_sk"].append(sk_t)
    pooled["nn_acc"].append(acc_n); pooled["nn_sk"].append(sk_n)
    pooled["n_folds"] += 1
    if acc_n > acc_t:
        pooled["tree_acc_folds"] += 1
    report["folds"][qq] = {
        "tree": {"acc": round(acc_t, 2), "skill_bp": round(sk_t, 2)},
        "nn": {"acc": round(acc_n, 2), "skill_bp": round(sk_n, 2),
               "n_windows": int(len(yte))}}
    print(f"фолд {qq}: TREE acc {acc_t:.2f}% sk {sk_t:+.1f}бп | NN acc "
          f"{acc_n:.2f}% sk {sk_n:+.1f}бп | {time.time()-t0:.0f}с",
          flush=True)

pooled_avg = {k: (round(float(np.mean(v)), 2) if len(v) else None)
              for k, v in pooled.items() if isinstance(v, list)}
tree_sk_avg = np.mean(pooled["tree_sk"]) if pooled["tree_sk"] else 0
nn_sk_avg = np.mean(pooled["nn_sk"]) if pooled["nn_sk"] else 0
hint = bool(nn_sk_avg - tree_sk_avg >= 1.0
            or pooled["tree_acc_folds"] >= 5)
report["pooled"] = pooled_avg
report["hint"] = hint
print(f"ПОУЛ: TREE sk {tree_sk_avg:+.2f}бп | NN sk {nn_sk_avg:+.2f}бп | "
      f"NN точнее в {pooled['tree_acc_folds']}/{pooled['n_folds']} фолдов "
      f"-> {'НАМЁК ЕСТЬ' if hint else 'намёка нет'}", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ПРОБА NN-СВЕЧИ ЗАВЕРШЕНА", flush=True)
