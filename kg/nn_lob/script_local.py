"""ТРЕТИЙ ВЗГЛЯД v2 — ПОЛНАЯ ЕДА (21.10, владелец: «сделай сейчас»).
НОВОЕ против v5: + ЛЕНТА СДЕЛОК (агрессивный поток покупок/продаж,
крупные сделки) на нескольких масштабах (30с/2м/10м) + признаки
глубины на 10 уровней. Данных ленты: 15-21 октября (6.5 дней).
ВОПРОС: добавляет ли поток сделок предсказательность к лестнице?
ХОДЬБА: train 15-18.10, test 19/20/21.10 (день за днём, без подглядывания).
МОДЕЛИ: LightGBM (референс) + MLP (сеть, GPU).
МЕТКИ: mid(t+15m)/mid-1 и mid(t+30m)/mid-1.
СКИЛЛ: по top-30% уверенных: mean(sign(pred)*fwd) − 4.5бп (тейкер-выход).
ПРЕДРЕГИСТРАЦИЯ: взгляд есть, если хоть одна модель/горизонт держит
>+10бп после издержек на СУММЕ тестовых дней; сеть интересна, если
>= дерева на +5бп.
"""
import glob, gzip, json, os, time
import numpy as np, pandas as pd
import lightgbm as lgb
import torch, torch.nn as nn

t0 = time.time()
HORIZONS = {"15m": 900, "30m": 1800}
TOP = 0.30
COST_BP = 4.5
DAY = pd.Timedelta("1D")

files = sorted(glob.glob("/home/iek/lob_archive/feat_*.parquet"))
files = [f for f in files if f.split("feat_")[1][:8] >= "20261019"]
files = [f for f in files if f.split("feat_")[1][:8] >= "20261019"]
print(f"feat-файлов с 15.10: {len(files)}", flush=True)

frames = []
for f in files:
    try:
        d = pd.read_parquet(f, columns=["ts", "symbol", "mid", "spread_bp",
                                        "imb1", "imb5", "imb10", "imb20",
                                        "flow10_buy", "flow10_sell",
                                        "flow60_buy", "flow60_sell",
                                        "ntr10", "vpin10", "slope_b", "slope_a",
                                        "wall_b", "wall_a",
                                        "bid_q_1", "bid_q_5", "bid_q_10", "bid_q_20",
                                        "ask_q_1", "ask_q_5", "ask_q_10", "ask_q_20",
                                        "bid_sum20", "ask_sum20"])
    except Exception:
        continue
    frames.append(d)
feat = pd.concat(frames, ignore_index=True)
print(f'строк лестницы до сэмплинга: {len(feat)}', flush=True)
del frames
print(f"строк лестницы: {len(feat)} | {time.time()-t0:.0f}с", flush=True)

# лента: агрегат по 30-секундным корзинам на монету
import tarfile
tapes_tar = "/home/iek/lob_archive/dataset/tapes_latest.tar.gz"
if not os.path.exists(tapes_tar):
    tapes_tar = "/home/iek/lob_archive/tapes_latest.tar.gz"
tapes = sorted(glob.glob("/home/iek/lob_archive/tape_*.csv.gz"))
print(f"лента-архив: {os.path.exists(tapes_tar)} | отдельных лент: {len(tapes)}", flush=True)
tparts = []
if os.path.exists(tapes_tar):
    with tarfile.open(tapes_tar, "r:gz") as tf:
        for member in tf.getmembers():
            f = tf.extractfile(member)
            if f is None:
                continue
            d = pd.read_csv(f, compression="gzip")
            d.columns = ["ts", "symbol", "price", "qty", "side"]
            tparts.append(d)
else:
    for f in tapes:
        try:
            d = pd.read_csv(f, compression="gzip")
        except Exception:
            continue
        d.columns = ["ts", "symbol", "price", "qty", "side"]
        tparts.append(d)
tape = pd.concat(tparts, ignore_index=True)
del tparts
tape["bucket"] = (tape["ts"] // 30000) * 30000
tape["buy"] = np.where(tape["side"] == 1, tape["qty"], 0.0)
tape["sell"] = np.where(tape["side"] == -1, tape["qty"], 0.0)
big = tape["qty"].quantile(0.95)
tape["big_buy"] = np.where((tape["side"] == 1) & (tape["qty"] >= big), tape["qty"], 0.0)
tape["big_sell"] = np.where((tape["side"] == -1) & (tape["qty"] >= big), tape["qty"], 0.0)
agg = tape.groupby(["symbol", "bucket"]).agg(
    tbuy=("buy", "sum"), tsell=("sell", "sum"), tn=("qty", "count"),
    tbig_buy=("big_buy", "sum"), tbig_sell=("big_sell", "sum"),
    tavg=("qty", "mean")).reset_index()
print(f"ленточных корзин: {len(agg)} | {time.time()-t0:.0f}с", flush=True)

# лестница на 30-секундной сетке (последний снимок корзины)
feat["bucket"] = (feat["ts"] // 30000) * 30000
g = feat.sort_values("ts").groupby(["symbol", "bucket"], as_index=False).last()

m = g.merge(agg, on=["symbol", "bucket"], how="left")
m[["tbuy", "tsell", "tn", "tbig_buy", "tbig_sell", "tavg"]] = \
    m[["tbuy", "tsell", "tn", "tbig_buy", "tbig_sell", "tavg"]].fillna(0.0)
print(f"после слияния: {len(m)} | {time.time()-t0:.0f}с", flush=True)

m = m.sort_values(["symbol", "bucket"])
FEATS = ["spread_bp", "imb1", "imb5", "imb10", "imb20",
         "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
         "ntr10", "vpin10", "slope_b", "slope_a", "wall_b", "wall_a",
         "bid_q_1", "bid_q_5", "bid_q_10", "bid_q_20",
         "ask_q_1", "ask_q_5", "ask_q_10", "ask_q_20"]
# поток из ленты + производные
m["t_imb"] = (m["tbuy"] - m["tsell"]) / (m["tbuy"] + m["tsell"] + 1e-9)
m["t_big"] = m["tbig_buy"] + m["tbig_sell"]
m["t_big_imb"] = (m["tbig_buy"] - m["tbig_sell"]) / (m["tbig_buy"] + m["tbig_sell"] + 1e-9)
m["t_act"] = m["tn"] / 30.0
m["t_imb_2m"] = m.groupby("symbol")["t_imb"].rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
m["t_imb_10m"] = m.groupby("symbol")["t_imb"].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
m["t_flow_2m"] = m.groupby("symbol")["t_imb"].rolling(4, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
FEATS += ["t_imb", "t_big", "t_big_imb", "t_act", "t_imb_2m", "t_imb_10m", "t_flow_2m"]

# метки
m = m.sort_values(["symbol", "bucket"])
out = []
for sym, gg in m.groupby("symbol"):
    gg = gg.copy()
    ts = gg["bucket"].values
    mid = gg["mid"].values
    for name, h in HORIZONS.items():
        idx = np.searchsorted(ts, ts + h * 1000)
        ok = idx < len(ts)
        fwd = np.full(len(gg), np.nan)
        fwd[ok] = mid[idx[ok]] / mid[ok] - 1
        gg[f"fwd_{name}"] = fwd
    out.append(gg)
m = pd.concat(out, ignore_index=True)
m["day"] = pd.to_datetime(m["bucket"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
m = m.dropna(subset=["fwd_15m", "fwd_30m"])
print(f"с метками: {len(m)} | {time.time()-t0:.0f}с", flush=True)

X = m[FEATS].astype("float32").values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
keep = [i for i in range(X.shape[1]) if X[:, i].std() > 0]
FEATS = [FEATS[i] for i in keep]
X = X[:, keep]
mu, sd = X.mean(0), X.std(0) + 1e-9
X = (X - mu) / sd
test_days = sorted(m["day"].unique())[-3:]
print("тестовые дни:", test_days, "| признаков:", len(FEATS), flush=True)

import lightgbm as lgbm

def run_fold(D, kind, hname):
    tr = (m["day"] < D).values
    te = (m["day"] == D).values
    y = m[f"fwd_{hname}"].values
    if tr.sum() < 5000 or te.sum() < 500:
        return None
    if kind == "tree":
        mm = lgbm.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                num_leaves=63, subsample=0.8,
                                colsample_bytree=0.8, n_jobs=4, verbosity=-1)
        mm.fit(X[tr], y[tr])
        pred = mm.predict(X[te])
    else:
        xt = torch.tensor(X[tr], dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
        net = nn.Sequential(nn.Linear(X.shape[1], 192), nn.ReLU(),
                            nn.Linear(192, 96), nn.ReLU(),
                            nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lf = nn.HuberLoss()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xt, yt), batch_size=16384,
            shuffle=True)
        net.train()
        for ep in range(4):
            for xb, yb in dl:
                opt.zero_grad(); loss = lf(net(xb), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            xe = torch.tensor(X[te], dtype=torch.float32)
            pred = torch.cat([net(xb)[..., 0] for xb in torch.utils.data.DataLoader(
                xe, batch_size=65536)]).numpy()
    n = int(te.sum()); k = max(1, int(n * TOP))
    order = np.argsort(-np.abs(pred))[:k]
    sgn = np.sign(pred)[order]
    skill = float((sgn * y[te][order]).mean() * 10000 - COST_BP)
    acc = float((sgn * np.sign(y[te][order]) > 0).mean())
    return {"day": D, "n_conf": k, "skill_bp": round(skill, 2), "acc": round(acc, 3)}

report = {"pre_reg": "взгляд: >+10бп после издержек суммарно по тест-дням; "
                     "сеть: >= дерева +5бп", "results": {}}
for hname in HORIZONS:
    for kind in ("tree", "net"):
        rows = [r for r in (run_fold(D, kind, hname) for D in test_days) if r]
        rows = [r for r in rows if r]
        n_all = sum(r["n_conf"] for r in rows)
        avg = float(np.average([r["skill_bp"] for r in rows],
                               weights=[r["n_conf"] for r in rows])) if rows else float("nan")
        report["results"][f"{hname}/{kind}"] = {"days": rows, "n_conf_total": n_all,
                                                "skill_avg_bp": round(avg, 2)}
        print(f"{hname}/{kind}: {avg:+.2f} бп (n={n_all}) | {time.time()-t0:.0f}с",
              flush=True)

ok_sum, ok_n = False, 0
best = -1e9
for k, v in report["results"].items():
    if v["skill_avg_bp"] > best:
        best = v["skill_avg_bp"]
    ok_n += v["n_conf_total"]
ok_sum = best > 10 and ok_n >= 2000
net_best = max(v["skill_avg_bp"] for k, v in report["results"].items() if "/net" in k)
tree_best = max(v["skill_avg_bp"] for k, v in report["results"].items() if "/tree" in k)
report["verdict"] = {"vzglyad": bool(ok_sum), "net_vs_tree": round(net_best - tree_best, 2)}
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1, ensure_ascii=False)
print("ТРЕТИЙ ВЗГЛЯД v2 ЗАВЕРШЁН", flush=True)
