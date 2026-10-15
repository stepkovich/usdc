"""ТРЕТИЙ ВЗГЛЯД v0 (15.10, идея владельца: нейронка на НАШЕЙ еде).
ВОПРОС: предсказывает ли микроструктура стакана направление на 15/30 мин.
ЕДА: feat_*.parquet — лестница 20 уровней, дисбалансы, потоки сделок,
микро/спред, VPIN; 36 USDC-монет, 25.09-15.10, ~40М строк, снапшот ~1.3с.
СЭМПЛИНГ: раз в ~50с на монету -> ~1М примеров.
ХОДЬБА: train < день D, тест = D; D = последние 6 дней.
МОДЕЛИ: LightGBM (референс) и MLP (сеть, GPU).
МЕТКА: fwd = mid(t+H)/mid(t)-1, H = 15мин и 30мин.
СКИЛЛ: по уверенным (top-30% |pred|): mean(sign(pred)*fwd) минус
издержки 4.5бп (вход мейкер 0 + выход тейкер на USDC).
ПРЕДРЕГИСТРАЦИЯ: третий взгляд состоялся, если хотя бы одна модель
держит >+10 бп после издержек при n>=2000 уверенных; сеть интересна,
если >= дерева на +5 бп. ВЕРДИКТ НЕ ДЕПЛОИТСЯ — это разведка еды.
"""
import glob, json, os, time
import numpy as np, pandas as pd
import lightgbm as lgb
import torch, torch.nn as nn

t0 = time.time()
HORIZONS = {"15m": 900, "30m": 1800}
TOP = 0.30
COST_BP = 4.5

cols = ["ts", "symbol", "mid", "spread_bp", "imb1", "imb5", "imb10", "imb20",
        "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell", "ntr10",
        "vpin10", "slope_b", "slope_a", "wall_b", "wall_a"]
for i in range(1, 6):
    cols += [f"bid_q_{i}", f"ask_q_{i}"]
files = sorted(glob.glob("/kaggle/input/usdc-lob-features/feat_*.parquet") or
               glob.glob("/kaggle/input/datasets/sewerted/usdc-lob-features/feat_*.parquet"))
print(f"файлов по маске: {len(files)}", flush=True)
if not files:
    for root, dirs, fs in os.walk("/kaggle/input"):
        print(" ", root, "->", len(fs), "файлов, примеры:", fs[:3], flush=True)
    raise SystemExit(1)
frames = []
bad = []
for f in files:
    try:
        d = pd.read_parquet(f, columns=cols)
    except Exception as e:
        if len(bad) < 2:
            bad.append(f"{f.split('/')[-1]}: {type(e).__name__} {str(e)[:150]}")
        continue
    d = d.iloc[::40]                      # ~50с между снапшотами
    frames.append(d)
if bad:
    print("ПРИМЕРЫ ОТКАЗОВ ЧТЕНИЯ:", flush=True)
    for b in bad:
        print("  ", b, flush=True)
if not frames:
    raise SystemExit(1)
df = pd.concat(frames, ignore_index=True)
del frames
df = df.dropna(subset=["mid", "ts"])
df = df.replace([np.inf, -np.inf], np.nan).ffill()
df["day"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
FEATS = [c for c in cols if c not in ("ts", "symbol", "mid", "day")]
print(f"строк после сэмплинга: {len(df)} | {time.time()-t0:.0f}с", flush=True)

# метки: fwd по каждому символу через поиск по времени
parts = []
for sym, g0 in df.groupby("symbol"):
    g = g0.sort_values("ts").copy()
    ts = g["ts"].values
    mid = g["mid"].values
    for name, h in HORIZONS.items():
        idx = np.searchsorted(ts, ts + h * 1000)
        ok = idx < len(ts)
        fwd = np.full(len(g), np.nan)
        fwd[ok] = mid[idx[ok]] / mid[ok] - 1
        g[f"fwd_{name}"] = fwd
    parts.append(g)
df = pd.concat(parts, ignore_index=True)
del parts
df = df.dropna(subset=["fwd_15m", "fwd_30m"])
print(f"с метками: {len(df)} | {time.time()-t0:.0f}с", flush=True)

days = sorted(df["day"].unique())
test_days = days[-6:]
X = df[FEATS].astype("float32").values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
dead = [FEATS[i] for i in range(len(FEATS)) if X[:, i].std() == 0]
if dead:
    print("пустые колонки выброшены:", dead, flush=True)
    keep = [i for i in range(len(FEATS)) if X[:, i].std() > 0]
    FEATS = [FEATS[i] for i in keep]
    X = X[:, keep]
mu, sd = X.mean(0), X.std(0) + 1e-9

def run_fold(D, model_kind, hname):
    tr = (df["day"] < D).values
    te = (df["day"] == D).values
    y = df[f"fwd_{hname}"].values
    if model_kind == "tree":
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=63, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=4,
                              verbosity=-1)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
    else:
        xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
        net = nn.Sequential(nn.Linear(len(FEATS), 256), nn.ReLU(),
                            nn.Linear(256, 128), nn.ReLU(),
                            nn.Linear(128, 64), nn.ReLU(),
                            nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.HuberLoss()
        ds = torch.utils.data.TensorDataset(xt, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=8192, shuffle=True)
        net.train()
        for ep in range(3):
            for xb, yb in dl:
                opt.zero_grad()
                loss = lossf(net(xb), yb)
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            xe = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
            pred = torch.cat([net(xb)[..., 0] for xb in
                              torch.utils.data.DataLoader(
                                  xe, batch_size=65536)]).numpy()
    n = int(te.sum())
    k = max(1, int(n * TOP))
    order = np.argsort(-np.abs(pred))[:k]
    sgn = np.sign(pred)[order]
    skill = float((sgn * y[te][order]).mean() * 10000 - COST_BP)
    return {"day": D, "n_conf": k, "skill_bp": round(skill, 2),
            "acc": round(float((sgn * np.sign(y[te][order]) > 0).mean()), 3)}

report = {"pre_reg": "третий взгляд: хоть одна модель >+10бп после издержек "
                     "при n>=2000; сеть: >= дерева на +5бп",
          "results": {}}
for hname in HORIZONS:
    for kind in ("tree", "net"):
        rows = [run_fold(D, kind, hname) for D in test_days]
        n_all = sum(r["n_conf"] for r in rows)
        avg = float(np.average([r["skill_bp"] for r in rows],
                               weights=[r["n_conf"] for r in rows]))
        report["results"][f"{hname}/{kind}"] = {
            "days": rows, "n_conf_total": n_all, "skill_avg_bp": round(avg, 2)}
        print(f"{hname}/{kind}: скилл {avg:+.2f} бп (n={n_all}) "
              f"| {time.time()-t0:.0f}с", flush=True)

for hname in HORIZONS:
    t = report["results"][f"{hname}/tree"]["skill_avg_bp"]
    n = report["results"][f"{hname}/net"]["skill_avg_bp"]
    nt = report["results"][f"{hname}/tree"]["n_conf_total"]
    report[f"hint_{hname}"] = {
        " vzglyad_est": bool(max(t, n) > 10 and
                             max(report["results"][f"{hname}/tree"]["n_conf_total"],
                                 report["results"][f"{hname}/net"]["n_conf_total"]) >= 2000),
        "net_beshe_dereva": bool(n >= t + 5)}
json.dump(report, open("/kaggle/working/report.json", "w"),
          indent=1, ensure_ascii=False)
print("ТРЕТИЙ ВЗГЛЯД v0 ЗАВЕРШЁН", flush=True)
