import glob, os, time, tarfile
import numpy as np, pandas as pd
import lightgbm as lgb
import torch, torch.nn as nn

t0 = time.time()
TOP = 0.30
COST_BP = 4.5

# 1) find feat files
files = []
for root, dirs, fs in os.walk("/kaggle/input"):
    for f in fs:
        if f.startswith("feat_") and f.endswith(".parquet"):
            files.append(os.path.join(root, f))
files = sorted(files)
print(f"feat files: {len(files)}", flush=True)
if not files:
    raise SystemExit(1)

# 2) load + sample + create bucket/day
frames = []
for f in files:
    import gc as _gc
    _cols = [c for c in pd.read_parquet(f, columns=[]).columns if c != "symbol" and not c.startswith("bid_p_") and not c.startswith("ask_p_")]
    d = pd.read_parquet(f, columns=["ts", "symbol", "mid"] + _cols)
    del _cols
    _gc.collect()
    d = d.iloc[::300]
    d["bucket"] = (d["ts"] // 60000) * 60000
    d["day"] = pd.to_datetime(d["bucket"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    frames.append(d)
    del d
    import gc as _gc2; _gc2.collect()
feat = pd.concat(frames, ignore_index=True)
del frames
print(f"rows: {len(feat)} | {time.time()-t0:.0f}s", flush=True)

# 3) load tapes from tar
tar_p = None
for root, dirs, fs in os.walk("/kaggle/input"):
    for f in fs:
        if "tapes_latest" in f:
            tar_p = os.path.join(root, f)
            break
print(f"tape tar: {tar_p}", flush=True)
tparts = []
if tar_p:
    with tarfile.open(tar_p, "r:gz") as tf:
        for mem in tf.getmembers():
            fo = tf.extractfile(mem)
            if fo is None: continue
            d = pd.read_csv(fo, compression="gzip")
            d["bucket"] = (d["ts"] // 60000) * 60000
            d["buy"] = np.where(d["side"] == 1, d["qty"], 0.0)
            d["sell"] = np.where(d["side"] == -1, d["qty"], 0.0)
            a = d.groupby(["symbol", "bucket"]).agg(
                tbuy=("buy", "sum"), tsell=("sell", "sum"), tn=("qty", "count")).reset_index()
            tparts.append(a)
            del d
    import gc as _gc2; _gc2.collect(), a
tape = pd.concat(tparts, ignore_index=True) if tparts else pd.DataFrame(
    columns=["symbol", "bucket", "tbuy", "tsell", "tn"])
del tparts
print(f"tape rows: {len(tape)}", flush=True)

# 4) merge
g = feat.merge(tape[["symbol", "bucket", "tbuy", "tsell", "tn"]],
               on=["symbol", "bucket"], how="left")
for c in ("tbuy", "tsell", "tn"):
    g[c] = g[c].fillna(0)
g["t_imb"] = (g["tbuy"] - g["tsell"]) / (g["tbuy"] + g["tsell"] + 1e-9)

# 5) ALL raw columns as features (log deviations for prices)
DROP = {"ts", "symbol", "bucket", "day", "mid"}
ALL_COLS = [c for c in g.columns if c not in DROP]
for c in [c for c in ALL_COLS if "_p_" in c]:
    g[f"l_{c}"] = np.log(g[c].clip(lower=1e-12) / g["mid"].clip(lower=1e-12))
    ALL_COLS.remove(c)
    ALL_COLS.append(f"l_{c}")
print(f"features: {len(ALL_COLS)}", flush=True)

# 6) labels: 15m and 30m forward
g = g.sort_values(["symbol", "bucket"])
out = []
for sym, gg in g.groupby("symbol"):
    gg = gg.sort_values("bucket").copy()
    ts = gg["bucket"].values
    mid = gg["mid"].values
    for name, h in (("15m", 900), ("30m", 1800)):
        idx = np.searchsorted(ts, ts + h * 1000)
        ok = idx < len(ts)
        fwd = np.full(len(gg), np.nan)
        fwd[ok] = mid[idx[ok]] / mid[ok] - 1
        gg[f"fwd_{name}"] = fwd
    out.append(gg)
m = pd.concat(out, ignore_index=True)
m = m.dropna(subset=["fwd_15m", "fwd_30m"])
print(f"with labels: {len(m)} | {time.time()-t0:.0f}s", flush=True)

# 7) features matrix
X = m[ALL_COLS].values.astype("float32")
X = np.nan_to_num(X, nan=0.0)
mu, sd = X.mean(0), X.std(0) + 1e-9
X = (X - mu) / sd
test_days = sorted(m["day"].unique())[-4:]
print(f"test days: {test_days}", flush=True)

# 8) walk-forward
def run_fold(D, kind, hname):
    tr = (m["day"] < D).values
    te = (m["day"] == D).values
    y = m[f"fwd_{hname}"].values
    if tr.sum() < 5000 or te.sum() < 500:
        return None
    if kind == "tree":
        mm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                               num_leaves=63, subsample=0.8,
                               colsample_bytree=0.8, n_jobs=4, verbosity=-1)
        mm.fit(X[tr], y[tr])
        pred = mm.predict(X[te])
    else:
        xt = torch.tensor(X[tr], dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
        net = nn.Sequential(
            nn.Linear(X.shape[1], 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
        lf = nn.HuberLoss()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xt, yt), batch_size=8192, shuffle=True)
        net.train()
        for ep in range(6):
            for xb, yb in dl:
                opt.zero_grad(); loss = lf(net(xb), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            xe = torch.tensor(X[te], dtype=torch.float32)
            preds = []
            for i in range(0, len(xe), 65536):
                preds.append(net(xe[i:i+65536])[..., 0])
            pred = torch.cat(preds).numpy()
    n = int(te.sum()); k = max(1, int(n * TOP))
    order = np.argsort(-np.abs(pred))[:k]
    sgn = np.sign(pred)[order]
    skill = float((sgn * y[te][order]).mean() * 10000 - COST_BP)
    acc = float((sgn * np.sign(y[te][order]) > 0).mean())
    return {"day": D, "n_conf": k, "skill_bp": round(skill, 2), "acc": round(acc, 3)}

report = {"pre_reg": "third view v3: >+10bp on raw orderbook+tape", "results": {}}
for hname in ("15m", "30m"):
    for kind in ("tree", "net"):
        rows = [r for r in (run_fold(D, kind, hname) for D in test_days) if r]
        rows = [r for r in rows if r]
        n_all = sum(r["n_conf"] for r in rows)
        avg = float(np.average([r["skill_bp"] for r in rows],
                               weights=[r["n_conf"] for r in rows])) if rows else 0
        report["results"][f"{hname}/{kind}"] = {"days": rows,
                                                "n_conf_total": n_all,
                                                "skill_avg_bp": round(avg, 2)}
        print(f"{hname}/{kind}: {avg:+.2f} bp (n={n_all}) | {time.time()-t0:.0f}s",
              flush=True)

best_net = max(v["skill_avg_bp"] for k, v in report["results"].items() if "/net" in k)
best_tree = max(v["skill_avg_bp"] for k, v in report["results"].items() if "/tree" in k)
report["verdict"] = {"third_view": bool(max(best_net, best_tree) > 10),
                     "net_vs_tree": round(best_net - best_tree, 2)}
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1, ensure_ascii=False)
print("THIRD VIEW v3 COMPLETE", flush=True)
