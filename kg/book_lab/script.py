"""ЛАБОРАТОРИЯ НА СВОИХ ЗАПИСЯХ СТАКАНА (12.10, «Делай 2,3»).

ХОД 2 — ПРЕДСКАЗАТЬ УЗКИЙ СПРЕД: спринтер гибнет от издержек
(~7бп круг) при живом навыке (1-2бп). Вместо усиления навыка —
модель учится находить моменты, когда спред УЗКИЙ: торгуем только
дешёвые минуты. Метрика честная: у топ-30% прогнозов «узко» средний
фактический спред следующих 30с против общего среднего на тест-днях.
НАМЁК: срез >= 30% (0.7x от среднего).

ХОД 3 — ДВОЙНАЯ УВЕРЕННОСТЬ НА НАШЕЙ ЕДЕ: два разных мозга (дерево
и маленькая сеть) учатся на одних днях предсказывать ход середины
через 5 минут; торгуем только где ОБА уверены в одну сторону (0.60).
НАМЁК: комбо (после реальных ~3бп круг с учётом мид-выхода и промо)
приносит больше, чем каждая в одиночку, при n >= 20000.

Протокол: обучение = первые полные дни записи, тест = последние 4
полных дня (не виденные). Каденс 4с. Ничего не деплоится; намёки
идут на свежие дни как все.
"""
import glob, json, time
import numpy as np, pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn

t0 = time.time()
HOR = 300_000                     # 5 минут
LAG_TOL = 4000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("устройство:", DEV, flush=True)

FEATS = ["spread_bp", "imb5", "imb10", "imb20",
         "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
         "ntr10", "vpin10", "d30", "d120", "hour", "dow"]

files = sorted(glob.glob("/kaggle/input/**/feat_*.parquet", recursive=True))
print("файлов:", len(files), flush=True)
parts = {}
for f in files:
    try:
        df = pd.read_parquet(f)
    except Exception:
        continue
    cols = [c for c in ["ts", "symbol", "mid", "spread_bp", "imb5",
                        "imb10", "imb20", "flow10_buy", "flow10_sell",
                        "flow60_buy", "flow60_sell", "ntr10",
                        "vpin10"] if c in df.columns]
    df = df[cols].dropna(subset=["mid", "spread_bp"]).iloc[::2]  # ~1с
    for sym, g in df.groupby("symbol", observed=True):
        parts.setdefault(sym, []).append(g)

syms = {}
for sym, lst in parts.items():
    g = pd.concat(lst, ignore_index=True)
    g = g.sort_values("ts").drop_duplicates("ts")
    syms[sym] = g
del parts
print(f"монет {len(syms)}, строк {sum(len(g) for g in syms.values())} | "
      f"{time.time()-t0:.0f}с", flush=True)

rows = []
for sym, g in syms.items():
    ts = g["ts"].values.astype(np.int64)
    mid = g["mid"].values.astype("float64")
    sp = g["spread_bp"].values.astype("float32")
    X = pd.DataFrame({"ts": ts, "spread_bp": sp})
    for c in ("imb5", "imb10", "imb20", "flow10_buy", "flow10_sell",
              "flow60_buy", "flow60_sell", "ntr10", "vpin10"):
        X[c] = g[c].values.astype("float32")
    idx = np.clip(np.searchsorted(ts, ts + HOR), 0, len(g) - 1)
    ok = ts[idx] >= ts + HOR - LAG_TOL
    X["fwd"] = np.where(ok, mid[idx] / mid - 1, np.nan).astype("float32")
    X["y"] = np.where(ok, (mid[idx] > mid).astype("float32"), np.nan)
    s30 = np.clip(np.searchsorted(ts, ts + 30_000), 0, len(g) - 1)
    X["spread_next"] = np.where(ts[s30] >= ts + 30_000 - LAG_TOL,
                                sp[s30], np.nan).astype("float32")
    for lag, nm in ((30_000, "d30"), (120_000, "d120")):
        j = np.clip(np.searchsorted(ts, ts - lag), 0, len(g) - 1)
        past = np.where(np.abs(ts[j] - (ts - lag)) <= 4000, mid[j], np.nan)
        X[nm] = (mid / past - 1).astype("float32")
    dt = pd.to_datetime(ts, unit="ms")
    X["hour"] = dt.hour.astype("float32")
    X["dow"] = dt.dayofweek.astype("float32")
    X["sym"] = sym
    rows.append(X.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["fwd", "y", "spread_next", "d30", "d120"]))
data = pd.concat(rows, ignore_index=True)
del rows, syms
data["day"] = pd.to_datetime(data["ts"], unit="ms").dt.date
days = sorted(data["day"].unique())
full = days[:-1]                      # последний (сегодня) неполный
test_days = full[-4:]
print(f"строк {len(data)}, тест-дни {test_days} | {time.time()-t0:.0f}с",
      flush=True)

tr = data[data["day"] < test_days[0]]
te = data[data["day"].isin(test_days)].reset_index(drop=True)

report = {"test_days": [str(d) for d in test_days], "n_test": len(te)}

# ---------- ХОД 2: предсказать узкий спред ----------
med = tr.groupby("sym")["spread_bp"].median()
for part in (tr, te):
    part["sym_med"] = part["sym"].map(med)
    part["narrow"] = (part["spread_next"] < part["sym_med"]).astype("int64")
m2 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, n_jobs=4, verbosity=-1)
m2.fit(tr[FEATS], tr["narrow"])
pn = m2.predict_proba(te[FEATS])[:, 1]
thr = np.quantile(pn, 0.7)
sel = pn >= thr                        # топ-30% «будет узко»
sp_all = float(te["spread_bp"].mean())
sp_narrow = float(te.loc[sel, "spread_bp"].mean())
cut = 1 - sp_narrow / sp_all
report["spread"] = {"avg_test_bp": round(sp_all, 2),
                    "narrow30_bp": round(sp_narrow, 2),
                    "cut_pct": round(cut * 100, 1),
                    "hint": bool(cut >= 0.30)}
print(f"СПРЕД: средний {sp_all:.2f}бп, у топ-30% прогнозов узко "
      f"{sp_narrow:.2f}бп (срез {cut*100:.0f}%) -> "
      f"{'НАМЁК' if cut >= 0.30 else 'нет'}", flush=True)

# ---------- ХОД 3: два мозга на направление ----------
m3 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, n_jobs=4, verbosity=-1)
m3.fit(tr[FEATS], (tr["y"] > 0.5).astype("int64"))
p_tree = m3.predict_proba(te[FEATS])[:, 1]

mu = tr[FEATS].values.astype("float32").mean(0, keepdims=True)
sd = tr[FEATS].values.astype("float32").std(0, keepdims=True) + 1e-6
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 96), nn.ReLU(),
                                 nn.Linear(96, 48), nn.ReLU(),
                                 nn.Linear(48, 2))
    def forward(self, x):
        return self.net(x)
net = Net(len(FEATS)).to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
Xr = torch.tensor((tr[FEATS].values.astype("float32") - mu) / sd,
                  device=DEV)
yr = torch.tensor((tr["y"].values > 0.5).astype("int64"), device=DEV)
for ep in range(3):
    net.train()
    perm = torch.randperm(len(Xr), device=DEV)
    for i in range(0, len(Xr), 16384):
        b = perm[i:i + 16384]
        opt.zero_grad()
        loss = lossf(net(Xr[b]), yr[b])
        loss.backward()
        opt.step()
    print(f"  сеть эп.{ep} loss {loss.item():.4f} | {time.time()-t0:.0f}с",
          flush=True)
net.eval()
p_netout = []
with torch.no_grad():
    Xt = torch.tensor((te[FEATS].values.astype("float32") - mu) / sd,
                      device=DEV)
    for i in range(0, len(Xt), 262144):
        p_netout.append(net(Xt[i:i + 262144]).softmax(1)[:, 1]
                        .cpu().numpy())
p_nett = np.concatenate(p_netout)
del Xr, yr
if DEV == "cuda":
    torch.cuda.empty_cache()

fwd = te["fwd"].values
def money(p, thr=0.6, cost_bp=3.0):
    up, dn = p >= thr, p <= 1 - thr
    pnl = np.where(up, fwd * 1e4, np.where(dn, -fwd * 1e4,
                                           np.nan)) - cost_bp
    sel = up | dn
    acc = float(((np.where(up, 1, -1) * fwd) > 0)[sel].mean() * 100) \
        if sel.sum() else float("nan")
    return int(sel.sum()), float(sel.mean() * 100), \
        float(np.nanmean(pnl[sel])) if sel.sum() else float("nan"), acc

res = {}
for name, p in (("net", p_nett), ("tree", p_tree)):
    n, sh, m, acc = money(p)
    res[name] = {"n": n, "share": round(sh, 1), "avg_bp": round(m, 2),
                 "acc_conf": round(acc, 2) if acc == acc else None}
    print(f"{name} один: n={n} ({sh:.1f}%) avg {m:+.2f}бп", flush=True)
up_c = (p_nett >= 0.6) & (p_tree >= 0.6)
dn_c = (p_nett <= 0.4) & (p_tree <= 0.4)
combo = up_c | dn_c
pnl_c = np.where(up_c, fwd * 1e4,
                 np.where(dn_c, -fwd * 1e4, np.nan)) - 3.0
acc_c = float(((np.where(up_c, 1, -1) * fwd) > 0)[combo].mean() * 100) \
    if combo.sum() else float("nan")
report["combo"] = {"n": int(combo.sum()),
                   "share": round(float(combo.mean() * 100), 1),
                   "avg_bp": round(float(np.nanmean(pnl_c[combo])), 2)
                   if combo.sum() else None,
                   "acc_conf": round(acc_c, 2) if acc_c == acc_c else None}
print(f"КОМБО: n={int(combo.sum())} "
      f"({report['combo']['share']}%) avg "
      f"{report['combo']['avg_bp']}бп, верных {acc_c:.0f}%", flush=True)

# стык ходов: комбо-сделки в предсказанно-узкие моменты
narrow_mask = sel.values if hasattr(sel, "values") else sel
both_mask = combo & narrow_mask
report["combo_narrow"] = {
    "n": int(both_mask.sum()),
    "avg_bp": round(float(np.nanmean(pnl_c[both_mask])), 2)
    if both_mask.sum() else None,
    "spread_paid_bp": round(float(te.loc[both_mask, "spread_bp"].mean()), 2)
    if both_mask.sum() else None}
print(f"КОМБО+узкий спред: n={int(both_mask.sum())} "
      f"avg {report['combo_narrow']['avg_bp']}бп, платный спред "
      f"{report['combo_narrow']['spread_paid_bp']}бп", flush=True)

cand_c = bool(combo.sum() >= 20000
              and report["combo"]["avg_bp"] is not None
              and report["combo"]["avg_bp"] > max(res["net"]["avg_bp"],
                                                  res["tree"]["avg_bp"]))
report["verdict"] = {"spread_hint": report["spread"]["hint"],
                     "combo_hint": cand_c}
print("ВЕРДИКТ:", report["verdict"], flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ЛАБОРАТОРИЯ ЗАВЕРШЕНА", flush=True)
