"""СВЕЖИЙ ОПЫТ ВЛАДЕЛЬЦА (12.10, его словами): «есть история всех монет;
накинь индикаторы; корми модель частями и обучай её с подтверждением;
посмотрим что будет». Ничего от существующей системы: ни первой модели,
ни её сигналов. Просто история (520 монет, 30м свечи, 2020-2026) +
24 индикатора.

ПРОТОКОЛ «ЧАСТЯМИ С ПОДТВЕРЖДЕНИЕМ»: время идёт по месяцам. Модель
учится на последних 6 месяцах, предсказывает СЛЕДУЮЩИЙ месяц (не
виденный), месяц прирастает — предсказал, узнал правду, поправился.
Параллельно тот же протокол у дерева (переучивается раз в 3 месяца,
предсказывает каждый месяц).

ВОПРОС: закроется ли цена через 10 баров (5ч) выше или ниже.
СЕТЬ: обычная плотная сеть (24 признака -> 128 -> 64 -> 2), GPU.
ДЕРЕВО: LightGBM, как в бою.

СУДЬЯ (записано заранее): на пуле всех невиданных месяцев
(1) точность против монетки; (2) «уверенность в деньгах» = средний ход
у самых уверенных 20% прогнозов минус средний ход у самых неуверенных
20% (в базисных пунктах, ход цена-к-цене). НАМЁК = у сети эта разница
>= +10 бп И на >= 5 бп лучше дерева. Иначе — намёка нет.
ДОПОЛНИТЕЛЬНО (вопрос владельца про комбо): торгуем только где обе
модели уверены в одну сторону (порог 0.60); намёк = комбо приносит
больше каждой модели в одиночку при n >= 50000.
Оговорка: сети уже трижды проиграли дереву на других протоколах; это
первая проба «частями с подтверждением».
"""
import glob, io, json, time, zipfile, zlib
import numpy as np, pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn

t0 = time.time()
H = 10
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("устройство:", DEV, flush=True)

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
    return X

frames = []
for i, name in enumerate(members, 1):
    sym = name.replace("_30m.csv", "")
    df = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["open_time", "high", "low", "close",
                              "volume", "quote_volume", "taker_buy_base"])
    for c in df.columns:
        if c != "open_time":
            df[c] = df[c].astype("float32")
    df["open_time"] = df["open_time"].astype(np.int64)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    X = make_feats(df, btc, sym)
    close = df["close"].values.astype("float64")
    fut = np.full(len(df), np.nan, dtype="float64")
    fut[:-H] = close[H:]
    X["fwd"] = (fut / close - 1).astype("float32")
    X["ot"] = df["open_time"].values.astype(np.int64)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna())
    if i % 100 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
FEATS = [c for c in data.columns if c not in ("fwd", "ot")]
data["month"] = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"),
                               freq="M").astype(str)
_before = len(data)
data = data.dropna(subset=FEATS).reset_index(drop=True)   # BTC-строки:
                                       # у биткоина нет btc-признаков,
                                       # после склейки они пустые — сети
                                       # нельзя, дереву было всё равно
print(f"чистка: {_before} -> {len(data)}", flush=True)
print(f"строк {len(data)}, месяцев {data['month'].nunique()} | "
      f"{time.time()-t0:.0f}с", flush=True)

months = sorted(data["month"].unique())
WARMUP = 6                            # первые 6 месяцев — стартовый запас
eval_months = months[WARMUP:]
Xall = data[FEATS].values.astype("float32")
fall = data["fwd"].values.astype("float32")
mall = data["month"].values
mu0 = np.concatenate([Xall[mall == m] for m in months[:WARMUP]]) \
    .mean(0, keepdims=True)
sd0 = np.concatenate([Xall[mall == m] for m in months[:WARMUP]]) \
    .std(0, keepdims=True) + 1e-6
Xn = np.clip((Xall - mu0) / sd0, -10.0, 10.0)   # обрезка крайностей:
                                                 # иначе сеть захлёбывается

class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 2))
    def forward(self, x):
        return self.net(x)

def fit_net(rows):
    net = Net(len(FEATS)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xr = torch.tensor(Xn[rows], device=DEV)
    yr = torch.tensor((fall[rows] > 0).astype("int64"), device=DEV)
    for ep in range(2):
        net.train()
        perm = torch.randperm(len(Xr), device=DEV)
        for i in range(0, len(Xr), 8192):
            b = perm[i:i + 8192]
            opt.zero_grad()
            loss = lossf(net(Xr[b]), yr[b])
            loss.backward()
            opt.step()
    net.eval()
    return net

def predict_net(net, rows):
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 262144):
            Xb = torch.tensor(Xn[rows[i:i + 262144]], device=DEV)
            out.append(net(Xb).softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])

midx = {m: np.nonzero(mall == m)[0] for m in months}
tree = None
tree_age = 99
p_net = np.full(len(data), np.nan, dtype="float32")
p_tree = np.full(len(data), np.nan, dtype="float32")
for m in eval_months:
    m_rows = midx[m]
    mi = months.index(m)
    tr_months = months[max(0, mi - 6): mi]
    tr_rows = np.concatenate([midx[x] for x in tr_months])
    net = fit_net(tr_rows)
    p_net[m_rows] = predict_net(net, m_rows)
    if m == eval_months[0]:
        p1 = p_net[m_rows]
        nan_par = any(bool(np.isnan(p.detach().cpu().numpy()).sum())
                      for p in net.parameters())
        print(f"ДИАГНОСТИКА первый месяц {m}: строк {len(m_rows)}, "
              f"NaN прогнозов {int(np.isnan(p1).sum())}, std "
              f"{float(np.nanstd(p1)):.4f}, NaN в весах сети {nan_par}, "
              f"NaN в Xn(train) {int(np.isnan(Xn[tr_rows]).sum())}",
              flush=True)
    del net
    if DEV == "cuda":
        torch.cuda.empty_cache()
    if tree is None or tree_age >= 3:
        trd = data.iloc[tr_rows]
        tree = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05,
                                  max_depth=5, subsample=0.8,
                                  colsample_bytree=0.8, random_state=42,
                                  n_jobs=4, verbosity=-1)
        tree.fit(trd[FEATS], (trd["fwd"] > 0).astype("int64"))
        tree_age = 0
    p_tree[m_rows] = tree.predict_proba(Xall[m_rows])[:, 1]
    tree_age += 1
    print(f"месяц {m}: сеть и дерево отработали | {time.time()-t0:.0f}с",
          flush=True)

def judge(p):
    ok = ~np.isnan(p)
    if ok.sum() < 1000:
        return {"n": int(ok.sum()), "acc": None, "skill_bp": None,
                "top20_after_costs_bp": None}
    acc = float(((p[ok] > 0.5) == (fall[ok] > 0)).mean() * 100)
    q20, q80 = np.quantile(p[ok], [0.2, 0.8])
    f = fall[ok]
    skill = float((f[p[ok] >= q80].mean() - f[p[ok] <= q20].mean()) * 1e4)
    top_after = float(f[p[ok] >= q80].mean() * 1e4 - 10.0)
    return {"n": int(ok.sum()), "acc": round(acc, 2),
            "skill_bp": round(skill, 2),
            "top20_after_costs_bp": round(top_after, 2)}

v_net = judge(p_net)
v_tree = judge(p_tree)
print(f"ДИАГНОСТИКА: валидных прогнозов сети {v_net['n']} из {len(data)}, "
      f"дерева {v_tree['n']}", flush=True)
sk_n = v_net["skill_bp"] if v_net["skill_bp"] is not None else -999.0
sk_t = v_tree["skill_bp"] if v_tree["skill_bp"] is not None else -999.0
hint = bool(sk_n >= 10.0 and sk_n >= sk_t + 5.0)
report = {"pre_reg": "НАМЁК: сеть skill >= +10бп И на >=5бп лучше дерева",
          "net": v_net, "tree": v_tree, "hint": hint,
          "per_month": {}}
for m in eval_months:
    r = midx[m]
    ok = ~np.isnan(p_net[r])
    if ok.sum() < 1000:
        continue
    f = fall[r][ok]
    q20, q80 = np.quantile(p_net[r][ok], [0.2, 0.8])
    report["per_month"][m] = {
        "acc": round(float(((p_net[r][ok] > 0.5) == (f > 0)).mean() * 100), 2),
        "skill_bp": round(float((f[p_net[r][ok] >= q80].mean()
                                 - f[p_net[r][ok] <= q20].mean()) * 1e4), 1)}

# --- КОМБО: торгуем только где УВЕРЕНЫ ОБЕ и смотрят в одну сторону ---
def trade_money(p, thr):
    """Деньги уверенных сделок: уверены вверх -> лонг, вниз -> шорт.
    Возвращает (n, доля от всех, средняя в бп после 10бп издержек)."""
    ok = ~np.isnan(p)
    pr, f = p[ok], fall[ok]
    up, dn = pr >= thr, pr <= 1 - thr
    pnl = np.where(up, f * 1e4, np.where(dn, -f * 1e4, np.nan)) - 10.0
    sel = up | dn
    if sel.sum() == 0:
        return 0, 0.0, float("nan")
    return int(sel.sum()), float(sel.mean() * 100), float(pnl[sel].mean())

ok_both = ~np.isnan(p_net) & ~np.isnan(p_tree)
pn, pt = p_net[ok_both], p_tree[ok_both]
fb = fall[ok_both]
up_both = (pn >= 0.6) & (pt >= 0.6)
dn_both = (pn <= 0.4) & (pt <= 0.4)
combo = up_both | dn_both
pnl_combo = np.where(up_both, fb * 1e4,
                     np.where(dn_both, -fb * 1e4, np.nan)) - 10.0
n_net, sh_net, m_net = trade_money(p_net, 0.6)
n_tree, sh_tree, m_tree = trade_money(p_tree, 0.6)
either = ((pn >= 0.6) | (pn <= 0.4)) | ((pt >= 0.6) | (pt <= 0.4))
agree_rate = float(combo.sum() / max(1, ((pn >= 0.6) | (pn <= 0.4)) &
            ((pt >= 0.6) | (pt <= 0.4)) | combo.sum() + 0) ) if False else None
both_conf = ((pn >= 0.6) | (pn <= 0.4)) & ((pt >= 0.6) | (pt <= 0.4))
same_dir = (up_both | dn_both) & both_conf
overlap = float(same_dir.sum() / max(1, both_conf.sum()) * 100)
report["combo"] = {
    "moments": int(ok_both.sum()),
    "combo_n": int(combo.sum()),
    "combo_share_pct": round(float(combo.sum() / max(1, ok_both.sum()) * 100), 1),
    "combo_avg_bp": round(float(np.nanmean(pnl_combo[combo])), 2)
    if combo.sum() else None,
    "net_alone": {"n": n_net, "share_pct": round(sh_net, 1),
                  "avg_bp": round(m_net, 2)},
    "tree_alone": {"n": n_tree, "share_pct": round(sh_tree, 1),
                   "avg_bp": round(m_tree, 2)},
    "overlap_pct": round(overlap, 1),
    "both_conf_n": int(both_conf.sum()),
    "pre_reg": "намёк: комбо > каждой в одиночку и n >= 50000"}
cand = bool(combo.sum() >= 50000
            and report["combo"]["combo_avg_bp"] is not None
            and report["combo"]["combo_avg_bp"] > max(m_net, m_tree))
report["combo"]["hint"] = cand
print(f"КОМБО: моментов {int(ok_both.sum())}, обе уверены в одну сторону "
      f"{int(combo.sum())} ({report['combo']['combo_share_pct']}%), "
      f"средняя {report['combo']['combo_avg_bp']}бп | "
      f"сеть одна: {m_net:+.2f}бп (n={n_net}) | дерево одно: "
      f"{m_tree:+.2f}бп (n={n_tree}) | совпадение уверенных "
      f"{overlap:.0f}% -> {'НАМЁК' if cand else 'нет'}", flush=True)

print("СЕТЬ:", v_net, flush=True)
print("ДЕРЕВО:", v_tree, flush=True)
print("ВЕРДИКТ:", "НАМЁК ЕСТЬ" if hint else "намёка нет", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ОПЫТ ЗАВЕРШЁН", flush=True)
