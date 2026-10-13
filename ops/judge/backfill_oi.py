"""Разовая приклейка ОИ-топлива к уже решённым сигналам судьи."""
import io, json, sys, zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd, urllib.request

JDIR = Path("/home/iek/PycharmProjects/USDC/ops/judge")
MROOT = JDIR / "metrics"
MCOLS = ["create_time", "sum_open_interest",
         "count_toptrader_long_short_ratio",
         "sum_toptrader_long_short_ratio",
         "sum_taker_long_short_vol_ratio"]

def dl(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception:
        return None

def metrics_frame(sym, days):
    parts = []
    for d in days:
        fn = MROOT / sym / f"{d}.csv"
        if not fn.exists():
            raw = dl("https://data.binance.vision/data/futures/um/"
                     f"daily/metrics/{sym}/{sym}-metrics-{d}.zip")
            if raw is None:
                continue
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw))
                fn.parent.mkdir(parents=True, exist_ok=True)
                fn.write_bytes(zf.read(zf.namelist()[0]))
            except Exception:
                continue
        try:
            parts.append(pd.read_csv(fn, usecols=MCOLS))
        except Exception:
            continue
    if not parts:
        return None
    m = pd.concat(parts, ignore_index=True)
    m["t"] = (pd.to_datetime(m["create_time"], utc=True)
              - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1ms")
    return m.drop_duplicates("t").sort_values("t")

def oi_feats(m, ot):
    t = m["t"].values
    j = int(np.searchsorted(t, ot))
    if j >= len(t) or t[j] - ot > 10 * 60_000:
        return None
    r = m.iloc[j]
    v_now = float(r["sum_open_interest"])
    out = {}
    for lag_ms, nm in ((3600_000, "1h"), (6 * 3600_000, "6h")):
        k = int(np.searchsorted(t, ot - lag_ms))
        if k >= len(t) or t[k] < ot - lag_ms - 10 * 60_000:
            out[f"oi_chg_{nm}"] = np.nan
        else:
            v0 = float(m.iloc[k]["sum_open_interest"])
            out[f"oi_chg_{nm}"] = (v_now / v0 - 1) if v0 > 0 else np.nan
    out["fuel"] = bool(out["oi_chg_6h"] > 0) \
        if out["oi_chg_6h"] == out["oi_chg_6h"] else None
    return out

rows = [json.loads(l) for l in open(JDIR / "results.jsonl")]
frames = {}
changed = 0
for r in rows:
    d0 = datetime.fromtimestamp(r["ot"] / 1000, tz=timezone.utc).date()
    days = [str(d0 + timedelta(days=k)) for k in (-2, -1, 0, 1)]
    if r["sym"] not in frames:
        frames[r["sym"]] = metrics_frame(r["sym"], days)
    m = frames.get(r["sym"])
    if m is None:
        r["fuel"] = None
        continue
    f = oi_feats(m, r["ot"])
    if f:
        r["oi_chg_1h"] = round(f["oi_chg_1h"], 6) \
            if f["oi_chg_1h"] == f["oi_chg_1h"] else None
        r["oi_chg_6h"] = round(f["oi_chg_6h"], 6) \
            if f["oi_chg_6h"] == f["oi_chg_6h"] else None
        r["fuel"] = f["fuel"]
    changed += 1
    if changed % 50 == 0:
        print(f"{changed}/{len(rows)}", flush=True)

import shutil
shutil.copy(JDIR / "results.jsonl", JDIR / "results.jsonl.bak")
with open(JDIR / "results.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

df = pd.DataFrame(rows)
print("\n=== ОИ-ТОПЛИВО НА 1-11 ОКТЯБРЯ ===")
for side, nm in ((1, "ЛОНГИ"), (-1, "шорты")):
    d = df[(df.side == side) & df.fuel.notna()]
    if len(d) == 0:
        continue
    f = d[d.fuel == True]   # noqa: E712
    nf = d[d.fuel != True]
    print(f"{nm}: с топливом n={len(f)} {f.pnl_bp.mean():+.1f}бп "
          f"(верных {float((f.pnl_bp>0).mean()*100):.0f}%) | "
          f"без n={len(nf)} "
          f"{nf.pnl_bp.mean() if len(nf) else float('nan'):+.1f}бп")
allf = df[df.fuel == True]  # noqa: E712
alln = df[df.fuel != True]
print(f"ВСЕ: топливо n={len(allf)} {allf.pnl_bp.mean():+.1f}бп | "
      f"без n={len(alln)} {alln.pnl_bp.mean():+.1f}бп")
print(f"покрыто ОИ: {int(df.fuel.notna().sum())} из {len(df)}")
