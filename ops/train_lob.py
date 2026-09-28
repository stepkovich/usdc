"""Локальный тренер модели стакана: паркет-архивы (rsync с VPS) ->
walk-forward по дням -> models/lob.txt -> scp на сервер."""
from __future__ import annotations

import glob
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

PARQUET_DIR = Path("/home/iek/lob_archive")
OUT = Path("/home/iek/PycharmProjects/USDC/models/lob.txt")
HORIZON_S = 300
LAG_TOL_MS = 2500
SEED = 42
MIN_TEST_DAYS = 3

FEATS_RICH = ["spread_bp", "microprice_rel", "imb5", "imb10", "imb20",
              "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
              "ntr10", "vpin10", "d30", "d120"]


def load_archive() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(PARQUET_DIR / "feat_*.parquet"))):
        frames.append(pd.read_parquet(f))
    df = pd.concat(frames, ignore_index=True) \
        .drop_duplicates(["ts", "symbol"]) \
        .sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    parts = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        ts = g["ts"].values
        mid = g["mid"].values
        idx = ts.searchsorted(ts + HORIZON_S * 1000)
        idx = np.clip(idx, 0, len(g) - 1)
        ok = ts[idx] >= ts + HORIZON_S * 1000 - 1500
        g["mid_fut"] = np.where(ok, mid[idx], np.nan)
        g["y"] = (g["mid_fut"] / mid > 1).astype(float)
        g.loc[g["mid_fut"].isna(), "y"] = np.nan
        for lag_ms, name in ((30_000, "d30"), (120_000, "d120")):
            j = ts.searchsorted(ts - lag_ms)
            j = np.clip(j, 0, len(g) - 1)
            past_ok = np.abs(ts[j] - (ts - lag_ms)) <= LAG_TOL_MS
            past = np.where(past_ok, mid[j], np.nan)
            g[name] = mid / past - 1
        ba = g["ask_sum20"].values
        bq = g["bid_sum20"].values
        tot = bq + ba
        g["microprice_rel"] = (g["microprice"] / g["mid"] - 1) * 10000
        for c in ("flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
                  "ntr10"):
            med = g[c].median()
            g[c] = g[c] / (med if med and med > 0 else 1.0)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    feats = FEATS_RICH
    out = out.dropna(subset=feats + ["y"])
    out["day"] = pd.to_datetime(out["ts"], unit="ms").dt.date
    return out, feats


def main() -> None:
    print("загружаю архив...", flush=True)
    df = load_archive()
    print(f"строк {len(df)}, символов {df['symbol'].nunique()}", flush=True)
    ds, feats = build_dataset(df)
    days = sorted(ds["day"].unique())
    print(f"дней {len(days)}, строк после чистки {len(ds)}", flush=True)
    if len(days) < MIN_TEST_DAYS + 1:
        print(f"мало дней ({len(days)}): нужно >= {MIN_TEST_DAYS+1} — "
              f"пока не обучаю", flush=True)
        sys.exit(2)
    test_days = days[-MIN_TEST_DAYS:]
    rows = []
    for day in test_days:
        tr, te = ds[ds["day"] < day], ds[ds["day"] == day]
        if len(te) < 5000 or len(tr) < 20000:
            continue
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05,
                               max_depth=4, subsample=0.8, colsample_bytree=0.8,
                               random_state=SEED, n_jobs=2, verbosity=-1)
        m.fit(tr[feats], tr["y"])
        p = m.predict_proba(te[feats])[:, 1]
        fwd = te["mid_fut"] / te["mid"] - 1
        q75, q25 = np.quantile(p, 0.75), np.quantile(p, 0.25)
        spread = (fwd[p >= q75].mean() - fwd[p <= q25].mean()) * 10000
        acc = ((p > 0.5) == (te["y"] == 1)).mean() * 100
        rows.append({"day": str(day), "acc": acc, "spread_bp": spread})
        print(f"{day}: acc {acc:.1f}%, чистый спред {spread:+.1f} бп",
              flush=True)
    ok_days = sum(1 for r in rows if r["spread_bp"] > 0)
    print(f"плюсовых тестовых дней: {ok_days}/{len(rows)}", flush=True)
    m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05,
                           max_depth=4, subsample=0.8, colsample_bytree=0.8,
                           random_state=SEED, n_jobs=2, verbosity=-1)
    m.fit(ds[feats], ds["y"])
    m.booster_.save_model(str(OUT))
    print(f"модель -> {OUT}", flush=True)
    import json
    meta = {"trained_at": pd.Timestamp.utcnow().isoformat(),
            "features": feats, "days": str(days[0]) + ".." + str(days[-1]),
            "rows": int(len(ds)), "plus_test_days": f"{ok_days}/{len(rows)}"}
    OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
