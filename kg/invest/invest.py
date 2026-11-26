import json, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/repo")
from ml5h.features import make_features, g_cross, sym_id  # noqa
import lightgbm as lgb

META = json.load(open("/repo/models/ml5h_meta.json"))
FEATS = list(META["features"])
GATE = float(META.get("gate", 0.65))
model = lgb.Booster(model_file="/repo/models/ml5h.txt")
cal = json.load(open("/repo/models/calibration.json"))
CX = np.array(cal["iso_x"]); CY = np.array(cal["iso_y"])

USECOLS = ["open_time", "high", "low", "close", "volume",
           "quote_volume", "taker_buy_base"]
NEED = 1500
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
PANEL = Path("/repo/data_cache_30m")
KC = Path("/repo/ops/judge/klines")
PANEL_FROM = date(2026, 9, 27)

def read_daily(sym, d):
    fn = KC / sym / f"{d.isoformat()}.csv"
    if not fn.exists(): return None
    try:
        return pd.read_csv(fn, header=None, usecols=[0,2,3,4,5,7,9],
                           names=["open_time","high","low","close",
                                  "volume","quote_volume","taker_buy_base"])
    except Exception: return None

def load_sym(sym):
    parts = []
    pc = PANEL / f"{sym}_30m.csv"
    if pc.exists():
        try: parts.append(pd.read_csv(pc, usecols=USECOLS))
        except Exception: pass
    d = PANEL_FROM
    end = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    while d <= end:
        df = read_daily(sym, d)
        if df is not None: parts.append(df)
        d += timedelta(days=1)
    if not parts: return None
    df = pd.concat(parts, ignore_index=True)
    for c in USECOLS[1:]: df[c] = pd.to_numeric(df[c], errors="coerce")
    ot = pd.to_numeric(df["open_time"], errors="coerce").astype("float64")
    ot = np.where(ot > 1e14, ot/1000.0, ot)
    df["open_time"] = ot.astype("int64")
    df = df.dropna(subset=["close"]).drop_duplicates("open_time") \
           .sort_values("open_time").tail(NEED)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    return df if len(df) >= 300 else None

_cache = {}
def sym_df(sym):
    if sym not in _cache: _cache[sym] = load_sym(sym)
    return _cache[sym]

btc = sym_df("BTCUSDT")
btc_close = btc["close"] if btc is not None else None

def feats_row(sym, ts_ms):
    """Признаки и p для бара, закрывающегося в ts_ms (open_time = ts-30min)."""
    df = sym_df(sym)
    if df is None: return None, None
    otv = df["open_time"].values
    j = int(np.searchsorted(otv, ts_ms - 1800_000))
    if j >= len(otv) or otv[j] != ts_ms - 1800_000: return None, None
    X = make_features(df, btc_close, sym).replace([np.inf,-np.inf], np.nan)
    Xg = g_cross(df, btc_close, sym)
    for c in Xg.columns: X[c] = Xg[c]
    X["sym_id"] = float(sym_id(sym))
    if X[FEATS].iloc[j].isna().any(): return None, None
    x = np.array([[float(X[f].iloc[j]) for f in FEATS]])
    return float(model.predict(x)[0]), j

def honest_outcome(sym, j):
    df = sym_df(sym)
    cls = df["close"].values; lo = df["low"].values; hi = df["high"].values
    c0 = cls[j]; stop = c0 * 0.85
    for k in range(j+1, min(j+11, len(cls))):
        if lo[k] <= stop: return stop, True, c0
    return cls[min(j+10, len(cls)-1)], False, c0

W = json.load(open("/repo/kg/invest/week_log.json"))
print("=== 1. ЦЕНА ЧЕСТНОГО ГЕЙТА (128 отсечённых лонгов) ===")
tot_bp, found, unresolved, sides = 0.0, 0, 0, {}
per_day = {}
for ts, sym, chance in W["honest"]:
    # бар закрылся в границе <= ts: candidates = две последние границы
    cands = []
    for boundary in ((ts // 1800000) * 1800000, ((ts // 1800000) - 1) * 1800000):
        p, j = feats_row(sym, boundary)
        if p is None: continue
        t = float(np.interp(p, CX, CY)) * 100
        cands.append((abs(t - chance), boundary, p, t, j))
    if not cands:
        unresolved += 1; continue
    cands.sort()
    diff, boundary, p, t, j = cands[0]
    if diff > 1.6:
        unresolved += 1; continue
    exit_px, stopped, c0 = honest_outcome(sym, j)
    pnl = (exit_px / c0 - 1) * 1e4 - 10.0 - (12.0 if stopped else 0.0)
    tot_bp += pnl; found += 1
    d = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%d.%m")
    per_day.setdefault(d, [0, 0.0]); per_day[d][0] += 1; per_day[d][1] += pnl
print(f"опознано {found}/128 (не опознано {unresolved})")
print(f"суммарный исход отсечённых: {tot_bp:+.0f} бп = {tot_bp*15/1e4:+.2f} USDT (номинал 15)")
if found: print(f"в среднем {tot_bp/found:+.1f} бп на сигнал")
for d in sorted(per_day):
    n, s = per_day[d]
    print(f"  {d}: {n} шт, {s:+.0f} бп ({s*15/1e4:+.2f} USDT)")

print("\n=== 2. ПРОСКАЛЬЗЫВАНИЕ ВХОДА (реальные сделки бота) ===")
slips = []
for ts, sym, side, pnl, conf, epx in W["trades"]:
    df = sym_df(sym)
    if df is None: continue
    boundary = (ts // 1800000) * 1800000
    otv = df["open_time"].values
    j = int(np.searchsorted(otv, boundary - 1800000))
    if j >= len(otv) or otv[j] != boundary - 1800000: continue
    sc = float(df["close"].values[j])
    s = 1 if side == "LONG" else -1
    slips.append(s * (epx / sc - 1) * 1e4)
if slips:
    import statistics
    print(f"сделок {len(slips)} | среднее отклонение входа от закрытия бара: "
          f"{statistics.mean(slips):+.1f} бп | медиана {statistics.median(slips):+.1f} бп")

print("\n=== 3. ПЕРЕЗАХОДЫ БЕЗ КУЛДАУНА (монета повторно в течение 5ч) ===")
tr = sorted(W["trades"], key=lambda x: x[0])
last = {}
re_pnl = iso_pnl = re_n = iso_n = 0
for ts, sym, side, pnl, conf, epx in tr:
    if ts - last.get(sym, -10**18) < 5 * 3600_000:
        re_pnl += pnl; re_n += 1
    else:
        iso_pnl += pnl; iso_n += 1
    last[sym] = ts
print(f"перезаходы (<5ч по монете): {re_n} шт, {re_pnl:+.2f} USDT")
print(f"первые входы: {iso_n} шт, {iso_pnl:+.2f} USDT")
