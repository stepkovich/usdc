"""ПРОГОН СТРАТЕГИИ КОЛЛЕГИ (VWAP-deviation mean-reversion) на НАШЕЙ панели.
Его суть: цена ушла от дневного VWAP на 3%+ -> против прокола; TP = якорь,
SL = 4xATR, time-stop 72ч, 1 сигнал/день, вход по закрытию бара.

Проход 1 — СТРОГО его код (vwap7_backtest.py; меняется только источник
  данных: наша 5м-панель пересобрана в его шаг 15м). Его параметры, его
  МСК-граница дня, его комиссии (TP бесплатно, стоп 4.55бп+слип 3бп).
  Вселенные: его 7 монет + ВСЯ наша панель; окна: вся история + его 90 дней.
Проход 2 — ПРАВКИ: граница дня UTC, наши комиссии (вход+TP мейкер 2бп,
  стоп/тайм тейкер 5бп+слип 3бп), вся панель. Параметры НЕ трогаем.
Проход 3 — его «финальная» конфигурация (+ фандинг-фильтр F1: лонг только
  при funding>0, шорт при funding<0, запись не старше 8ч; + ignition
  RSI(14,15m) скачок >=10 за бар) на окне, покрытом нашим фандингом
  (2024+), его комиссии и с нашими.
Ничего не подбираем: параметры коллеги заморожены во всех проходах.
"""
import io, json, os, time, zipfile
import numpy as np, pandas as pd

t0 = time.time()
N0 = 5.0
VWAP_DEV, SL_ATR, ATR_N = 0.03, 4.0, 96
TIME_STOP_MS = 72 * 3600 * 1000
FEE_TAKER_HIS, SLIP_HIS = 0.000455, 0.0003
FEE_MAKER_OUR, FEE_TAKER_OUR, SLIP_OUR = 0.0002, 0.0005, 0.0003
T0_HIS = 1792684800000                 # его окно кончается 24.10.2026
WIN_HIS = T0_HIS - 90 * 86400 * 1000
MSK = 3 * 3600 * 1000                  # его day_start = полночь по МСК
HIS7 = ["HYPEUSDT", "ADAUSDT", "ENAUSDT", "XRPUSDT", "ONDOUSDT",
        "ETHUSDT", "SOLUSDT"]
RSI_TH, FR_MAX_AGE = 10.0, 8 * 3600 * 1000


def log(m):
    print(f"{m} | {time.time()-t0:.0f}с", flush=True)


def day_start(ms, tz_offset):
    d = (ms + tz_offset) // 86400_000
    return d * 86400_000 - tz_offset


def find_zip(name):
    for root, _d, fs in os.walk("/kaggle/input"):
        if name in fs:
            return os.path.join(root, name)
    return None


def load_15m(zf, name):
    """5м-бар -> 15м-бар (open первый, high max, low min, close последний,
    volume сумма). Списки как у коллеги: ot, hi, lo, cl, vo."""
    df = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["open_time", "high", "low", "close", "volume"])
    for c in df.columns:
        if c != "open_time":
            df[c] = df[c].astype("float64")
    df["open_time"] = df["open_time"].astype(np.int64)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    r = df.resample("15min").agg({"high": "max", "low": "min",
                                  "close": "last", "volume": "sum"})
    r = r.dropna(subset=["close"])
    if len(r) < 500:
        return None
    return dict(ot=(r.index.asi8 // 10**6).astype(np.int64),
                hi=r["high"].to_numpy(), lo=r["low"].to_numpy(),
                cl=r["close"].to_numpy(), vo=r["volume"].fillna(0).to_numpy())


def prep(b):
    """ATR(96,15m) и дневной VWAP c ресетом по границе дня (tz_offset)."""
    ot, hi, lo, cl, vo = b["ot"], b["hi"], b["lo"], b["cl"], b["vo"]
    n = len(cl)
    tr = np.empty(n); tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
    c = np.cumsum(np.insert(tr, 0, 0.0))
    idx = np.arange(n)
    lo_i = np.maximum(0, idx - ATR_N + 1)
    atr = (c[idx + 1] - c[lo_i]) / (idx - lo_i + 1)
    vwap = np.full(n, np.nan)
    pv = pq = 0.0
    cur = None
    tp = (hi + lo + cl) / 3.0
    for i in range(n):
        d = day_start(int(ot[i]), CUR_TZ)
        if d != cur:
            cur = d; pv = pq = 0.0
        pv += tp[i] * vo[i]; pq += vo[i]
        if pq > 0:
            vwap[i] = pv / pq
    b["atr"], b["vwap"] = atr, vwap
    return b


def rsi_series(cl, p=14):
    d = np.diff(cl, prepend=cl[0])
    ups = np.where(d > 0, d, 0.0); dns = np.where(d < 0, -d, 0.0)
    a = 2.0 / (p + 1)
    ru = np.zeros_like(ups); rd = np.zeros_like(dns)
    ru[0] = ups[0]; rd[0] = dns[0]
    for i in range(1, len(ups)):
        ru[i] = ru[i-1] + a * (ups[i] - ru[i-1])
        rd[i] = rd[i-1] + a * (dns[i] - rd[i-1])
    return 100 - 100 / (1 + ru / np.maximum(rd, 1e-12))


def funding_at(ft, fr, t):
    """Его series_at: последняя запись не старше 8ч."""
    if len(ft) == 0:
        return None
    j = np.searchsorted(ft, t, "right") - 1
    if j < 0 or t - ft[j] > FR_MAX_AGE:
        return None
    return fr[j]


def run(b, *, fee_mode, use_f1, use_rsi, t_start, ft=None, fr=None):
    """Его цикл 1:1. fee_mode: 'his' | 'ours'."""
    global CUR_TZ
    ot, hi, lo, cl = b["ot"], b["hi"], b["lo"], b["cl"]
    atr, vwap = b["atr"], b["vwap"]
    n = len(cl)
    rj = rsi_series(cl)
    rj = np.diff(rj, prepend=rj[0])
    trades = []
    curve = [0.0]; eq = 0.0
    pos = None; last_day = None
    tz = CUR_TZ
    for i in range(200, n):
        v = vwap[i]
        if not np.isfinite(v) or atr[i] <= 0:
            continue
        if pos:
            side, entry, sl, tp, t_in = pos
            exit_px = fee_out = None
            if side == 1:
                if lo[i] <= sl: exit_px, fee_out = sl * (1 - SLIP_EFF), FEE_STOP_EFF
                elif hi[i] >= tp: exit_px, fee_out = tp, FEE_TP_EFF
            else:
                if hi[i] >= sl: exit_px, fee_out = sl * (1 + SLIP_EFF), FEE_STOP_EFF
                elif lo[i] <= tp: exit_px, fee_out = tp, FEE_TP_EFF
            if exit_px is None and ot[i] - t_in > TIME_STOP_MS:
                exit_px, fee_out = cl[i], FEE_TIME_EFF
            if exit_px is not None:
                dpx = (exit_px - entry) if side == 1 else (entry - exit_px)
                pnl = dpx / entry * N0 - N0 * FEE_IN_EFF - N0 * fee_out
                eq += pnl; curve.append(eq); trades.append((ot[i], pnl))
                pos = None
        else:
            if ot[i] < t_start:
                continue
            d = day_start(int(ot[i]), tz)
            if d == last_day:
                continue
            dv = cl[i] / v - 1
            side = 0
            if dv <= -VWAP_DEV: side = 1
            elif dv >= VWAP_DEV: side = -1
            if side == 0:
                continue
            if use_f1:
                f = funding_at(ft, fr, int(ot[i]))
                if f is None: continue
                if side == 1 and f <= 0: continue
                if side == -1 and f >= 0: continue
            if use_rsi and rj[i] < RSI_TH:
                continue
            if side == 1:
                pos = (1, cl[i], cl[i] - SL_ATR * atr[i], v, ot[i])
            else:
                pos = (-1, cl[i], cl[i] + SL_ATR * atr[i], v, ot[i])
            last_day = d
    return trades, curve


def summarize(name, tr_by_sym):
    all_tr = [t for ts in tr_by_sym.values() for t in ts]
    wins = sum(1 for _, p in all_tr if p > 0)
    losses = [p for _, p in all_tr if p <= 0]
    pf = (sum(p for _, p in all_tr if p > 0) / max(1e-9, -sum(losses))) \
        if losses else 99.0
    peak = dd = 0.0
    print(f"{name}: сделок {len(all_tr)}, WR {wins/max(1,len(all_tr))*100:.0f}%, "
          f"PnL {sum(p for _, p in all_tr):+.2f}$, PF {pf:.2f}", flush=True)
    return {"n": len(all_tr),
            "wr": round(wins / max(1, len(all_tr)) * 100, 1),
            "pnl": round(sum(p for _, p in all_tr), 2), "pf": round(pf, 2)}


zp = find_zip("panel_5m.zip")
assert zp, "panel_5m.zip не найден"
zf = zipfile.ZipFile(zp)
members = sorted(n for n in zf.namelist() if n.endswith("_5m.csv"))
have = {n.replace("_5m.csv", "") for n in members}
log(f"панель: {len(members)} монет; его 7 монет: "
    f"{[s for s in HIS7 if s in have]} (нет на нашей бирже: "
    f"{[s for s in HIS7 if s not in have]})")

FUND_DIR = None
for root, _d, fs in os.walk("/kaggle/input"):
    if any(f.startswith("ETHUSDT.npz") for f in fs) and "f2025" in root:
        FUND_DIR = os.path.dirname(root)
        break
FUND = {}
if FUND_DIR:
    for yr in ("f2024", "f2025"):
        p = os.path.join(FUND_DIR, yr)
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            s = f.replace(".npz", "")
            z = np.load(os.path.join(p, f))
            FUND.setdefault(s, ([], []))
            FUND[s][0].append(z["t"]); FUND[s][1].append(z["rate"])
    FUND = {s: (np.concatenate(v[0]), np.concatenate(v[1]))
            for s, v in FUND.items()}
    log(f"фандинг: {len(FUND)} монет")

REPORT = {}

# ---------- Проход 1: СТРОГО его код ----------
CUR_TZ = MSK
SLIP_EFF, FEE_STOP_EFF, FEE_TP_EFF, FEE_TIME_EFF, FEE_IN_EFF = \
    SLIP_HIS, FEE_TAKER_HIS, 0.0, FEE_TAKER_HIS, 0.0
for label, syms, t_start in (("его 7 монет, вся история", HIS7, 0),
                             ("его 7 монет, его 90 дней", HIS7, WIN_HIS)):
    by = {}
    for s in syms:
        nm = f"{s}_5m.csv"
        if nm not in zf.namelist():
            continue
        b = prep(load_15m(zf, nm))
        tr, _ = run(b, fee_mode="his", use_f1=False, use_rsi=False,
                    t_start=t_start)
        by[s] = tr
    REPORT[f"P1|{label}"] = summarize(label, by)
log("проход 1а (его код, его монеты) готов")

by = {}
for k, nm in enumerate(members):
    s = nm.replace("_5m.csv", "")
    b = prep(load_15m(zf, nm))
    tr, _ = run(b, fee_mode="his", use_f1=False, use_rsi=False, t_start=0)
    by[s] = tr
    if (k + 1) % 100 == 0:
        log(f"P1 всё: {k+1}/{len(members)}")
REPORT["P1|вся панель, вся история"] = summarize("P1 вся панель", by)
# его 90 дней на всей панели: те же сделки, вход после WIN_HIS
by90 = {s: [t for t in ts if t[0] >= WIN_HIS] for s, ts in by.items()}
REPORT["P1|вся панель, его 90 дней"] = summarize("P1 панель 90д", by90)
# по годам
years = {}
for s, ts in by.items():
    for t, p in ts:
        years.setdefault(pd.Timestamp(t, unit="ms").year, []).append(p)
for y in sorted(years):
    w = sum(1 for p in years[y] if p > 0)
    l_ = [p for p in years[y] if p <= 0]
    pf = sum(p for p in years[y] if p > 0) / max(1e-9, -sum(l_)) if l_ else 99
    print(f"  P1 {y}: сделок {len(years[y])}, WR {w/len(years[y])*100:.0f}%, "
          f"PnL {sum(years[y]):+.1f}$, PF {pf:.2f}", flush=True)
log("проход 1б (его код, вся панель) готов")

# ---------- Проход 2: ПРАВКИ (UTC-день, наши комиссии) ----------
CUR_TZ = 0
SLIP_EFF, FEE_STOP_EFF, FEE_TP_EFF, FEE_TIME_EFF, FEE_IN_EFF = \
    SLIP_OUR, FEE_TAKER_OUR, FEE_MAKER_OUR, FEE_TAKER_OUR, FEE_MAKER_OUR
by = {}
for k, nm in enumerate(members):
    s = nm.replace("_5m.csv", "")
    b = prep(load_15m(zf, nm))          # vwap пересчитается с UTC-границей
    tr, _ = run(b, fee_mode="ours", use_f1=False, use_rsi=False, t_start=0)
    by[s] = tr
    if (k + 1) % 100 == 0:
        log(f"P2: {k+1}/{len(members)}")
REPORT["P2|вся панель, UTC + наши комиссии"] = summarize("P2 вся панель", by)
years = {}
for s, ts in by.items():
    for t, p in ts:
        years.setdefault(pd.Timestamp(t, unit="ms").year, []).append(p)
for y in sorted(years):
    w = sum(1 for p in years[y] if p > 0)
    l_ = [p for p in years[y] if p <= 0]
    pf = sum(p for p in years[y] if p > 0) / max(1e-9, -sum(l_)) if l_ else 99
    print(f"  P2 {y}: сделок {len(years[y])}, WR {w/len(years[y])*100:.0f}%, "
          f"PnL {sum(years[y]):+.1f}$, PF {pf:.2f}", flush=True)
log("проход 2 (правки) готов")

# ---------- Проход 3: его финальная конфига (+ фандинг + ignition), 2024+ ----------
T2024 = 1704067200000
for tag, msk in (("его день+его комиссии", True), ("UTC+наши комиссии", False)):
    if msk:
        CUR_TZ = MSK
        SLIP_EFF, FEE_STOP_EFF, FEE_TP_EFF, FEE_TIME_EFF, FEE_IN_EFF = \
            SLIP_HIS, FEE_TAKER_HIS, 0.0, FEE_TAKER_HIS, 0.0
    else:
        CUR_TZ = 0
        SLIP_EFF, FEE_STOP_EFF, FEE_TP_EFF, FEE_TIME_EFF, FEE_IN_EFF = \
            SLIP_OUR, FEE_TAKER_OUR, FEE_MAKER_OUR, FEE_TAKER_OUR, FEE_MAKER_OUR
    by = {}
    for k, nm in enumerate(members):
        s = nm.replace("_5m.csv", "")
        if s not in FUND:
            continue
        b = prep(load_15m(zf, nm))
        ft, fr = FUND[s]
        tr, _ = run(b, fee_mode="x", use_f1=True, use_rsi=True,
                    t_start=T2024, ft=ft, fr=fr)
        by[s] = tr
        if (k + 1) % 150 == 0:
            log(f"P3[{tag}]: {k+1}/{len(members)}")
    REPORT[f"P3 {tag}|2024+"] = summarize(f"P3 {tag} (F1+ignition)", by)
log("проход 3 готов")

with open("/kaggle/working/vwap_report.json", "w") as f:
    json.dump(REPORT, f, indent=1, default=float)
print("\n=== СВОДКА ===", flush=True)
for k, v in REPORT.items():
    print(f"{k:44} n={v['n']:5} WR {v['wr']:5}% PnL {v['pnl']:+9.2f}$ PF {v['pf']}", flush=True)
log("VWAP-ПРОГОН ГОТОВ")
