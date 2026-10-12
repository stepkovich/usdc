"""ДНЕВНОЙ ТЕНЕВОЙ СУДЬЯ БУРИ (конвейер подтверждения, 12.10, решение
владельца «в»: фильтр «тихая треть» в демо + офлайн-подтверждение).

Что делает раз в сутки (утро, после публикации вчерашних дневных
пакетов биржи):
1. Докачивает дневные пакеты 30м-свечей (data.binance.vision) с
   27.09.2026 до вчерашнего дня — поверх кеша панели data_cache_30m.
2. Прогоняет ЗАМОРОЖЕННУЮ модель (models/ml5h.txt) на барах вчерашнего
   дня по всем монетам — как боевой движок: те же признаки, gate из
   мета, cooldown 10 баров, живая терциль atr_pct по вселенной бара
   (тот же фильтр «тихая треть», что в бою с 12.10).
3. Каждый сигнал (с фильтром и без) получает исход side×fwd(10 баров)
   − 10бп, когда закрытие t+10 доступно (хвост дня доезжает завтра).
4. Копит ops/judge/results.jsonl и печатает накопленное: средняя
   С фильтром vs БЕЗ фильтра vs все, по сторонам. Это и есть
   статистика подтверждения кандидата на свежих днях.

Паритет с боем: признаки из ml5h.features (та же функция), модель та
же, порог из артефакта. Отличия честно: судья не знает про слоты,
GTX-отказы и позиции движка (несколько лишних сигналов — против нас
не играет: сравнение с фильтром/без на одних сигналах).
"""
import io
import json
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path("/home/iek/PycharmProjects/USDC")
sys.path.insert(0, str(ROOT))
from ml5h.features import make_features, g_cross, sym_id  # noqa: E402

JDIR = ROOT / "ops" / "judge"
KCACHE = JDIR / "klines"
JDIR.mkdir(exist_ok=True, parents=True)
PANEL = ROOT / "data_cache_30m"
PANEL_FROM = date(2026, 9, 27)          # кеш панели кончается ~28.09
NEED_BARS = 1200                        # 25 дней: запас для всех окон

META = json.load(open(ROOT / "models" / "ml5h_meta.json"))
FEATS = list(META["features"])
GATE = float(META.get("gate", 0.65))
SYMS = [s for s in META["symbols"] if s != "BTCUSDT"] + ["BTCUSDT"]
model = lgb.Booster(model_file=str(ROOT / "models" / "ml5h.txt"))
META_PATH = JDIR / "meta.txt"
meta_model = None
if META_PATH.exists():
    meta_model = lgb.Booster(model_file=str(META_PATH))
MFEATS = FEATS + ["p_sig", "atr_rel"]
USECOLS = ["open_time", "high", "low", "close", "volume",
           "quote_volume", "taker_buy_base"]

def yesterday_utc() -> date:
    return (datetime.now(timezone.utc) - timedelta(days=1)).date()

def daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)

def csv_path(sym: str, d: date) -> Path:
    return KCACHE / sym / f"{d.isoformat()}.csv"

def fetch_one(sym: str, d: date) -> None:
    fn = csv_path(sym, d)
    if fn.exists():
        return
    url = (f"https://data.binance.vision/data/futures/um/daily/klines/"
           f"{sym}/30m/{sym}-30m-{d.isoformat()}.zip")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
            raw = zf.read(zf.namelist()[0])
        fn.parent.mkdir(exist_ok=True, parents=True)
        fn.write_bytes(raw)
    except Exception:
        pass                                  # нет пакета = нет монеты/дня

def download_days(days: list[date], syms: list[str]) -> None:
    jobs = [(s, d) for d in days for s in syms if not csv_path(s, d).exists()]
    if not jobs:
        return
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda j: fetch_one(*j), jobs))
    got = sum(1 for s, d in jobs if csv_path(s, d).exists())
    print(f"докачано {got}/{len(jobs)} пакетов за {time.time()-t0:.0f}с",
          flush=True)

def read_daily(sym: str, d: date):
    fn = csv_path(sym, d)
    if not fn.exists():
        return None
    try:
        df = pd.read_csv(fn, header=None, usecols=[0, 2, 3, 4, 5, 7, 9],
                         names=["open_time", "high", "low", "close",
                                "volume", "quote_volume",
                                "taker_buy_base"])
    except Exception:
        return None
    return df

def load_sym(sym: str) -> pd.DataFrame | None:
    parts = []
    pc = PANEL / f"{sym}_30m.csv"
    if pc.exists():
        try:
            parts.append(pd.read_csv(pc, usecols=USECOLS))
        except Exception:
            pass
    for d in daterange(PANEL_FROM, yesterday_utc()):
        df = read_daily(sym, d)
        if df is not None:
            parts.append(df)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    for c in USECOLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ot = pd.to_numeric(df["open_time"], errors="coerce")
    ot = ot.astype("float64")
    ot = np.where(ot > 1e14, ot / 1000.0, ot)      # микро->миллисекунды
    df["open_time"] = ot.astype(np.int64)
    df = df.dropna(subset=["close"]).drop_duplicates("open_time") \
        .sort_values("open_time").tail(NEED_BARS)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    return df if len(df) >= 300 else None

def signals_and_rows(day: date):
    """(сигналы дня, {open_time: q33}, кадры признаков для исходов)."""
    D0 = int(datetime(day.year, day.month, day.day,
                      tzinfo=timezone.utc).timestamp() * 1000)
    D1 = int(datetime(day.year, day.month, day.day,
                      tzinfo=timezone.utc).timestamp() * 1000) + 86_400_000
    rows = []          # (sym, ot, p, atr, close, feats_ok)
    for sym in SYMS:
        df = load_sym(sym)
        if df is None:
            continue
        X = make_features(df, None if sym == "BTCUSDT"
                          else load_sym_btc, sym)
        X = X.replace([np.inf, -np.inf], np.nan)
        Xg = g_cross(df, load_sym_btc, sym)
        for c in Xg.columns:
            X[c] = Xg[c]
        if "sym_id" in FEATS:
            X["sym_id"] = float(sym_id(sym))
        if not set(FEATS).issubset(X.columns):
            continue                      # как feature_row: нет колонок -> None
        mask = day_mask = (df["open_time"].values >= D0) & \
                          (df["open_time"].values < D1)
        sub = df[mask]
        Xd = X[mask]
        if len(sub) == 0:
            continue
        ok = Xd[FEATS].notna().all(axis=1).values
        if not ok.any():
            continue
        M = np.column_stack([Xd[f].values for f in FEATS])[ok]
        p = model.predict(M)
        closes = sub["close"].values[ok]
        ots = sub["open_time"].values[ok]
        atrs = Xd["atr_pct"].values[ok]
        vecs = np.column_stack([M, p, atrs])
        rows.extend((sym, int(o), float(pp), float(a), float(c), v)
                    for o, pp, a, c, v
                    in zip(ots, p, atrs, closes, vecs))
    # живая терциль ATR по вселенной каждого бара
    by_bar = {}
    for sym, ot, p, atr, c, v in rows:
        if np.isfinite(atr):
            by_bar.setdefault(ot, []).append(atr)
    q33 = {ot: float(np.quantile(v, 1 / 3)) for ot, v in by_bar.items()
           if len(v) >= 30}
    return rows, q33

load_sym_btc = None

def main():
    D = yesterday_utc()
    print("судья за", D, flush=True)
    download_days(list(daterange(PANEL_FROM, D)), SYMS)
    global load_sym_btc
    btc = load_sym("BTCUSDT")
    load_sym_btc = btc["close"] if btc is not None else None
    rows, q33 = signals_and_rows(D)
    # cooldown 10 баров на символ, как в бою
    rows.sort(key=lambda r: r[1])
    last_sig = {}
    signals = []
    for sym, ot, p, atr, c, vec in rows:
        if ot - last_sig.get(sym, -10**18) >= 10 * 1800_000 \
                and (p > GATE or p < 1 - GATE):
            side = 1 if p > GATE else -1
            in_f = np.isfinite(atr) and q33.get(ot) is not None \
                and atr >= q33[ot]
            s = {"sym": sym, "ot": ot, "side": side,
                 "p": round(p, 4), "atr": atr,
                 "in_filter": bool(in_f), "day": str(D),
                 "vec": [round(float(x), 6) for x in vec]}
            if meta_model is not None:
                pw = float(meta_model.predict(np.array([vec]))[0])
                s["pw"] = round(pw, 4)
                s["meta_ok"] = bool(pw >= 0.5)
            signals.append(s)
            last_sig[sym] = ot
    sg_path = JDIR / "signals.jsonl"
    seen = set()
    if sg_path.exists():
        for line in sg_path.read_text().splitlines():
            try:
                s = json.loads(line)
                seen.add((s["sym"], s["ot"], s["side"]))
            except Exception:
                pass
    new = [s for s in signals if (s["sym"], s["ot"], s["side"]) not in seen]
    with open(sg_path, "a") as f:
        for s in new:
            f.write(json.dumps(s) + "\n")
    print(f"новых сигналов {len(new)} (всего {len(seen) + len(new)})",
          flush=True)

    # --- разрешение исходов: закрываемые при текущем кеше ---
    closes_all = {}
    for sym in {s["sym"] for s in new} | \
               {s["sym"] for s in _load_signals()}:
        df = load_sym(sym)
        if df is not None:
            closes_all[sym] = (df["open_time"].values,
                               df["close"].values)
    res_path = JDIR / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["sym"], r["ot"], r["side"]))
            except Exception:
                pass
    resolved = []
    for s in _load_signals():
        key = (s["sym"], s["ot"], s["side"])
        if key in done:
            continue
        ots, cls = closes_all.get(s["sym"], (np.array([]), np.array([])))
        j = int(np.searchsorted(ots, s["ot"]))
        if j + 10 >= len(ots) or ots[j] != s["ot"]:
            continue                          # ещё не доехал
        pnl = s["side"] * (cls[j + 10] / cls[j] - 1) * 1e4 - 10.0
        out = dict(s, pnl_bp=round(float(pnl), 2))
        resolved.append(out)
    if resolved:
        with open(res_path, "a") as f:
            for r in resolved:
                f.write(json.dumps(r) + "\n")
    # --- сводка ---
    allr = _load_results()
    if allr:
        df = pd.DataFrame(allr)
        inf = df[df["in_filter"]]
        exc = df[~df["in_filter"]]
        line = (f"судья: накоплено {len(df)} сигналов | "
                f"с фильтром {len(inf)}: {inf.pnl_bp.mean():+.1f}бп | "
                f"без {len(exc)}: "
                f"{exc.pnl_bp.mean() if len(exc) else float('nan'):+.1f}бп")
        if "meta_ok" in df.columns:
            mo = df[df["meta_ok"]]
            line += (f" | мета-да {len(mo)}: "
                     f"{mo.pnl_bp.mean() if len(mo) else float('nan'):+.1f}бп")
        print(line, flush=True)
        for side, name in ((1, "лонги"), (-1, "шорты")):
            d2 = inf[inf["side"] == side]
            if len(d2):
                print(f"  {name}: n={len(d2)} {d2.pnl_bp.mean():+.1f}бп",
                      flush=True)
        if "meta_ok" in df.columns:
            both = df[df["meta_ok"] & df["in_filter"]]
            if len(both):
                print(f"  мета+фильтр: n={len(both)} "
                      f"{both.pnl_bp.mean():+.1f}бп", flush=True)
        (JDIR / "summary.txt").write_text(line + "\n")

def _load_signals():
    p = JDIR / "signals.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def _load_results():
    p = JDIR / "results.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

if __name__ == "__main__":
    main()
