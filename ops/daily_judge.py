"""ДНЕВНОЙ ТЕНЕВОЙ СУДЬЯ БУРИ (конвейер подтверждения, 12.10, решение
владельца «в»: фильтр «тихая треть» в демо + офлайн-подтверждение).

Что делает раз в сутки (утро, после публикации вчерашних дневных
пакетов биржи):
1. Докачивает дневные пакеты 30м-свечей (data.binance.vision) с
   27.09.2026 до вчерашнего дня — поверх кеша панели data_cache_30m.
2. Прогоняет модель из ЖИВОЙ папки боевых моделей (models/, маунт :ro)
   на барах вчерашнего дня по всем монетам — как боевой движок: те же признаки, gate из
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
import hashlib
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("JUDGE_ROOT", "/home/iek/PycharmProjects/USDC"))
sys.path.insert(0, str(ROOT))
from ml5h.features import make_features, g_cross, sym_id  # noqa: E402

JDIR = ROOT / "ops" / "judge"
KCACHE = JDIR / "klines"
JDIR.mkdir(exist_ok=True, parents=True)
PANEL = ROOT / "data_cache_30m"
PANEL_FROM = date(2026, 9, 27)          # кеш панели кончается ~28.09
NEED_BARS = 1500                        # 31 день: запас для окон + оборот 30д

META = json.load(open(ROOT / "models" / "ml5h_meta.json"))
FEATS = list(META["features"])
GATE = float(META.get("gate", 0.65))
SYMS = [s for s in META["symbols"] if s != "BTCUSDT"] + ["BTCUSDT"]
model = lgb.Booster(model_file=str(ROOT / "models" / "ml5h.txt"))
# мета судьи = БОЕВАЯ мета (живая папка боевых, один источник правды)
META_PATH = ROOT / "models" / "meta.txt"
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


def tg(text: str) -> None:
    tok = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return
    try:
        data = urllib.parse.urlencode(
            {"chat_id": chat, "text": text}).encode()
        urllib.request.urlopen(urllib.request.Request(
            f"https://api.telegram.org/bot{tok}/sendMessage", data=data),
            timeout=20)
    except Exception as e:
        print(f"ТГ не ушёл: {e}", flush=True)


def model_audit() -> None:
    """(в) штамп аудита: md5 трёх моделей в результаты каждого прогона,
    ТГ только при изменении с прошлого прогона (помощник, 09.11)."""
    files = [("ml5h.txt", ROOT / "models" / "ml5h.txt"),
             ("ml5h_meta.json", ROOT / "models" / "ml5h_meta.json"),
             ("meta.txt", META_PATH),
             ("calibration.json", ROOT / "models" / "calibration.json")]
    stamp = {"run": str(yesterday_utc()),
             "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for name, p in files:
        try:
            stamp[name] = hashlib.md5(p.read_bytes()).hexdigest()[:12]
        except Exception:
            stamp[name] = "нет"
    apath = JDIR / "model_audit.jsonl"
    prev = None
    if apath.exists():
        try:
            prev = json.loads(apath.read_text().strip().splitlines()[-1])
        except Exception:
            prev = None
    with open(apath, "a") as f:
        f.write(json.dumps(stamp, ensure_ascii=False) + "\n")
    changed = [k for k, _ in files if prev and stamp[k] != prev.get(k)]
    if changed:
        tg("⚖️ СУДЬЯ: сменился состав моделей!\n" + "\n".join(
            f"{k}: {prev.get(k, '?')} → {stamp[k]}" for k in changed))
        print("аудит моделей: изменилось " + ", ".join(changed), flush=True)

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

def oi_feats(sym, ot, cap_day):
    d0 = datetime.fromtimestamp(ot / 1000, tz=timezone.utc).date()
    days = [str(d0 + pd.Timedelta(days=k)) for k in (-2, -1, 0, 1)]
    m = metrics_frame(sym, [x for x in days if x <= cap_day])
    if m is None or len(m) < 50:
        return None
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

def signals_and_rows(day: date):
    """(сигналы дня, {open_time: q33}, {sym: оборот за 30д до дня})."""
    D0 = int(datetime(day.year, day.month, day.day,
                      tzinfo=timezone.utc).timestamp() * 1000)
    D1 = int(datetime(day.year, day.month, day.day,
                      tzinfo=timezone.utc).timestamp() * 1000) + 86_400_000
    rows = []          # (sym, ot, p, atr, close, feats_ok)
    vol30 = {}         # оборот монеты за 30 дней ДО дня — флажок топ-100
    for sym in SYMS:
        df = load_sym(sym)
        if df is None:
            continue
        w = (df["open_time"].values >= D0 - 30 * 86400_000) \
            & (df["open_time"].values < D0)
        vol30[sym] = float(df.loc[w, "quote_volume"].sum())
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
    return rows, q33, vol30

load_sym_btc = None

def main():
    if len(sys.argv) > 1:                     # задним числом: 2026-10-05
        D = datetime.strptime(sys.argv[1], "%Y-%m-%d") \
            .replace(tzinfo=timezone.utc).date()
    else:
        D = yesterday_utc()
    print("судья за", D, flush=True)
    model_audit()
    download_days(list(daterange(PANEL_FROM, D)), SYMS)
    global load_sym_btc
    btc = load_sym("BTCUSDT")
    load_sym_btc = btc["close"] if btc is not None else None
    rows, q33, vol30 = signals_and_rows(D)
    # флажок «топ-100 по обороту» (владелец 02.11): оборот за 30д ДО дня —
    # без подглядывания; список дня сохраняется для аудита
    top100 = set(sorted(vol30, key=vol30.get, reverse=True)[:100])
    tdir = JDIR / "top100"
    tdir.mkdir(exist_ok=True)
    (tdir / f"{D}.json").write_text(json.dumps(
        sorted(top100), indent=0))
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
                 "top100": sym in top100,
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
    for s in signals:
        f = oi_feats(s["sym"], s["ot"], str(D))
        if f:
            s["oi_chg_1h"] = round(f["oi_chg_1h"], 6) \
                if f["oi_chg_1h"] == f["oi_chg_1h"] else None
            s["oi_chg_6h"] = round(f["oi_chg_6h"], 6) \
                if f["oi_chg_6h"] == f["oi_chg_6h"] else None
            s["fuel"] = f["fuel"]
    new = [s for s in signals if (s["sym"], s["ot"], s["side"]) not in seen]
    with open(sg_path, "a") as f:
        for s in new:
            f.write(json.dumps(s) + "\n")
    print(f"новых сигналов {len(new)} (всего {len(seen) + len(new)})",
          flush=True)
    # ФЛАГ ПЕРЕОБУЧЕНИЯ: если бот его выставил — уведомить владельца
    # (решение о внеплановом пересмотре — его, автозапуска нет)
    flag = ROOT / "data" / "signals" / "retrain_needed"
    if flag.exists():
        import bot.telegram as tgmod
        tgmod.fire("🧠⚠️ Флаг переобучения активен (30 сделок в минусе) — "
                   "требуется решение владельца")

    # --- разрешение исходов: закрываемые при текущем кеше ---
    closes_all = {}
    for sym in {s["sym"] for s in new} | \
               {s["sym"] for s in _load_signals()}:
        df = load_sym(sym)
        if df is not None:
            closes_all[sym] = (df["open_time"].values,
                               df["close"].values)
    # ЧЕСТНАЯ ФОРМУЛА ЭКЗАМЕНА: стоп -15% по реальным low/high свечей
    # (худший случай внутри бара: стоп раньше тейка), комиссии как в бою.
    lows_all = {}; highs_all = {}
    for sym in {s["sym"] for s in _load_signals()} | {s["sym"] for s in new}:
        df = load_sym(sym)
        if df is not None:
            lows_all[sym] = df["low"].values; highs_all[sym] = df["high"].values
    def honest_pnl(sym, ot, side, cls, j):
        stop = cls[j] * (1 - 0.15) if side == 1 else cls[j] * (1 + 0.15)
        for k in range(j + 1, min(j + 11, len(cls))):
            lo, hi = lows_all[sym][k], highs_all[sym][k]
            if side == 1 and lo <= stop: return stop, True
            if side == -1 and hi >= stop: return stop, True
        return cls[min(j + 10, len(cls) - 1)], False
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
        exit_px, stopped = honest_pnl(s["sym"], s["ot"], s["side"], cls, j)
        pnl = s["side"] * (exit_px / cls[j] - 1) * 1e4 - 10.0
        if stopped: pnl -= 12.0   # стоп = тейкер-выход: ком+слип поверх
        out = dict(s, pnl_bp=round(float(pnl), 2), stopped=bool(stopped))
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
        avg_all = df.pnl_bp.mean()
        avg_inf = inf.pnl_bp.mean() if len(inf) else float("nan")
        avg_exc = exc.pnl_bp.mean() if len(exc) else float("nan")

        def verd(delta, need=10):
            if delta >= need:
                return "уже в плюсе ✓"
            if delta <= -need:
                return "пока в минусе ✗"
            return "пока ничья"

        d_filt = avg_inf - avg_exc
        v_filt = verd(d_filt)
        plain = [f"судья: накоплено {len(df)} сигналов | "
                 f"с фильтром {len(inf)}: {avg_inf:+.1f}бп | "
                 f"без {len(exc)}: {avg_exc:+.1f}бп"]
        html = [f"🧠 <b>Судья Бури</b> — накоплено сигналов: <b>{len(df)}</b>",
                "",
                "📐 <b>Фильтр «тихая треть»</b> (сонные монеты мимо):",
                f"• пропущено им: {len(inf)} → <b>{avg_inf:+.1f} бп</b>",
                f"• им отсеяно: {len(exc)} → {avg_exc:+.1f} бп",
                f"• вердикт: {v_filt}"]
        meta_lines = []
        if "meta_ok" in df.columns:
            mo = df[df["meta_ok"] == True]   # noqa: E712 (NaN-устойчиво)
            avg_mo = mo.pnl_bp.mean() if len(mo) else float("nan")
            d_meta = avg_mo - avg_all
            v_meta = verd(d_meta, 5)
            plain.append(f" | мета-да {len(mo)}: "
                         f"{avg_mo if len(mo) else float('nan'):+.1f}бп")
            meta_lines = ["",
                          "🤝 <b>Мета — «второе мнение»</b>:",
                          f"• подтверждено ею: {len(mo)} → "
                          f"<b>{avg_mo:+.1f} бп</b>",
                          f"• средний сигнал: {avg_all:+.1f} бп",
                          f"• вердикт: {v_meta}"]
        plain_sides, sides = [], []
        for side, name in ((1, "Лонги"), (-1, "Шорты")):
            d2 = inf[inf["side"] == side]
            if len(d2):
                plain_sides.append(f"  {name.lower()}: n={len(d2)} "
                                   f"{d2.pnl_bp.mean():+.1f}бп")
                sides.append(f"• {name}: {len(d2)} → "
                             f"<b>{d2.pnl_bp.mean():+.1f} бп</b>")
        both_txt = ""
        if "meta_ok" in df.columns:
            both = df[(df["meta_ok"] == True) & df["in_filter"]]  # noqa: E712
            if len(both):
                both_txt = (f"  мета+фильтр: n={len(both)} "
                            f"{both.pnl_bp.mean():+.1f}бп")
                plain_sides.append(both_txt)
        top_lines = []
        if "top100" in df.columns:
            t1 = df[df["top100"] == True]   # noqa: E712
            t0_ = df[df["top100"] == False]  # noqa: E712
            if len(t1) and len(t0_):
                d_top = t1.pnl_bp.mean() - t0_.pnl_bp.mean()
                v_top = verd(d_top, 5)
                plain.append(f" | топ-100 {len(t1)}: "
                             f"{t1.pnl_bp.mean():+.1f}бп")
                top_lines = ["",
                             "🏪 <b>Топ-100 по обороту</b> (идея владельца "
                             "02.11, замер обещал +3.4бп):",
                             f"• из топ-100: {len(t1)} → "
                             f"<b>{t1.pnl_bp.mean():+.1f} бп</b>",
                             f"• вне топ-100: {len(t0_)} → "
                             f"{t0_.pnl_bp.mean():+.1f} бп",
                             f"• вердикт: {v_top}"]
        fuel_lines = []
        if "fuel" in df.columns:
            lf = df[(df["side"] == 1) & (df["fuel"] == True)]  # noqa: E712
            ln = df[(df["side"] == 1) & (df["fuel"] != True)]  # noqa: E712
            if len(lf):
                plain.append(f" | топливо-да {len(lf)}: "
                             f"{lf.pnl_bp.mean():+.1f}бп")
            if len(lf) and len(ln):
                d_fuel = lf.pnl_bp.mean() - ln.pnl_bp.mean()
                v_fuel = verd(d_fuel, 5)
                plain_sides.append(f"  лонги+топливо: n={len(lf)} "
                                   f"{lf.pnl_bp.mean():+.1f}бп")
                plain_sides.append(f"  лонги без топлива: n={len(ln)} "
                                   f"{ln.pnl_bp.mean():+.1f}бп")
                fuel_lines = ["",
                              "⛽ <b>ОИ-топливо лонгов</b> (интерес рос "
                              "перед сигналом):",
                              f"• с топливом: {len(lf)} → "
                              f"{lf.pnl_bp.mean():+.1f} бп",
                              f"• без топлива: {len(ln)} → "
                              f"{ln.pnl_bp.mean():+.1f} бп",
                              f"• вердикт: {v_fuel}"]
        print("\n".join(plain + plain_sides), flush=True)
        html += meta_lines
        html += ["", "🧭 <b>Стороны</b>:"] + sides
        html += top_lines
        html += fuel_lines
        (JDIR / "summary.txt").write_text("\n".join(html) + "\n")

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
