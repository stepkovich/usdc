"""ДВУНОГИЙ СТРЭДДЛ (предрегистрированная проба 12.10, идея владельца
от 07.10 «открывать сразу ЛОНГ И ШОРТ и на хвосте попытаться
заработать», приказ «го»).

МЕХАНИКА ЦИКЛА (по записи стакана, 37 USDC-пар, ~17 дней):
- вход: ОБЕ ноги GTX-лимитками сразу (лонг на биде, шорт на аске),
  мейкер, комиссия 0 (промо USDC mainnet);
  консервативное заполнение: мид должен пройти ЗА уровень на полспреда
  (очередь не видна); окно входа TTL 60с;
- одна нога открылась, вторая не успела -> «сирота»: закрыть тейкером
  в конце TTL (минус полспреда и 5бп), цикл зачтён как сирота;
- обе открыты (дельта 0): у КАЖДОЙ ноги свой тейк k×S и своя резка
  s×S от СВОЕГО входа (S = полный спред на старте). Первое касание:
  резка — тейкером (5бп + полспреда), тейк — мейкером (0); если в
  одном семпле и резка и тейк — считается РЕЗКА (консервативно);
  оставшаяся нога бежит до своего тейка или таймаута T=300с, в
  таймаут — тейкером. Гэпы бьют по фактической цене. Кулдаун 60с.

СЕТКА: тейк k ∈ {2, 4, до-таймаута} × резка s ∈ {0.5, 1.0} (в спредах).
ГЛАВНЫЙ ВАРИАНТ (объявлен до прогона): k=4, s=1.0 — арифметика: цикл
на быстром ходе = (k−s)×S − S/2 − 5бп, при типичных 2-4бп спреда
покрыть издержки может только k−s >= 3 спредов.

РЕЖИМ «заряжено» (прокси модели-бури на наших данных): 30-минутный
размах мида >= 70-го перцентиля этого размаха у данной монеты. В бою
на это место встанет сама модель-буря.

ВЕРДИКТ (до прогона): КАНДИДАТ, если на заряженных циклах главного
варианта: средняя >= +1.0 бп, n >= 1000, медиана дневных средних >= 0,
И заряженные >= тихие (иначе фильтр не нужен). Полная сетка — для
прозрачности, выбор постфактум запрещён. Ничего не деплоится.

ОГОВОРКИ: очередь заявок не видна; 1с дискретизация; 17 дней одного
режима; кандидат идёт на свежие дни перед любым боем.
"""
import glob, json, time
import numpy as np
import pandas as pd

t0 = time.time()
TAKES = [2.0, 4.0, 1e9]          # 1e9 = тейка нет, нога до таймаута
CUTS = [0.5, 1.0]
T_WINDOW = 300.0
TTL = 60.0
COOLDOWN = 60.0
TAKER = 0.0005
FILL_EXTRA = 0.5
WIDE_BP = 50.0
PRIMARY = (4.0, 1.0)

files = sorted(glob.glob("/kaggle/input/**/feat_*.parquet", recursive=True))
if not files:
    files = sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True))
print("файлов:", len(files), flush=True)

parts = {}
for f in files:
    try:
        df = pd.read_parquet(f, columns=["ts", "symbol", "mid", "spread_bp"])
    except Exception:
        continue
    df = df.dropna(subset=["mid", "spread_bp"]).iloc[::2]     # каденс ~1с
    for sym, g in df.groupby("symbol", observed=True):
        parts.setdefault(sym, []).append(
            (g["ts"].values.astype(np.int64),
             g["mid"].values.astype(np.float64),
             g["spread_bp"].values.astype(np.float64)))

syms = {}
for sym, lst in parts.items():
    ts = np.concatenate([a[0] for a in lst])
    mid = np.concatenate([a[1] for a in lst])
    sp = np.concatenate([a[2] for a in lst])
    order = np.argsort(ts, kind="stable")
    ts, mid, sp = ts[order], mid[order], sp[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts, mid, sp = ts[keep], mid[keep], sp[keep]
    if len(ts) >= 50_000:                  # порог ПОСЛЕ склейки, не по файлу
        syms[sym] = (ts, mid, sp)
print(f"монет {len(syms)}, строк {sum(len(v[0]) for v in syms.values())}"
      f" | {time.time()-t0:.0f}с", flush=True)

# ---------- режим «заряжено»: 30-мин размах >= p70 своей истории ----------
charged_map = {}
for sym, (ts, mid, _) in syms.items():
    grid = np.arange(ts[0], ts[-1], 10_000)
    m10 = mid[np.clip(np.searchsorted(ts, grid), 0, len(mid) - 1)]
    s10 = pd.Series(m10)
    r30 = ((s10.rolling(180).max() - s10.rolling(180).min()) / s10).values
    thr = np.nanpercentile(r30, 70)
    charged_map[sym] = (grid, r30 >= thr)

def is_charged(sym, t_ms):
    grid, ch = charged_map[sym]
    i = np.searchsorted(grid, t_ms) - 1
    return bool(ch[max(0, min(i, len(ch) - 1))])

def first_le(seg, lvl):
    a = np.nonzero(seg <= lvl)[0]
    return int(a[0]) if len(a) else -1

def first_ge(seg, lvl):
    a = np.nonzero(seg >= lvl)[0]
    return int(a[0]) if len(a) else -1

def run_sym(ts, mid, sp, k, s):
    """Список (t_start_ms, pnl_bp, kind) по одному символу и ячейке."""
    out = []
    n = len(ts)
    i = 0
    free_at = 0
    while i < n:
        if ts[i] < free_at:
            i += 1
            continue
        s_bp = sp[i]
        if not np.isfinite(mid[i]) or s_bp < 0.2 or s_bp > WIDE_BP:
            i += 1
            continue
        ref = mid[i]
        S = ref * s_bp / 1e4
        h = S / 2.0
        bid0, ask0 = ref - h, ref + h
        j_ttl = np.searchsorted(ts, ts[i] + int(TTL * 1000))
        if j_ttl <= i + 1:
            i += 1
            continue
        seg = mid[i + 1:j_ttl]
        ia = first_le(seg, bid0 - FILL_EXTRA * h)    # залился лонг
        ib = first_ge(seg, ask0 + FILL_EXTRA * h)    # залился шорт
        if ia == -1 and ib == -1:
            free_at = ts[j_ttl - 1] + int(COOLDOWN * 1000)
            i = j_ttl
            continue
        if ib == -1 or (ia != -1 and ia < ib):
            first = "long"
            j_first = i + 1 + ia
            seg2 = mid[j_first + 1:j_ttl]
            j2 = (j_first + 1 + first_ge(seg2, ask0 + FILL_EXTRA * h)
                  if first_ge(seg2, ask0 + FILL_EXTRA * h) != -1 else -1)
        else:
            first = "short"
            j_first = i + 1 + ib
            seg2 = mid[j_first + 1:j_ttl]
            j2 = (j_first + 1 + first_le(seg2, bid0 - FILL_EXTRA * h)
                  if first_le(seg2, bid0 - FILL_EXTRA * h) != -1 else -1)
        if j2 == -1:
            # сирота
            px = mid[j_ttl - 1]
            if first == "long":
                p = (px - h - bid0) / ref - TAKER
            else:
                p = (ask0 - (px + h)) / ref - TAKER
            out.append((int(ts[i]), p * 1e4, "orphan"))
            free_at = ts[j_ttl - 1] + int(COOLDOWN * 1000)
            i = j_ttl
            continue
        # --- обе ноги открыты ---
        long_take = bid0 + k * S
        long_cut = bid0 - s * S
        short_take = ask0 - k * S
        short_cut = ask0 + s * S
        w_end = max(np.searchsorted(ts, ts[j2] + int(T_WINDOW * 1000)),
                    j2 + 1)
        pnl = 0.0
        long_done = short_done = False
        pos = j2 + 1
        while pos < w_end and not (long_done and short_done):
            seg = mid[pos:w_end]
            cands = []
            if not long_done:
                a = first_le(seg, long_cut)
                if a != -1:
                    cands.append((a, 0, "long", "cut"))
                b = first_ge(seg, long_take)
                if b != -1:
                    cands.append((b, 1, "long", "take"))
            if not short_done:
                a = first_ge(seg, short_cut)
                if a != -1:
                    cands.append((a, 0, "short", "cut"))
                b = first_le(seg, short_take)
                if b != -1:
                    cands.append((b, 1, "short", "take"))
            if not cands:
                break
            cands.sort()
            a, prio, leg, kind = cands[0]
            px = mid[pos + a]
            if leg == "long" and kind == "cut":
                pnl += (px - h - bid0) / ref - TAKER
                long_done = True
            elif leg == "long":
                pnl += (long_take - bid0) / ref
                long_done = True
            elif kind == "cut":
                pnl += (ask0 - (px + h)) / ref - TAKER
                short_done = True
            else:
                pnl += (ask0 - short_take) / ref
                short_done = True
            pos = pos + a                  # не перескакиваем семпл: на нём
                                           # мог сработать и второй уровень
        if not long_done:
            pnl += (mid[w_end - 1] - h - bid0) / ref - TAKER
        if not short_done:
            pnl += (ask0 - (mid[w_end - 1] + h)) / ref - TAKER
        out.append((int(ts[i]), pnl * 1e4, "cycle"))
        free_at = ts[w_end - 1] + int(COOLDOWN * 1000)
        i = w_end
    return out

report = {"предрегистрация": "ГЛАВНЫЙ k=4 s=1.0: заряженные avg>=+1бп, "
                             "n>=1000, медиана дней>=0, заряженные>=тихие",
          "grid": {}, "primary": None}
rows_all = []
for sym, (ts, mid, sp) in syms.items():
    for k in TAKES:
        klabel = "TO" if k == 1e9 else f"{k:g}"
        for s in CUTS:
            key = f"k{klabel}_s{s:g}"
            for t_start, pnl, kind in run_sym(ts, mid, sp, k, s):
                rows_all.append((key, sym, t_start, pnl, kind))
    print(f"{sym} готов | {time.time()-t0:.0f}с", flush=True)

df = pd.DataFrame(rows_all, columns=["cell", "sym", "t", "pnl", "kind"])
df["day"] = pd.to_datetime(df["t"], unit="ms").dt.date
df["charged"] = [is_charged(s, t) for s, t in zip(df["sym"], df["t"])]

def cell_stats(d):
    if len(d) == 0:
        return {"n": 0}
    daily = d.groupby("day")["pnl"].mean()
    ch = d[d["charged"]]
    qu = d[~d["charged"]]
    return {"n": int(len(d)),
            "avg_bp": round(float(d["pnl"].mean()), 2),
            "median_day_bp": round(float(daily.median()), 2),
            "orphan_share": round(float((d["kind"] == "orphan").mean() * 100), 1),
            "charged": {"n": int(len(ch)),
                        "avg_bp": round(float(ch["pnl"].mean()), 2)
                        if len(ch) else None},
            "quiet": {"n": int(len(qu)),
                      "avg_bp": round(float(qu["pnl"].mean()), 2)
                      if len(qu) else None}}

for key, d in df.groupby("cell"):
    st = cell_stats(d)
    report["grid"][key] = st
    kk, ss = key.split("_")
    if kk == "k4" and ss == "s1":
        ch = st["charged"]
        cand = bool(st["n"] >= 1000 and ch["n"] >= 1000
                    and ch["avg_bp"] is not None and ch["avg_bp"] >= 1.0
                    and st["median_day_bp"] >= 0
                    and ch["avg_bp"] >= (st["quiet"]["avg_bp"] or -99))
        st["candidate"] = cand
        report["primary"] = st
        print(f"ГЛАВНЫЙ k=4 s=1.0: {json.dumps(st, ensure_ascii=False)} -> "
              f"{'КАНДИДАТ' if cand else 'НЕТ'}", flush=True)
    print(f"{key}: n={st.get('n')} avg={st.get('avg_bp')}бп "
          f"заряж={st.get('charged', {}).get('avg_bp')} "
          f"тих={st.get('quiet', {}).get('avg_bp')}", flush=True)

report["meta"] = {"minutes": round((time.time() - t0) / 60, 1),
                  "coins": len(syms)}
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("СТРЭДДЛ ЗАВЕРШЁН", flush=True)
