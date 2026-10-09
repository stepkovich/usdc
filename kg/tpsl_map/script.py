"""КАРТА ВЫХОДОВ: ТЕЙК x СТОП x ВРЕМЯ (предрегистрация 07.10).

ВОПРОС ВЛАДЕЛЬЦА: где идеальный баланс тейк/стоп? Ответ строим как
поверхность: сетка тейк {0.5,1,2} x spread, стоп {0.5,1,2} x spread,
время {60с,120с,300с} = 27 ячеек, все на ОДНИХ И ТЕХ ЖЕ циклах.

ЦИКЛ (механика без модели, максимальная выборка): вход мейкером на
биде (касание бида за 60с, иначе отмена), дальше путь цены по секундам
до 345с: первое касание тейк-уровня -> плюс k1 x spread (мейкер 0),
первое касание стоп-уровня -> минус k2 x spread (мейкер 0, лимитка на
уровне исполняется касанием), таймаут -> рынок по последней цене
(минус 10 бп). Пауза после цикла 45с. Гонка в одну секунду = тейк.

ДИСЦИПЛИНА: поверхность рисуется ТОЛЬКО на тренировочных днях (все
до тестовых); лучшая ячейка (max avg при n>=1000) проверяется на
СВЕЖИХ тест-днях (последние 4), которых карта не видела.
ВЕРДИКТ (записан до прогона): ячейка ЖИВА, если на тесте avg >= +1 бп
при n >= 300. Ничего не деплоится. Оговорка: симуляция 1с-мидов не
видит очередь — результат оптимистичен, планка +1 бп с запасом.
"""
import glob, json, time
import numpy as np, pandas as pd

t0 = time.time()
N_TEST_DAYS = 4
TPK = [0.5, 1.0, 2.0]
SLK = [0.5, 1.0, 2.0]
TTLS = [60_000, 120_000, 300_000]
SYMS = ["1000BONKUSDC", "1000PEPEUSDC", "1000SHIBUSDC", "AAVEUSDC",
        "ADAUSDC", "ARBUSDC", "AVAXUSDC", "BCHUSDC", "BIOUSDC",
        "BNBUSDC", "BOMEUSDC", "BTCUSDC", "CRVUSDC", "DATAIPUSDC",
        "DOGEUSDC", "ENAUSDC", "ETHFIUSDC", "ETHUSDC", "FILUSDC",
        "HBARUSDC", "KAITOUSDC", "LINKUSDC", "LTCUSDC", "NEARUSDC",
        "NEOUSDC", "ORDIUSDC", "PENGUUSDC", "PNUTUSDC", "SOLUSDC",
        "SUIUSDC", "TIAUSDC", "TRUMPUSDC", "UNIUSDC", "WIFUSDC",
        "WLDUSDC", "WLFIUSDC", "XRPUSDC", "ZECUSDC"]

book_frames = []
for f in sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True)):
    try:
        bf = pd.read_parquet(f, columns=["ts", "symbol", "mid",
                                         "spread_bp"])
        book_frames.append(bf.iloc[::2])          # 1с каденс
    except Exception:
        pass
book = pd.concat(book_frames, ignore_index=True)
del book_frames
book = book.dropna(subset=["mid", "spread_bp"])
book["day"] = pd.to_datetime(book["ts"], unit="ms").dt.date.astype(str)
days = sorted(book["day"].unique())
test_days = days[-N_TEST_DAYS:]
print(f"строк {len(book)}, дней {len(days)}, тест {test_days} "
      f"| {time.time()-t0:.0f}с", flush=True)

paths = []
for sym, g in book.groupby("symbol"):
    if sym not in SYMS:
        continue
    g = g.sort_values("ts").reset_index(drop=True)
    ts = g["ts"].values
    mid = g["mid"].values.astype("float64")
    sp = g["spread_bp"].values
    day = g["day"].values
    n = len(g)
    i = 0
    while i < n:
        s = sp[i]
        entry_px = mid[i] * (1 - s / 2 / 10000)
        j = i
        f = None
        while j < n and ts[j] <= ts[i] + 60_000:
            if mid[j] <= entry_px:
                f = j
                break
            j += 1
        if f is None:
            i += 1
            continue
        k_end = int(np.searchsorted(ts, ts[f] + 345_000))
        path = mid[f:k_end].astype("float64")
        paths.append({"day": str(day[i]), "entry": entry_px, "s": s,
                      "path": path})
        i = k_end + 45
print(f"циклов {len(paths)} | {time.time()-t0:.0f}с", flush=True)

cycle_res = []
for p in paths:
    s_frac = p["s"] / 1e4                    # спред цикла в долях
    tp_idx = []
    for k in TPK:
        w = np.nonzero(p["path"] >= p["entry"] * (1 + k * s_frac))[0]
        tp_idx.append(int(w[0]) if len(w) else 10**9)
    sl_idx = []
    for k in SLK:
        w = np.nonzero(p["path"] <= p["entry"] * (1 - k * s_frac))[0]
        sl_idx.append(int(w[0]) if len(w) else 10**9)
    last = p["path"][-1] if len(p["path"]) else np.nan
    cycle_res.append((p["day"], p["entry"], p["s"], len(p["path"]),
                      tp_idx, sl_idx, last))
print(f"касания посчитаны | {time.time()-t0:.0f}с", flush=True)

def eval_cell2(cycles, tp_i, sl_i, ttl_s):
    """Ячейка: тейк-уровень entry*(1+TPP), стоп entry*(1-SPP).
    Первое касание решает: тейк -> +TPP[tp_i] (мейкер 0), стоп ->
    -SPP[sl_i] (мейкер 0), таймаут -> рынок (минус 10 бп).
    ttl_s приходит УЖЕ В СЕКУНДАХ (ядерный цикл конвертирует мс)."""
    cut = int(ttl_s)
    pnls = np.full(len(cycles), np.nan)
    for r, (day, entry, s, plen, tps, sls, last) in enumerate(cycles):
        t = tps[tp_i]
        u = sls[sl_i]
        lim = min(cut, plen)
        t_hit = t < lim
        u_hit = u < lim
        if t_hit and (not u_hit or t <= u):
            pnls[r] = TPK[tp_i] * s / 1e4             # +тейк, мейкер 0
        elif u_hit and (not t_hit or u < t):
            pnls[r] = -SLK[sl_i] * s / 1e4            # -стоп, мейкер 0
        else:
            pnls[r] = (last / entry - 1) - 0.0010     # таймаут, рынок
    return pnls

res_table = []
tr_cyc = [c for c in cycle_res if c[0] < test_days[0]]
te_cyc = [c for c in cycle_res if c[0] >= test_days[0]]
print(f"циклов: train {len(tr_cyc)}, test {len(te_cyc)}", flush=True)
for tp_k in TPK:
    for sl_k in SLK:
        for ttl in TTLS:
            ti = TPK.index(tp_k)
            si = SLK.index(sl_k)
            pn_tr = eval_cell2(tr_cyc, ti, si, int(ttl / 1000))
            pn_te = eval_cell2(te_cyc, ti, si, int(ttl / 1000))
            tr_avg = float(np.nanmean(pn_tr) * 10000) if len(pn_tr) \
                else float("nan")
            te_avg = float(np.nanmean(pn_te) * 10000) if len(pn_te) \
                else float("nan")
            res_table.append({"tp": tp_k, "sl": sl_k,
                              "ttl_s": int(ttl / 1000),
                              "train_n": len(pn_tr),
                              "train_avg_bp": round(tr_avg, 2),
                              "test_n": len(pn_te),
                              "test_avg_bp": round(te_avg, 2)})
            print(f"тйк {tp_k} x стп {sl_k} x {int(ttl/1000):3d}с: "
                  f"train {tr_avg:+8.2f} бп (n={len(pn_tr)}) | "
                  f"test {te_avg:+8.2f} бп (n={len(pn_te)})", flush=True)

valid = [r for r in res_table if r["train_n"] >= 1000]
best = max(valid, key=lambda r: r["train_avg_bp"])
print(f"\nЛУЧШАЯ НА ТРЕНИРОВКЕ: тейк {best['tp']}xspread, стоп "
      f"{best['sl']}xspread, {best['ttl_s']}с -> train "
      f"{best['train_avg_bp']:+.2f} бп", flush=True)
print(f"ПРОВЕРКА НА СВЕЖИХ ДНЯХ: {best['test_avg_bp']:+.2f} бп "
      f"(n={best['test_n']})", flush=True)
alive = bool(best["test_avg_bp"] >= 1.0 and best["test_n"] >= 300)
report = {"grid": res_table, "best_train": best,
          "verdict": ("ЯЧЕЙКА ЖИВА" if alive else
                      "поверхность под водой и на свежих днях"),
          "alive": alive}
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ВЕРДИКТ:", report["verdict"], flush=True)
print("КАРТА ВЫХОДОВ завершена", flush=True)
