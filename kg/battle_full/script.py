"""БОЙ НА ВСЕЙ ИСТОРИИ — usdc-battle-full v1 (21.11, вопрос владельца:
«боевая версия не проверена никем и никак — бери бой, переноси на кегл,
полный тест на всей истории»).

Симуляция 1:1 воспроизводит боевой движок ml5h/engine.py (коммит
e6dc2023) с боевыми моделями (md5 сверены 21.11: a21342e4 / 80dda0a0 /
cc13e1d1 / 6018dd36).

ПРАВИЛА БОЯ — каждое из кода движка (строки в скобках):
 1) осмотр КАЖДОГО закрытого 30-минутного бара каждой монеты; кулдауна
    НЕТ (откат 18.11 окно убрал); монеты с открытой позицией/заявкой
    не осматриваются (on_bar:406);
 2) монета входит во вселенную с 300-го закрытого бара (warmup:165-167);
 3) BTCUSDT не торгуется: без btc-колонок feature_row=None (features.py
    :47,103-105) — движок молча пропускает;
 4) сырой гейт из паспорта: p > gate лонг, p < 1-gate шорт (:422);
 5) мета (env ML5H_META=1): вектор = признаки паспорта (24, с sym_id) +
    [p, atr]; вето при p_win < 0.5; мета зовётся ТОЛЬКО при конечном
    atr (:423-441);
 6) тихая треть: порог = 1/3-квантиль atr_pct ВСЕЙ ОЦЕНЁННОЙ вселенной
    бара (atrs собираются со всех оценённых монет ДО гейта, :404-421);
    атров < 30 — фильтр спит; NaN-атр кандидат режется (:449-458);
 7) сортировка по |p-0.5|, при равенстве — порядок паспорта (:462);
 8) стол 10 (ML5H_MAX_SLOTS=10), потолок урезается свободной маржей;
    кому места не хватило — сигнал ПРОПАДАЕТ (очереди нет, :469-487);
 9) честный шанс: true_prob = np.interp(p, iso_x, iso_y) из
    calibration.json; лонг t >= 0.60, шорт 1-t >= 0.60, иначе пропуск
    (:490-501); проверка ПОСЛЕ места в столе (порядок цикла :483-503);
10) вход: post-only по цене сигнального момента (бид ~ закрытие бара);
    TTL 60с; не взялась — СИГНАЛ ПРОПУСКАЕТСЯ, погони нет (:599-633,
    :966-971). Барная грануляция: заявка исполнена, если СЛЕДУЮЩИЙ бар
    монеты коснулся уровня (low<=c0 лонг / high>=c0 шорт); не коснулся
    или монета не имеет следующего бара вовремя — пропуск;
11) биржевой стоп -15% от входа: внутрибарный по high/low, срабатывает
    РАНЬШЕ будильника (place_stop:537-565; «стоп раньше решения»);
12) будильник: выход по wall-time вход + 5ч - 60с (:1013-1016) — в
    симуляции = закрытие ПЕРВОГО СУЩЕСТВУЮЩЕГО бара монеты на момент
    horizon (пропуски баров в данных — приближение);
13) издержки 12 бп на круг (мейкер-вход 2 + тейкер-выход 5 +
    проскальзывание 5) — как во всех лабах, сравнимо с cd-trio;
14) фандинг: события 00/08/16 UTC внутри удержания: net -= side*rate
    (лонг платит при rate>0); окно (вход, выход] — как в storm_meta
    (:208-221).

ДЕНЬГИ: фиксированный депозит 100 USDT (без компаундирования), сайзинг
боя: номинал 15 (15% депо), маржа 5/позиция, стол 10, дневной кап 4 с
защёлкой 18.11 (день <= -4 — новых входов нет до 00:00 UTC, открытые
позиции доезжают, :316-337). Дневной регистр по времени потока: realized
в момент закрытия, фандинг в момент события (как биржевой день боя).

ЗАФИКСИРОВАННЫЕ ПРИБЛИЖЕНИЯ (до запуска, менять после — запрещено):
 - TTL 60с -> целый следующий бар (симуляция входит чуть чаще боя);
 - стоп без проскальзывания (боевой STOP_MARKET в шторме берёт хуже);
 - частичных исполнений нет (бой закрывает их сразу — редкость);
 - позиция без данных закрывается по последней цене монеты (delist);
 - мин-нотионал пола нет (при депо 100 пол 5.5 < 15 не работает).

ОТЧЁТ: воронка фильтров; итоги сделок; по годам; учебник vs слепые дни;
деньги на депо 100. Обучающая граница модели берётся из её паспорта.
"""
import bisect
import io
import json
import os
import time
import zipfile
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

t0 = time.time()


def log(m):
    print(f"{m} | {time.time() - t0:.0f}с", flush=True)


# ---------- параметры боя (ml5h/config.py + env боевого контейнера) ----
GATE_TRUE = 0.60          # ML5H_GATE_TRUE не задан -> дефолт кода
STOP_PCT = 0.15           # emergency_stop_pct
H = 10                    # hold_bars
BAR_MS = 30 * 60_000
SLOTS = 10                # ML5H_MAX_SLOTS=10
FEE_BP = 12.0             # издержки на круг (стандарт лаб)
DEPOSIT = 100.0           # фиксированный депозит для денежной симуляции
NOTIONAL = DEPOSIT * 0.5 * 3 * 0.10          # 15 (сайзинг v3)
MARGIN_ONE = NOTIONAL / 3.0                  # 5 маржи на позицию
DAY_CAP = DEPOSIT * 0.04                     # 4 с защёлкой
WARM_MIN_BARS = 300        # warmup: монета с меньшей историей не торгуется
FUND_FILE = "funding/"


def find_input(name):
    for root, _dirs, fs in os.walk("/kaggle/input"):
        if name in fs or name in os.path.basename(root):
            return root
    return None


# ---------- боевые модели (датасет usdc-battle-models) -----------------
mroot = find_input("usdc-battle-models")
assert mroot, "датасет с боевыми моделями не найден"
passport = json.load(open(os.path.join(mroot, "ml5h_meta.json")))
FEATS = list(passport["features"])
GATE = float(passport.get("gate", 0.65))
bur = lgb.Booster(model_file=os.path.join(mroot, "ml5h.txt"))
met = lgb.Booster(model_file=os.path.join(mroot, "meta.txt"))
cal = json.load(open(os.path.join(mroot, "calibration.json")))
CAL_X = np.array(cal["iso_x"], dtype="float64")
CAL_Y = np.array(cal["iso_y"], dtype="float64")
assert bur.num_feature() == len(FEATS), "Буря/паспорт: признаки разошлись"
assert met.num_feature() == len(FEATS) + 2, "мета: вектор не 24+2"
log(f"модели: Буря {bur.num_trees()} деревьев / {len(FEATS)} фич, "
    f"мета {met.num_trees()} деревьев, карта {len(CAL_X)} точек, "
    f"гейт {GATE}, символы паспорта {len(passport['symbols'])}")

# обучающая граница модели (для разреза учебник/слепые): модель
# обучена 01.11 07:29 UTC (паспорт trained_at), ела панель конца
# октября — гарантированно слепые дни только с 1.11; свежий хвост с
# 25.10 показан отдельно (его голова могла быть в учебнике)
BLIND_FROM_MS = int(pd.Timestamp("2026-11-01", tz="UTC").value // 10**6)
FRESH_FROM_MS = int(pd.Timestamp("2026-10-25", tz="UTC").value // 10**6)
log("слепая граница: 2026-11-01 (гарантированно); свежий хвост: 2026-10-25")

# ---------- признаки: дословно ml5h/features.py (паритет с боем) -------
def sym_id(symbol):
    return zlib.crc32(symbol.encode()) & 0xFFFF


def make_features(df, btc_close, symbol):
    close = df["close"]
    high, low, vol = df["high"], df["low"], df["volume"]
    tb = df["taker_buy_base"]
    rets = close.pct_change()
    out = pd.DataFrame(index=df.index)
    for h in [1, 4, 10, 20]:
        out[f"ret_{h}"] = close.pct_change(h)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["atr_pct"] = tr.rolling(14).mean() / close
    out["std_20"] = rets.rolling(20).std()
    out["vol_ratio"] = vol / vol.rolling(20).mean().replace(0, np.nan)
    out["taker_buy"] = (tb / vol.replace(0, np.nan)).rolling(
        20, min_periods=10).mean()
    rng = (high - low).replace(0, np.nan)
    out["clv"] = ((close - low) / rng - 0.5).rolling(
        20, min_periods=10).mean()
    sma = close.rolling(200, min_periods=50).mean()
    std = close.rolling(200, min_periods=50).std()
    out["zscore"] = ((close - sma) / std).replace([np.inf, -np.inf], np.nan)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["skew"] = rets.rolling(20, min_periods=10).skew()
    sma50 = close.rolling(50).mean()
    sma20 = close.rolling(20).mean()
    out["sma_cross"] = ((close > sma50).astype(float)
                        - (close > sma20).astype(float))
    if btc_close is not None and symbol != "BTCUSDT":
        btc = btc_close.reindex(df.index).ffill()
        out["btc_ret_10"] = btc.pct_change(10)
        out["btc_ret_20"] = btc.pct_change(20)
        bret = btc.pct_change()
        out["btc_corr"] = rets.rolling(50, min_periods=20).corr(bret)
    out["h4_momentum"] = close.pct_change(48)
    ts = pd.to_datetime(df["open_time"], unit="ms")
    out["hour"] = ts.dt.hour.values
    out["dow"] = ts.dt.dayofweek.values
    return out, tr


def g_cross(df, btc_close, symbol):
    close = df["close"]
    rets = close.pct_change()
    out = pd.DataFrame(index=df.index)
    if btc_close is None or symbol == "BTCUSDT":
        out["rel_ret_20"] = 0.0
        out["rel_ret_60"] = 0.0
        out["beta_20"] = 0.0
        out["corr_20"] = 0.0
    else:
        btc = btc_close.reindex(df.index).ffill()
        bret = btc.pct_change()
        out["rel_ret_20"] = rets.rolling(20).sum() - bret.rolling(20).sum()
        out["rel_ret_60"] = rets.rolling(60).sum() - bret.rolling(60).sum()
        cov = rets.rolling(480).cov(bret)
        var = bret.rolling(480).var()
        out["beta_20"] = cov / var.replace(0, np.nan)
        out["corr_20"] = rets.rolling(480, min_periods=100).corr(bret)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


# ---------- панель ------------------------------------------------------
def find_zip(name):
    for root, _d, fs in os.walk("/kaggle/input"):
        if name in fs:
            return os.path.join(root, name)
    return None


zp = find_zip("panel.zip")
assert zp, "panel.zip не найден"
zf = zipfile.ZipFile(zp)
members = sorted(n for n in zf.namelist() if n.endswith("_30m.csv"))
log(f"панель: {len(members)} монет")

btc = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                  usecols=["open_time", "close"])
btc["open_time"] = btc["open_time"].astype(np.int64)
btc.index = pd.to_datetime(btc["open_time"], unit="ms")
btc_close = btc["close"].astype("float64")
del btc

# фандинг: funding/{sym}_funding.csv (calc_time,rate) — как в storm_meta
FUND = {}
for name in zf.namelist():
    if name.startswith(FUND_FILE) and name.endswith(".csv"):
        sym = name.split("/")[-1].replace("_funding.csv", "")
        pts = []
        for r in zf.read(name).decode().strip().split("\n")[1:]:
            c = r.split(",")
            try:
                ct = int(float(c[0]))
                if ct > 10 ** 14:
                    ct //= 1000
                pts.append((ct, float(c[1])))
            except Exception:
                pass
        pts.sort()
        FUND[zlib.crc32(sym.encode()) & 0xFFFF] = pts
log(f"фандинг: {len(FUND)} монет")

# монеты = паспорт ∩ панель (как вселенная боя: self.symbols из паспорта)
SYMS = [s for s in passport["symbols"]
        if f"{s}_30m.csv" in zf.namelist()]
log(f"вселенная: {len(SYMS)} из паспорта {len(passport['symbols'])} "
    f"имеются в панели")

coins = {}          # sym -> dict(ot, hi, lo, cl, atr, p, F)
ot_all, sid_all, gri_all = [], [], []
hi_all, lo_all, cl_all, atr_all, p_all = [], [], [], [], []
CODE = {}
surv = 0          # индекс в списке ВЫЖИВШИХ монет — глобальные sid
                  # обязаны совпадать с ним (SYMS ниже пересобирается
                  # только из выживших; нумерация паспорта с пропусками
                  # разъезжалась бы на два шага — баг v2, IndexError)
for i, sym in enumerate(SYMS):
    if sym == "BTCUSDT":
        continue          # бой: без btc-колонок feature_row=None (:103) —
                          # BTC никогда не оценивается и не торгуется
    code = sym_id(sym)
    CODE[sym] = code
    df = pd.read_csv(io.BytesIO(zf.read(f"{sym}_30m.csv")),
                     usecols=["open_time", "high", "low", "close",
                              "volume", "taker_buy_base"])
    df["open_time"] = df["open_time"].astype(np.int64)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    n = len(df)
    if n < WARM_MIN_BARS + 1:
        continue                     # бой: <300 баров — монеты нет
    X, tr = make_features(df, btc_close, sym)
    Xg = g_cross(df, btc_close, sym)
    for c in Xg.columns:
        X[c] = Xg[c]
    X["sym_id"] = float(sym_id(sym))    # feature_row:103 — метка монеты
    X = X.replace([np.inf, -np.inf], np.nan)
    F = X[FEATS].to_numpy(dtype="float32")       # порядок паспорта
    p = bur.predict(F.astype("float64")).astype("float64")
    atr = (tr.rolling(14).mean() / df["close"]).to_numpy(dtype="float64")
    # строки с 300-го бара — до этого монета не во вселенной (warmup)
    keep = np.arange(n) >= WARM_MIN_BARS
    ot = df["open_time"].to_numpy(dtype="int64")[keep]
    c = dict(sym=sym, code=code, n=int(keep.sum()),
             ot=ot,
             hi=df["high"].to_numpy(dtype="float32")[keep],
             lo=df["low"].to_numpy(dtype="float32")[keep],
             cl=df["close"].to_numpy(dtype="float32")[keep],
             atr=atr[keep], p=p[keep], F=F[keep])
    coins[sym] = c
    ot_all.append(ot)
    sid_all.append(np.full(len(ot), surv, dtype=np.int32))
    surv += 1
    gri_all.append(np.arange(len(ot), dtype=np.int32))
    hi_all.append(c["hi"]); lo_all.append(c["lo"]); cl_all.append(c["cl"])
    atr_all.append(c["atr"].astype("float32"))
    p_all.append(c["p"].astype("float32"))
    if (i + 1) % 100 == 0:
        log(f"признаки {i + 1}/{len(SYMS)}")
SYMS = list(coins.keys())          # порядок паспорта, только живые
CODE = {s: coins[s]["code"] for s in SYMS}

GOT = np.concatenate(ot_all)
order = np.argsort(GOT, kind="stable")     # стабильный: порядок паспорта
GOT = GOT[order]
GSID = np.concatenate(sid_all)[order]
GRI = np.concatenate(gri_all)[order]
GHI = np.concatenate(hi_all)[order]
GLO = np.concatenate(lo_all)[order]
GCL = np.concatenate(cl_all)[order]
GATR = np.concatenate(atr_all)[order].astype("float64")
GP = np.concatenate(p_all)[order].astype("float64")
del ot_all, sid_all, gri_all, hi_all, lo_all, cl_all, atr_all, p_all
uniq_ots, first_idx = np.unique(GOT, return_index=True)
# САННИТИ: каждая глобальная строка обязана указывать ровно на свой
# бар своей монеты (баг v2 — сдвиг нумерации — здесь ловится сразу)
import random as _random
_chk_n = min(2000, len(GOT))
for gi in _random.Random(7).sample(range(len(GOT)), _chk_n):
    cn_s = coins[SYMS[int(GSID[gi])]]
    _ri = int(GRI[gi])
    assert int(cn_s["ot"][_ri]) == int(GOT[gi]), \
        f"глобальная строка {gi} разошлась со своей монетой"
    assert float(cn_s["cl"][_ri]) == float(GCL[gi]), \
        f"цены разошлись на строке {gi}"
log(f"санити стыковки: {_chk_n}/{_chk_n} строк указывают на свои бары")
log(f"строки {len(GOT):,} | баров {len(uniq_ots):,} "
    f"({pd.Timestamp(int(uniq_ots[0]), unit='ms', tz='UTC')} .. "
    f"{pd.Timestamp(int(uniq_ots[-1]), unit='ms', tz='UTC')})")

# ---------- скан: бой бар за баром --------------------------------------
funnel = dict(bars=0, cand=0, meta_veto=0, quiet_cut=0, honest_skip=0,
              slot_skip=0, cap_block=0, orders=0, unfilled=0, trades=0)
positions = {}     # sid -> pos dict
pending = {}       # sid -> order dict (живёт ровно один бар)
trades = []
ledger = {}        # UTC-день (номер) -> деньги (realized + фандинг) —
                   # ЖИВОЙ: наполняется в момент закрытия сделки, кап
                   # внутри скана читает его настоящим (баг v4: регистр
                   # заполнялся после скана и кап был мёртв)
cap_latched_day = None


def settle(tr):
    """Расчёт сделки при закрытии: bp, фандинг, живой дневной регистр.
    Фандинг: события (вход, выход]; будильник закрывается за 60с до
    close бара горизонта — его окно короче на эти 60с (стоп/делист
    закрываются внутри бара — до его закрытия)."""
    side = tr["side"]
    gross_bp = side * (tr["exit_px"] / tr["c0"] - 1) * 1e4
    hi_ms = tr["exit_T"] - 60_000 if tr["kind"] == "budilnik" \
        else tr["exit_T"]
    paid = 0.0
    fevents = []
    fpts = FUND.get(tr["code"], [])
    if fpts:
        lo_b = bisect.bisect_right(fpts, (tr["entry_T"], 10 ** 9))
        hi_b = bisect.bisect_right(fpts, (hi_ms, 10 ** 9))
        for _t, _r in fpts[lo_b:hi_b]:
            paid += side * _r
            fevents.append((_t, -side * _r * NOTIONAL))
    tr["gross_bp"] = gross_bp
    tr["fee_bp"] = FEE_BP
    tr["fund_bp"] = paid * 1e4
    tr["net_bp"] = gross_bp - FEE_BP - paid * 1e4
    tr["win"] = tr["net_bp"] > 0
    tr["blind"] = tr["entry_T"] >= BLIND_FROM_MS
    dt = pd.Timestamp(tr["entry_T"], unit="ms", tz="UTC")
    tr["year"] = dt.year
    tr["month"] = dt.strftime("%Y-%m")
    tr["fevents"] = fevents
    # живой регистр дня: realized БЕЗ фандинга в день закрытия (фандинг
    # уже сидит в net_bp — класть его в регистр второй раз нельзя,
    # баг v5: двойной счёт фандинга раздувал худшие дни), фандинг —
    # в день своего события (как биржевой день боя)
    d_exit = day_of(tr["exit_T"])
    ledger[d_exit] = ledger.get(d_exit, 0.0) \
        + NOTIONAL * (gross_bp - FEE_BP) / 1e4
    for ft, famt in fevents:
        d_f = day_of(ft)
        ledger[d_f] = ledger.get(d_f, 0.0) + famt


def true_prob(p):
    return float(np.interp(p, CAL_X, CAL_Y))


def day_of(ms):
    # целочисленный номер UTC-дня (быстро, без Timestamp на каждый бар)
    return int(ms) // 86_400_000


def day_str(dn):
    return pd.Timestamp(dn * 86_400_000, unit="ms", tz="UTC").strftime(
        "%Y-%m-%d")


for bi in range(len(uniq_ots)):
    T = int(uniq_ots[bi])
    lo_i = int(first_idx[bi])
    hi_i = int(first_idx[bi + 1]) if bi + 1 < len(uniq_ots) else len(GOT)
    funnel["bars"] += 1

    # --- 1) развязка заявок прошлого бара (TTL 60с -> этот бар) --------
    for sid in list(pending.keys()):
        od = pending[sid]
        sym = od["sym"]
        cn = coins[sym]
        ri_next = od["ri"] + 1
        touched = False
        if ri_next < cn["n"] and int(cn["ot"][ri_next]) == T:
            c0 = od["c0"]
            if od["side"] == 1:
                touched = float(cn["lo"][ri_next]) <= c0
            else:
                touched = float(cn["hi"][ri_next]) >= c0
        if touched:
            # позиция открыта по c0 (мейкер); стоп жив с этого момента:
            # если ЭТОТ ЖЕ бар коснулся стопа — худший случай, стоп
            stop_px = c0 * (1 - STOP_PCT) if od["side"] == 1 \
                else c0 * (1 + STOP_PCT)
            hit = False
            if od["side"] == 1:
                hit = float(cn["lo"][ri_next]) <= stop_px
            else:
                hit = float(cn["hi"][ri_next]) >= stop_px
            if hit:
                tr = dict(sym=sym, code=od["code"], side=od["side"],
                          entry_T=od["T"], exit_T=T, c0=od["c0"],
                          p=od["p"],
                          exit_px=stop_px, kind="stop", entry_ri=od["ri"],
                          exit_ri=ri_next)
                trades.append(tr)
                settle(tr)
                funnel["trades"] += 1
            else:
                positions[sid] = dict(sym=sym, code=od["code"],
                                      side=od["side"], c0=od["c0"],
                                      p=od["p"],
                                      entry_T=od["T"], ri=ri_next,
                                      entry_ri=od["ri"],
                                      horizon_T=od["T"] + H * BAR_MS - 60_000,
                                      stop_px=stop_px)
        else:
            funnel["unfilled"] += 1      # бой: не взялась за TTL — пропуск
        pending.pop(sid, None)

    # --- 2) стоп/будильник по ЭТОМУ бару (стоп раньше решения) ---------
    for sid in list(positions.keys()):
        pos = positions[sid]
        sym = pos["sym"]
        cn = coins[sym]
        ri = pos["ri"] + 1
        if ri >= cn["n"] or int(cn["ot"][ri]) != T:
            continue                     # данных по монете на этот бар нет
        pos["ri"] = ri
        if pos["side"] == 1 and float(cn["lo"][ri]) <= pos["stop_px"]:
            tr = dict(sym=sym, code=pos["code"], side=pos["side"],
                      entry_T=pos["entry_T"], exit_T=T, c0=pos["c0"],
                      p=pos["p"],
                      exit_px=pos["stop_px"], kind="stop",
                      entry_ri=pos["entry_ri"], exit_ri=ri)
            trades.append(tr)
            settle(tr)
            funnel["trades"] += 1
            positions.pop(sid, None)
            continue
        if pos["side"] == -1 and float(cn["hi"][ri]) >= pos["stop_px"]:
            tr = dict(sym=sym, code=pos["code"], side=pos["side"],
                      entry_T=pos["entry_T"], exit_T=T, c0=pos["c0"],
                      p=pos["p"],
                      exit_px=pos["stop_px"], kind="stop",
                      entry_ri=pos["entry_ri"], exit_ri=ri)
            trades.append(tr)
            settle(tr)
            funnel["trades"] += 1
            positions.pop(sid, None)
            continue
        if T >= pos["horizon_T"]:
            tr = dict(sym=sym, code=pos["code"], side=pos["side"],
                      entry_T=pos["entry_T"], exit_T=T, c0=pos["c0"],
                      p=pos["p"],
                      exit_px=float(cn["cl"][ri]), kind="budilnik",
                      entry_ri=pos["entry_ri"], exit_ri=ri)
            trades.append(tr)
            settle(tr)
            funnel["trades"] += 1
            positions.pop(sid, None)

    # --- 3) скоринг бара (дословно on_bar) -----------------------------
    sids = GSID[lo_i:hi_i]
    atrs = GATR[lo_i:hi_i]
    ps = GP[lo_i:hi_i]
    gris = GRI[lo_i:hi_i]
    busy = set(positions.keys()) | set(pending.keys())
    elig = np.fromiter((int(s) not in busy for s in sids),
                       dtype=bool, count=len(sids))
    # вселенная бара для квантиля: ВСЕ оценённые монеты с конечным atr
    ea = atrs[elig]
    ea = ea[np.isfinite(ea)]
    quiet_on = len(ea) >= 30
    q33 = float(np.quantile(ea, 1 / 3)) if quiet_on else 0.0
    cand_mask = elig & ((ps > GATE) | (ps < 1 - GATE))
    funnel["cand"] += int(cand_mask.sum())
    if not cand_mask.any():
        continue
    cidx = np.flatnonzero(cand_mask)
    # мета: вектор = 24 фич паспорта (с sym_id) + [p, atr]; только
    # конечный atr (движок:423-441)
    rows_meta = []
    for k in cidx:
        if np.isfinite(float(atrs[k])):
            rows_meta.append(k)
    if rows_meta:
        Fm = np.empty((len(rows_meta), len(FEATS) + 2), dtype="float64")
        for j, k in enumerate(rows_meta):
            cn = coins[SYMS[int(sids[k])]]
            Fm[j, :len(FEATS)] = cn["F"][int(gris[k])]
            Fm[j, len(FEATS)] = ps[k]
            Fm[j, len(FEATS) + 1] = atrs[k]
        # движок:429-430 — None/NaN в векторе меты заменяются нулём
        Fm = np.where(np.isfinite(Fm), Fm, 0.0)
        pw = met.predict(Fm)
        vetod = {rows_meta[j] for j in range(len(rows_meta))
                 if float(pw[j]) < 0.5}
        funnel["meta_veto"] += len(vetod)
    else:
        vetod = set()
    cand = []
    for k in cidx:
        if k in vetod:
            continue
        a = float(atrs[k])
        if quiet_on:
            if not np.isfinite(a) or a < q33:
                funnel["quiet_cut"] += 1
                continue
        cand.append((float(abs(ps[k] - 0.5)), float(ps[k]), k))
    # сортировка по уверенности, ties — порядок паспорта (стабильный)
    cand.sort(key=lambda t: t[0], reverse=True)
    # стол: позиции + живые заявки; потолок урезается свободной маржей
    slots = len(positions) + len(pending)
    margin_used = MARGIN_ONE * len(positions)
    avail = DEPOSIT - margin_used
    slot_cap = SLOTS
    if MARGIN_ONE > 0 and avail < MARGIN_ONE:
        slot_cap = min(SLOTS, int(avail // MARGIN_ONE))
        if slot_cap < slots:
            slot_cap = slots          # уже открытое не считается лишним
    today = day_of(T)
    if cap_latched_day is not None and cap_latched_day != today:
        cap_latched_day = None        # новый день — защёлка сброшена
    day_pnl_now = ledger.get(today, 0.0)
    for _conf, p_v, k in cand:
        if slots >= slot_cap:
            funnel["slot_skip"] += 1
            continue
        t_true = true_prob(p_v)
        side = 1 if p_v > GATE else -1
        if side == 1 and t_true < GATE_TRUE:
            funnel["honest_skip"] += 1
            continue
        if side == -1 and 1 - t_true < GATE_TRUE:
            funnel["honest_skip"] += 1
            continue
        # дневной кап с защёлкой (деньги — слой боя)
        if cap_latched_day == today or day_pnl_now <= -DAY_CAP:
            if day_pnl_now <= -DAY_CAP:
                cap_latched_day = today
                funnel["cap_block"] += 1
            else:
                funnel["cap_block"] += 1
            continue
        k_i = int(k)
        sym = SYMS[int(sids[k_i])]
        cn = coins[sym]
        ri_k = int(gris[k_i])
        pending[int(sids[k_i])] = dict(
            sym=sym, code=CODE[sym], side=side, c0=float(cn["cl"][ri_k]),
            T=T, ri=ri_k, p=float(p_v))
        slots += 1
        funnel["orders"] += 1

log("скан завершён")

# ---------- закрытие хвостов позиций (делист/конец данных) -------------
n_tails = 0
for sid, pos in positions.items():
    cn = coins[pos["sym"]]
    last_ri = cn["n"] - 1
    raw_px = float(cn["cl"][last_ri])
    # биржевой стоп боя жив до делистинга: цена выхода не хуже стопа
    if pos["side"] == 1:
        exit_px = max(raw_px, pos["stop_px"])
    else:
        exit_px = min(raw_px, pos["stop_px"])
    tr = dict(sym=pos["sym"], code=pos["code"], side=pos["side"],
              entry_T=pos["entry_T"], exit_T=int(cn["ot"][last_ri]),
              c0=pos["c0"], p=pos["p"], exit_px=exit_px, kind="delist",
              entry_ri=pos["entry_ri"], exit_ri=last_ri)
    trades.append(tr)
    settle(tr)
    funnel["trades"] += 1
    n_tails += 1
log(f"хвостов закрыто: {n_tails}")

# ---------- отчёт (сделки уже рассчитаны settle'ом) ------
n = len(trades)
wins = sum(1 for t in trades if t["win"])
stopped = sum(1 for t in trades if t["kind"] == "stop")
net_sum = float(sum(t["net_bp"] for t in trades))
fund_sum = float(sum(t["fund_bp"] for t in trades))
kinds = {}
for t in trades:
    kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1

def agg(rows):
    if not rows:
        return dict(n=0, win=0.0, avg=0.0, sum_bp=0.0, stops=0)
    w = sum(1 for t in rows if t["win"])
    return dict(n=len(rows), win=round(w / len(rows) * 100, 1),
                avg=round(float(np.mean([t["net_bp"] for t in rows])), 1),
                sum_bp=round(float(sum(t["net_bp"] for t in rows)), 0),
                stops=sum(1 for t in rows if t["kind"] == "stop"))

by_year = {}
for t in trades:
    by_year.setdefault(t["year"], []).append(t)
blind_rows = [t for t in trades if t["blind"]]
insample_rows = [t for t in trades if not t["blind"]]
fresh_rows = [t for t in trades if t["entry_T"] >= FRESH_FROM_MS]

days = sorted(ledger.keys())
daily = [ledger[d] for d in days]
cum = np.cumsum(daily)
peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
dd = cum - peak
maxdd = float(dd.min()) if len(dd) else 0.0
red_days = sum(1 for v in daily if v < 0)
streak = worst = 0
for v in daily:
    if v < 0:
        streak += 1
        worst = max(worst, streak)
    else:
        streak = 0
months_ledger = {}
for d, v in ledger.items():
    mkey = day_str(d)[:7]
    months_ledger.setdefault(mkey, 0.0)
    months_ledger[mkey] += v
month_vals = sorted(months_ledger.values())
pos_months = sum(1 for v in month_vals if v > 0)

report = dict(
    pre_reg="БОЙ 1:1 на всей истории: без кулдауна, мета 24+2 с sym_id, "
            "тихая треть по вселенной бара, карта 6018dd36, стоп раньше "
            "будильника, издержки 12бп + фандинг; депозит 100 фикс",
    models=dict(burya="a21342e4", passport="80dda0a0", meta="cc13e1d1",
                calibration="6018dd36"),
    gate=GATE, gate_true=GATE_TRUE, slots=SLOTS, fee_bp=FEE_BP,
    deposit=DEPOSIT,
    blind_from=pd.Timestamp(BLIND_FROM_MS, unit="ms", tz="UTC").strftime(
        "%Y-%m-%d"),
    funnel=funnel,
    totals=dict(trades=n, win=round(wins / max(1, n) * 100, 1),
                avg_bp=round(net_sum / max(1, n), 1),
                median_bp=round(float(np.median([t["net_bp"] for t in trades]))
                                if n else 0.0, 1),
                sum_bp=round(net_sum, 0), stops=stopped,
                stop_pct=round(stopped / max(1, n) * 100, 1),
                funding_bp=round(fund_sum, 0),
                kinds=kinds),
    by_year={str(y): agg(rows) for y, rows in sorted(by_year.items())},
    in_sample=agg(insample_rows),
    blind=agg(blind_rows),
    fresh_tail=agg(fresh_rows),
    money=dict(
        deposit=DEPOSIT, notional=NOTIONAL, day_cap=DAY_CAP,
        final=round(float(cum[-1]) if len(cum) else 0.0, 2),
        days=len(days), red_days=red_days,
        red_pct=round(red_days / max(1, len(days)) * 100, 1),
        worst_streak=worst, maxdd=round(maxdd, 2),
        best_day=round(max(daily) if daily else 0.0, 2),
        worst_day=round(min(daily) if daily else 0.0, 2),
        months=len(month_vals), months_pos=pos_months,
        months_pos_pct=round(pos_months / max(1, len(month_vals)) * 100, 1),
        cap_blocks=funnel["cap_block"]),
    monthly={k: round(v, 2) for k, v in sorted(months_ledger.items())},
)
with open("/kaggle/working/battle_full_report.json", "w") as f:
    json.dump(report, f, indent=1, default=float)

# ПОСТРОЧНАЯ выгрузка сделок: независимая сверка с судьёй и аудит
with open("/kaggle/working/battle_full_trades.jsonl", "w") as f:
    for t in trades:
        f.write(json.dumps(dict(
            sym=t["sym"], side="LONG" if t["side"] == 1 else "SHORT",
            entry_ms=int(t["entry_T"]), exit_ms=int(t["exit_T"]),
            entry_px=float(t["c0"]), exit_px=float(t["exit_px"]),
            kind=t["kind"], p=float(t["p"]),
            gross_bp=float(t["gross_bp"]), fee_bp=float(t["fee_bp"]),
            fund_bp=float(t["fund_bp"]), net_bp=float(t["net_bp"]),
            win=bool(t["win"]), year=int(t["year"]), month=t["month"],
        ), default=float) + "\n")

print("\n=== ВОРОНКА ===")
print(f"баров {funnel['bars']:,} | кандидатов {funnel['cand']:,} | "
      f"-мета {funnel['meta_veto']:,} | -тихая {funnel['quiet_cut']:,} | "
      f"-честный {funnel['honest_skip']:,} | -слоты {funnel['slot_skip']:,} | "
      f"-кап {funnel['cap_block']:,} | заявок {funnel['orders']:,} | "
      f"не взялись {funnel['unfilled']:,} | сделок {n:,}")
print("\n=== ИТОГИ ===")
tt = report["totals"]
print(f"сделок {tt['trades']} | win {tt['win']}% | avg {tt['avg_bp']}бп | "
      f"медиана {tt['median_bp']}бп | сумма {tt['sum_bp']:.0f}бп | "
      f"стопов {tt['stops']} ({tt['stop_pct']}%) | фандинг "
      f"{tt['funding_bp']:.0f}бп")
print(f"виды выхода: {tt['kinds']}")
print("\n=== ПО ГОДАМ ===")
print("год | сделок | win% | avg бп | сумма бп | стопов")
for y, a in report["by_year"].items():
    print(f"{y} | {a['n']} | {a['win']} | {a['avg']} | {a['sum_bp']:.0f} | "
          f"{a['stops']}")
print(f"\nучебник (до {report['blind_from']}): "
      f"{report['in_sample']['n']} сд, win {report['in_sample']['win']}%, "
      f"avg {report['in_sample']['avg']}бп")
print(f"свежий хвост (с 25.10, голова могла быть в учебнике): "
      f"{report['fresh_tail']['n']} сд, win {report['fresh_tail']['win']}%, "
      f"avg {report['fresh_tail']['avg']}бп")
print(f"слепые дни (с 1.11, гарантированно): "
      f"{report['blind']['n']} сд, win {report['blind']['win']}%, "
      f"avg {report['blind']['avg']}бп")
mm = report["money"]
print("\n=== ДЕНЬГИ (депозит 100 фикс) ===")
print(f"итог {mm['final']:+.2f} за {mm['days']} дней | красных "
      f"{mm['red_days']} ({mm['red_pct']}%) | худшая серия {mm['worst_streak']} | "
      f"maxDD {mm['maxdd']} | лучший день {mm['best_day']:+.2f} | "
      f"худший {mm['worst_day']:+.2f}")
print(f"месяцев {mm['months']}, положительных {mm['months_pos']} "
      f"({mm['months_pos_pct']}%) | кап срабатывал {mm['cap_blocks']} раз")
log("БОЙ НА ВСЕЙ ИСТОРИИ ГОТОВ")
