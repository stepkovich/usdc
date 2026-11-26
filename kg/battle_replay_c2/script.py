"""САМАЯ ЧЕСТНАЯ РЕПЕТИЦИЯ — usdc-battle-replay v1 (21.11, решение
владельца после usdc-battle-full: «взять честный код боя, но учить
модель в процессе — так как было бы это в бою — и считать деньги по
всему балансу»).

Бой целиком (usdc-battle-full v7, проверен судьёй/живым боем/синтетикой)
+ модель учится В ПРОЦЕССЕ, месяц за месяцем, как живёт боевой конвейер:

  1) каждый месяц M Буря обучается С НУЛЯ на ВСЕХ закрытых данных до
     конца M-1 (рецепт боевого тренера kg/ml5h_wide ДОСЛОВНО: LightGBM
     200 деревьев, depth 5, lr 0.05, subsample/colsample 0.8,
     class_weight balanced, sym_id категорией; метки — вопрос бури
     ±0.5 ATR за 10 баров; месяц M этими данными НЕ видел);
  2) мета каждый месяц переучивается на тетради сигналов ПРОШЛЫХ
     месяцев репетиции (рецепт storm_meta: 120 деревьев, depth 3,
     leaves 7); включается с 1000 сигналов в тетради;
  3) карта калибровки (честный шанс 60%) строится как в боевом тренере:
     изотоника на НЕЗРЯЧИХ предсказаниях и ЯРЛЫКАХ НАПРАВЛЕНИЯ всех
     строк прошлых месяцев (не исходах сделок! ярлык сделки у шорта
     перевёрнут относительно направления — v1 с исходами сделок
     сплющивала карту и резала обе стороны: июль 2023 = 1773 сырых
     сигнала и 0 сделок); включается с 500 записей — с третьего-второго
     торгуемого месяца;
  4) вся цепочка боя — из v7 без изменений: осмотр каждого бара без
     кулдауна, сырой гейт 0.65 из паспорта (0.65 константа рецепта),
     вектор меты 24+2 с NaN->0 (engine:429), тихая треть по вселенной
     бара, сортировка по уверенности, стол 10, вход = касание уровня
     следующим баром, стоп -15% раньше будильника, издержки 12бп,
     фандинг по событиям (вход, выход];
  5) ДЕНЬГИ: полный компаундинг — баланс B=100 на старте, номинал
     позиции = 15% ТЕКУЩЕГО B, маржа 5%, дневной кап 4% ТЕКУЩЕГО B с
     защёлкой; каждый закрытый сделкой/фандингом доллар меняет B.
     РЯДОМ (для сравнения с прошлыми замерами) те же сделки по
     ФИКСИРОВАННОМУ номиналу 15 (депозит-сотня без компаундинга);
  6) месяцы торгуются с 2020-02 (2020-01 = прогрев) при >= 30000
     обучающих строк; позиция переносится через границу месяца
     (месяц не сбрасывает стол);
  7) дымовой извещатель: каждый месяц логируется, даёт ли модель
     сигналы (журнал, без гейта — решение владельца: репетиция 16.11
     показала 82 месяца без катастроф).

ПОДТВЕРЖДЕНИЕ + ПАУЗА ПОСЛЕ ВЫХОДА (ТЕСТ-2, схема владельца 23.11:
«смотрим каждые 30 мин; вышли — в монету 5ч не входим, даже если
сигнал есть»): сигналы из блока не считаются, после блока цепь
подтверждения начинается заново. Сравнение: v2 +24.0 / v3 решётка
+27.8 / ТЕСТ-1 — три колонки на тех же моделях.

ЗАФИКСИРОВАННЫЕ ОТКЛОНЕНИЯ ОТ БОЯ (до запуска):
 - экзамен-гейт не гоняется (8 фолдов x 82 месяца = 738 обучений,
     кернел не уложится; замена — извещатель-журнал);
 - карта репетиции строится на предсказаниях прошлых месяцев, у боя —
     на фолдах текущей модели (дух один: незрячие предсказания);
 - TTL 60с -> целый следующий бар (как в v7; сим входит ~в 1.8 раза
     чаще живого боя — деньги = верхняя оценка);
 - стоп без проскальзывания; делист защищён стопом.

ОЖИДАНИЯ (зафиксированы ДО запуска, менять после — запрещено):
 средняя на сделку должна выйти в диапазоне +48..+66бп (между
 cd-trio/без-КД и ходячими экзаменами). Если выйдет ~90+ как на
 учебнике v7 — сигнал утечки: остановиться и искать, не показывать
 как честный результат.
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
from sklearn.isotonic import IsotonicRegression

t0 = time.time()


def log(m):
    print(f"{m} | {time.time() - t0:.0f}с", flush=True)


# ---------- рецепт боя (ml5h/config.py + env) и тренера (ml5h_wide) ----
GATE = 0.65               # из паспорта рецепта тренера (GATE = 0.65)
GATE_TRUE = 0.60          # ML5H_GATE_TRUE не задан -> дефолт кода
STOP_PCT = 0.15
H = 10
BAR_MS = 30 * 60_000
SLOTS = 10
FEE_BP = 12.0
START_BALANCE = 100.0     # депозит репетиции (масштабируется)
FIXED_NOTIONAL = 15.0     # фиксированный вариант (для сравнения)
MARGIN_RATE = 0.05        # маржа на позицию = 5% баланса (v3)
NOTIONAL_RATE = 0.15      # номинал = 15% баланса (v3)
DAY_CAP_RATE = 0.04       # дневной кап 4% текущего баланса
MIN_TRAIN_ROWS = 30_000   # месяц не торгуется на меньшем учебнике
META_ON = 1000            # мета включается с 1000 сигналов в тетради
MAP_ON = 500              # карта включается с 500 записей
WARM_MIN_BARS = 300       # warmup: монета с меньшей историей не торгуется

PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
META_PARAMS = dict(n_estimators=120, learning_rate=0.05, max_depth=3,
                   num_leaves=7, subsample=0.8, colsample_bytree=0.8,
                   random_state=42, n_jobs=4, verbosity=-1)


def find_input(name):
    for root, _dirs, fs in os.walk("/kaggle/input"):
        if name in fs or name in os.path.basename(root):
            return root
    return None


def find_zip(name):
    for root, _d, fs in os.walk("/kaggle/input"):
        if name in fs:
            return os.path.join(root, name)
    return None


# ---------- панель и признаки (v7, дословно ml5h/features.py) ----------
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

FUND = {}
for name in zf.namelist():
    if name.startswith("funding/") and name.endswith(".csv"):
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

SYMS0 = [s for s in sorted(n.replace("_30m.csv", "") for n in members)
         if s != "BTCUSDT"]      # бой: BTC не оценивается (features:47)
log(f"вселенная панели: {len(SYMS0)} монет (без BTC)")

coins = {}
ot_all, sid_all, gri_all, hi_all, lo_all, cl_all = [], [], [], [], [], []
atr_all, y_all = [], []
surv = 0
F_list = []
for sym in SYMS0:
    code = sym_id(sym)
    df = pd.read_csv(io.BytesIO(zf.read(f"{sym}_30m.csv")),
                     usecols=["open_time", "high", "low", "close",
                              "volume", "taker_buy_base"])
    df["open_time"] = df["open_time"].astype(np.int64)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    n = len(df)
    if n < WARM_MIN_BARS + 1:
        continue
    X, tr = make_features(df, btc_close, sym)
    Xg = g_cross(df, btc_close, sym)
    for c in Xg.columns:
        X[c] = Xg[c]
    X["sym_id"] = float(sym_id(sym))
    X = X.replace([np.inf, -np.inf], np.nan)
    # FEATS — боевой список тренера (24 признака, sym_id последним)
    FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
             "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
             "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
             "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
             "hour", "dow", "sym_id"]
    Fm = X[FEATS].to_numpy(dtype="float32")
    # метки тренера (ml5h_wide:105-109): вопрос бури ±0.5 ATR за 10 баров
    close_v = df["close"].to_numpy(dtype="float64")
    atr_v = (tr.rolling(14).mean() / df["close"]).to_numpy(dtype="float64")
    fut = np.full(n, np.nan, dtype="float64")
    fut[:-10] = close_v[10:]
    fwd = fut / close_v - 1
    thr = 0.5 * np.nan_to_num(atr_v, nan=np.nan)
    y_v = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                   np.where(fwd > thr, 1.0,
                            np.where(fwd < -thr, 0.0, np.nan)))
    keep = np.arange(n) >= WARM_MIN_BARS
    ot = df["open_time"].to_numpy(dtype="int64")[keep]
    c = dict(sym=sym, code=code, n=int(keep.sum()), ot=ot,
             hi=df["high"].to_numpy(dtype="float32")[keep],
             lo=df["low"].to_numpy(dtype="float32")[keep],
             cl=df["close"].to_numpy(dtype="float32")[keep],
             atr=atr_v[keep], y=y_v[keep])
    coins[sym] = c
    F_list.append(Fm[keep])
    ot_all.append(ot)
    sid_all.append(np.full(len(ot), surv, dtype=np.int32))
    surv += 1
    gri_all.append(np.arange(len(ot), dtype=np.int32))
    hi_all.append(c["hi"]); lo_all.append(c["lo"]); cl_all.append(c["cl"])
    atr_all.append(c["atr"].astype("float32"))
    y_all.append(c["y"].astype("float32"))
    if len(coins) % 100 == 0:
        log(f"признаки {len(coins)}/{len(SYMS0)}")
SYMS = list(coins.keys())

GOT = np.concatenate(ot_all)
order = np.argsort(GOT, kind="stable")
GOT = GOT[order]
GSID = np.concatenate(sid_all)[order]
GRI = np.concatenate(gri_all)[order]
GHI = np.concatenate(hi_all)[order]
GLO = np.concatenate(lo_all)[order]
GCL = np.concatenate(cl_all)[order]
GATR = np.concatenate(atr_all)[order].astype("float64")
GY = np.concatenate(y_all)[order].astype("float64")
GF = np.concatenate(F_list)[order]
del ot_all, sid_all, gri_all, hi_all, lo_all, cl_all, atr_all, y_all
uniq_ots, first_idx = np.unique(GOT, return_index=True)
GM = ((_pd_y := pd.DatetimeIndex(pd.to_datetime(GOT, unit="ms"))).year
      - 2020) * 12 + _pd_y.month - 1
GP = np.full(len(GOT), np.nan, dtype="float64")   # предсказания по месяцам
log(f"строки {len(GOT):,} | баров {len(uniq_ots):,}")

import random as _random
_chk_n = min(2000, len(GOT))
for gi in _random.Random(7).sample(range(len(GOT)), _chk_n):
    cn_s = coins[SYMS[int(GSID[gi])]]
    _ri = int(GRI[gi])
    assert int(cn_s["ot"][_ri]) == int(GOT[gi]), "стыковка разошлась"
    assert float(cn_s["cl"][_ri]) == float(GCL[gi]), "цены разошлись"
log(f"санити стыковки: {_chk_n}/{_chk_n}")

# ---------- месячный цикл: обучение -> торговля -> тетрадь -------------
positions, pending, trades = {}, {}, []
ledger = {}            # UTC-день -> деньги компаунда (для капа)
ledger_fixed = {}      # UTC-день -> деньги фиксированной сотни
balance = START_BALANCE
cap_latched_day = None
hist_meta, hist_map = [], []     # тетрадь: векторы/пункты прошлых месяцев
meta_on_month = map_on_month = None
funnel = dict(cand=0, conf_wait=0, meta_veto=0, quiet_cut=0, honest_skip=0,
              slot_skip=0, cap_block=0, orders=0, unfilled=0, trades=0)
prev_cand = {}                 # sid -> (ms бара, сторона) — цепь подтверждения
block_until = {}               # sid -> ms, до которого монета закрыта после выхода
SYM_IDX = FEATS.index("sym_id")


def day_of(ms):
    return int(ms) // 86_400_000


def day_str(dn):
    return pd.Timestamp(dn * 86_400_000, unit="ms", tz="UTC").strftime(
        "%Y-%m-%d")


def settle(tr):
    """Закрытие сделки: bp, фандинг, компаунд баланса, дневные регистры."""
    side = tr["side"]
    notional = tr["notional"]
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
            fevents.append((_t, -side * _r * notional))
    tr["gross_bp"] = gross_bp
    tr["fee_bp"] = FEE_BP
    tr["fund_bp"] = paid * 1e4
    tr["net_bp"] = gross_bp - FEE_BP - paid * 1e4
    tr["win"] = tr["net_bp"] > 0
    dt = pd.Timestamp(tr["entry_T"], unit="ms", tz="UTC")
    tr["year"] = dt.year
    tr["month_key"] = dt.strftime("%Y-%m")
    tr["fevents"] = fevents
    global balance
    balance += notional * (gross_bp - FEE_BP) / 1e4
    d_exit = day_of(tr["exit_T"])
    ledger[d_exit] = ledger.get(d_exit, 0.0) \
        + notional * (gross_bp - FEE_BP) / 1e4
    ledger_fixed[d_exit] = ledger_fixed.get(d_exit, 0.0) \
        + FIXED_NOTIONAL * tr["net_bp"] / 1e4
    for ft, famt in fevents:
        d_f = day_of(ft)
        ledger[d_f] = ledger.get(d_f, 0.0) + famt
    # фандинг фиксированного варианта: те же события по номиналу 15
    for _t, _r in (fpts[lo_b:hi_b] if fpts else []):
        d_f = day_of(_t)
        ledger_fixed[d_f] = ledger_fixed.get(d_f, 0.0) \
            - side * _r * FIXED_NOTIONAL


def scan_bars(bi_lo, bi_hi, iso, meta_m):
    """Скан диапазона баров — тело v7 с живым балансом и месячной цепочкой."""
    global cap_latched_day, balance
    for bi in range(bi_lo, bi_hi):
        T = int(uniq_ots[bi])
        lo_i = int(first_idx[bi])
        hi_i = int(first_idx[bi + 1]) if bi + 1 < len(uniq_ots) else len(GOT)

        # 1) развязка заявок прошлого бара
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
                stop_px = c0 * (1 - STOP_PCT) if od["side"] == 1 \
                    else c0 * (1 + STOP_PCT)
                hit = False
                if od["side"] == 1:
                    hit = float(cn["lo"][ri_next]) <= stop_px
                else:
                    hit = float(cn["hi"][ri_next]) >= stop_px
                if hit:
                    block_until[sid] = T + 10 * BAR_MS   # пауза 5ч после выхода
                    tr = dict(sym=sym, code=od["code"], side=od["side"],
                              entry_T=od["T"], exit_T=T, c0=od["c0"],
                              p=od["p"], notional=od["notional"],
                              gi=od["gi"],
                              exit_px=stop_px, kind="stop",
                              entry_ri=od["ri"], exit_ri=ri_next)
                    trades.append(tr)
                    settle(tr)
                    funnel["trades"] += 1
                else:
                    positions[sid] = dict(sym=sym, code=od["code"],
                                          side=od["side"], c0=od["c0"],
                                          p=od["p"], gi=od["gi"],
                                          notional=od["notional"],
                                          entry_T=od["T"], ri=ri_next,
                                          entry_ri=od["ri"],
                                          horizon_T=od["T"] + H * BAR_MS
                                          - 60_000,
                                          stop_px=stop_px)
            else:
                funnel["unfilled"] += 1
            pending.pop(sid, None)

        # 2) стоп/будильник (стоп раньше решения)
        for sid in list(positions.keys()):
            pos = positions[sid]
            sym = pos["sym"]
            cn = coins[sym]
            ri = pos["ri"] + 1
            if ri >= cn["n"] or int(cn["ot"][ri]) != T:
                continue
            pos["ri"] = ri
            hit = (pos["side"] == 1
                   and float(cn["lo"][ri]) <= pos["stop_px"]) or \
                  (pos["side"] == -1
                   and float(cn["hi"][ri]) >= pos["stop_px"])
            if hit:
                block_until[sid] = T + 10 * BAR_MS   # пауза 5ч после выхода
                tr = dict(sym=sym, code=pos["code"], side=pos["side"],
                          entry_T=pos["entry_T"], exit_T=T, c0=pos["c0"],
                          p=pos["p"], notional=pos["notional"],
                          gi=pos["gi"],
                          exit_px=pos["stop_px"], kind="stop",
                          entry_ri=pos["entry_ri"], exit_ri=ri)
                trades.append(tr)
                settle(tr)
                funnel["trades"] += 1
                positions.pop(sid, None)
                continue
            if T >= pos["horizon_T"]:
                block_until[sid] = T + 10 * BAR_MS   # пауза 5ч после выхода
                tr = dict(sym=sym, code=pos["code"], side=pos["side"],
                          entry_T=pos["entry_T"], exit_T=T, c0=pos["c0"],
                          p=pos["p"], notional=pos["notional"],
                          gi=pos["gi"],
                          exit_px=float(cn["cl"][ri]), kind="budilnik",
                          entry_ri=pos["entry_ri"], exit_ri=ri)
                trades.append(tr)
                settle(tr)
                funnel["trades"] += 1
                positions.pop(sid, None)

        # 3) скоринг бара (v7 + решётка 5ч)
        sids = GSID[lo_i:hi_i]
        atrs = GATR[lo_i:hi_i]
        ps = GP[lo_i:hi_i]
        gris = GRI[lo_i:hi_i]
        busy = set(positions.keys()) | set(pending.keys())
        elig = np.fromiter((int(s) not in busy for s in sids),
                           dtype=bool, count=len(sids))
        ea = atrs[elig]
        ea = ea[np.isfinite(ea)]
        quiet_on = len(ea) >= 30
        q33 = float(np.quantile(ea, 1 / 3)) if quiet_on else 0.0
        gate_mask = (ps > GATE) | (ps < 1 - GATE)
        # ПОДТВЕРЖДЕНИЕ (ТЕСТ-2): то же, что в ТЕСТ-1, плюс пауза 5ч
        # ПОСЛЕ ВЫХОДА: из блока монета невидима — сигналы не считаются
        elig = np.fromiter(
            (elig[i] and T >= block_until.get(int(s), 0)
             for i, s in enumerate(sids)),
            dtype=bool, count=len(sids))
        confirmed = np.fromiter(
            (prev_cand.get(int(s), (0, 0)) == (T - BAR_MS,
                                               1 if ps[i] > GATE else -1)
             for i, s in enumerate(sids)),
            dtype=bool, count=len(sids))
        cand_mask = elig & gate_mask & confirmed
        funnel["cand"] += int(cand_mask.sum())
        funnel["conf_wait"] += int((elig & gate_mask & ~confirmed).sum())
        # цепочка подтверждений: свободные монеты обновляются всегда
        # (прошли порог — запомнили сторону; не прошли — цепь порвалась)
        for i, s in enumerate(sids):
            if elig[i]:
                sid = int(s)
                if gate_mask[i]:
                    prev_cand[sid] = (T, 1 if ps[i] > GATE else -1)
                else:
                    prev_cand.pop(sid, None)
        if not cand_mask.any():
            continue
        cidx = np.flatnonzero(cand_mask)
        rows_meta = []
        for k in cidx:
            if np.isfinite(float(atrs[k])) and np.isfinite(float(ps[k])):
                rows_meta.append(k)
        if meta_m is not None and rows_meta:
            Fm_ = np.empty((len(rows_meta), len(FEATS) + 2), dtype="float64")
            for j, k in enumerate(rows_meta):
                Fm_[j, :len(FEATS)] = GF[int(lo_i + k)]
                Fm_[j, len(FEATS)] = ps[k]
                Fm_[j, len(FEATS) + 1] = atrs[k]
            Fm_ = np.where(np.isfinite(Fm_), Fm_, 0.0)
            pw = meta_m.predict(Fm_)
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
        cand.sort(key=lambda t: t[0], reverse=True)
        slots = len(positions) + len(pending)
        margin_one = balance * MARGIN_RATE
        avail = balance - margin_one * len(positions)
        slot_cap = SLOTS
        if margin_one > 0 and avail < margin_one:
            slot_cap = min(SLOTS, int(avail // margin_one))
            if slot_cap < slots:
                slot_cap = slots
        today = day_of(T)
        if cap_latched_day is not None and cap_latched_day != today:
            cap_latched_day = None
        day_pnl_now = ledger.get(today, 0.0)
        day_cap = balance * DAY_CAP_RATE
        for _conf, p_v, k in cand:
            if slots >= slot_cap:
                funnel["slot_skip"] += 1
                continue
            side = 1 if p_v > GATE else -1
            if iso is not None:
                t_true = float(np.interp(p_v, iso.X_thresholds_,
                                         iso.y_thresholds_))
                if side == 1 and t_true < GATE_TRUE:
                    funnel["honest_skip"] += 1
                    continue
                if side == -1 and 1 - t_true < GATE_TRUE:
                    funnel["honest_skip"] += 1
                    continue

            if cap_latched_day == today or day_pnl_now <= -day_cap:
                cap_latched_day = today
                funnel["cap_block"] += 1
                continue
            k_i = int(k)
            sym = SYMS[int(sids[k_i])]
            cn = coins[sym]
            ri_k = int(gris[k_i])
            pending[int(sids[k_i])] = dict(
                sym=sym, code=cn["code"], side=side,
                c0=float(cn["cl"][ri_k]), T=T, ri=ri_k, p=float(p_v),
                gi=lo_i + k_i,
                notional=max(balance * NOTIONAL_RATE, 5.5))
            slots += 1
            funnel["orders"] += 1


month_codes = sorted(set(int(m) for m in GM))
month_bars = {}
for bi in range(len(uniq_ots)):
    month_bars.setdefault(int(GM[first_idx[bi]]), []).append(bi)

month_rows_log = []
for M in month_codes:
    ted = np.flatnonzero(GM == M)
    mtr = np.flatnonzero((GM < M) & np.isfinite(GY))
    if len(ted) == 0:
        continue
    if M == 0 or len(mtr) < MIN_TRAIN_ROWS:
        continue
    m_start_ms = int(GOT[ted[0]])
    assert m_start_ms >= int(pd.Timestamp("2020-02-01", tz="UTC").value
                            // 10**6)
    # месяц M данными до M не видел (assert утечки)
    assert int(GOT[mtr].max()) < m_start_ms, "УТЕЧКА: учебник заходит в месяц"
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(GF[mtr], GY[mtr], categorical_feature=[SYM_IDX])
    GP[ted] = m.predict_proba(GF[ted])[:, 1]
    gate_pass = int(((GP[ted] > GATE) | (GP[ted] < 1 - GATE)).sum())
    log(f"[{M}] модель обучена на {len(mtr):,} строках; сигналов сырых "
        f"{gate_pass:,} (извещатель)")
    iso = None
    if len(hist_map) >= MAP_ON:
        ip = np.array([h[0] for h in hist_map])
        iy = np.array([1.0 if h[1] else 0.0 for h in hist_map])
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(ip, iy)
        if map_on_month is None:
            map_on_month = M
    meta_m = None
    if len(hist_meta) >= META_ON:
        hx = np.array([h[0] for h in hist_meta])
        hy = np.array([1.0 if h[1] else 0.0 for h in hist_meta])
        meta_m = lgb.LGBMClassifier(**META_PARAMS)
        meta_m.fit(hx, hy)
        if meta_on_month is None:
            meta_on_month = M
    t_m = len(trades)
    bars_m = month_bars.get(M, [])
    if bars_m:
        scan_bars(min(bars_m), max(bars_m) + 1, iso, meta_m)
    m_trades = trades[t_m:]
    wins = sum(1 for t in m_trades if t["win"])
    avg_m = float(np.mean([t["net_bp"] for t in m_trades])) \
        if m_trades else 0.0
    # тетрадь меты: векторы сделок ЭТОГО месяца
    for tr in m_trades:
        gi = tr.get("gi")
        if gi is not None:
            vec = np.concatenate([GF[gi].astype("float64"),
                                  [tr["p"], float(GATR[gi])]])
            hist_meta.append((vec, tr["win"]))
    # тетрадь карты: (p, ярлык направления) ВСЕХ строк ПРОШЕДШЕГО месяца
    # с известным ярлыком — как oos_p/oos_y боевого тренера; попадает в
    # тетрадь только ПОСЛЕ торговли месяца (иначе карта месяца знала бы
    # его же ответы — заглядывание; карта месяца строится на месяцах до него)
    yk = np.isfinite(GY[ted])
    for _p, _y in zip(GP[ted][yk].tolist(), GY[ted][yk].tolist()):
        hist_map.append((float(_p), bool(_y >= 0.5)))
    month_rows_log.append(dict(
        month=pd.Timestamp(m_start_ms, unit="ms", tz="UTC").strftime("%Y-%m"),
        train_rows=int(len(mtr)), trades=len(m_trades),
        win=round(wins / max(1, len(m_trades)) * 100, 1),
        avg_bp=round(avg_m, 1), balance=round(balance, 2),
        meta=meta_m is not None, map=iso is not None))
    del m
    log(f"[{M}] сделок {len(m_trades)} | win {wins}/{len(m_trades)} | "
        f"avg {avg_m:+.1f}бп | баланс {balance:.2f}")

# ---------- хвосты (делист) --------------------------------------------
n_tails = 0
for sid, pos in list(positions.items()):
    cn = coins[pos["sym"]]
    last_ri = cn["n"] - 1
    raw_px = float(cn["cl"][last_ri])
    exit_px = max(raw_px, pos["stop_px"]) if pos["side"] == 1 \
        else min(raw_px, pos["stop_px"])
    tr = dict(sym=pos["sym"], code=pos["code"], side=pos["side"],
              entry_T=pos["entry_T"], exit_T=int(cn["ot"][last_ri]),
              c0=pos["c0"], p=pos["p"], notional=pos["notional"],
              exit_px=exit_px, kind="delist",
              entry_ri=pos["entry_ri"], exit_ri=last_ri)
    trades.append(tr)
    settle(tr)
    funnel["trades"] += 1
    n_tails += 1
log(f"хвостов закрыто: {n_tails}")

# ---------- отчёт ------------------------------------------------------
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
    by_year.setdefault(pd.Timestamp(t["entry_T"], unit="ms",
                                    tz="UTC").year, []).append(t)

# компаунд-кривая по дням
bal_days = sorted(ledger.keys())
bal_cum, b = [], START_BALANCE
for d in bal_days:
    b += ledger[d]
    bal_cum.append(b)
cum = np.array(bal_cum)
peak = np.maximum.accumulate(cum)
dd = (cum - peak) / peak
maxdd_pct = float(dd.min()) * 100 if len(dd) else 0.0
red_days = sum(1 for v in ledger.values() if v < 0)
streak = worst = 0
for d in bal_days:
    if ledger[d] < 0:
        streak += 1
        worst = max(worst, streak)
    else:
        streak = 0
months_bal = {}
for d, v in ledger.items():
    months_bal.setdefault(day_str(d)[:7], 0.0)
    months_bal[day_str(d)[:7]] += v
pos_months = sum(1 for v in months_bal.values() if v > 0)

fix_cum = np.cumsum([ledger_fixed.get(d, 0.0) for d in bal_days])
peak_f = np.maximum.accumulate(np.concatenate([[0.0], fix_cum]))[1:]
dd_f = fix_cum - peak_f

report = dict(
    pre_reg="репетиция боя: месячные модели (рецепт ml5h_wide), мета с "
            "1000 сигналов, карта с 500 записей, вся цепочка v7, "
            "компаундинг B=100 + фикс-сотня рядом",
    funnel=funnel, meta_on=None, map_on=None,
    totals=dict(trades=n, win=round(wins / max(1, n) * 100, 1),
                avg_bp=round(net_sum / max(1, n), 1),
                median_bp=round(float(np.median([t["net_bp"] for t in trades]))
                                if n else 0.0, 1),
                sum_bp=round(net_sum, 0), stops=stopped,
                stop_pct=round(stopped / max(1, n) * 100, 1),
                funding_bp=round(fund_sum, 0), kinds=kinds),
    by_year={str(y): agg(rows) for y, rows in sorted(by_year.items())},
    money=dict(
        start=START_BALANCE,
        final=round(balance, 2),
        growth_x=round(balance / START_BALANCE, 2),
        days=len(bal_days),
        red_days=red_days,
        red_pct=round(red_days / max(1, len(bal_days)) * 100, 1),
        worst_streak=worst,
        maxdd_pct=round(maxdd_pct, 1),
        best_day=round(max(ledger.values()) if ledger else 0.0, 2),
        worst_day=round(min(ledger.values()) if ledger else 0.0, 2),
        months=len(months_bal), months_pos=pos_months,
        cap_blocks=funnel["cap_block"]),
    money_fixed=dict(
        final=round(float(fix_cum[-1]) if len(fix_cum) else 0.0, 2),
        maxdd=round(float(dd_f.min()) if len(dd_f) else 0.0, 2),
        red_days=sum(1 for v in ledger_fixed.values() if v < 0)),
    monthly=month_rows_log,
    months_money={k: round(v, 2) for k, v in sorted(months_bal.items())},
)
report["meta_on"] = None
for mr in month_rows_log:
    if mr["meta"] and report["meta_on"] is None:
        report["meta_on"] = mr["month"]
    if mr["map"] and report["map_on"] is None:
        report["map_on"] = mr["month"]

with open("/kaggle/working/battle_replay_report.json", "w") as f:
    json.dump(report, f, indent=1, default=float)
with open("/kaggle/working/battle_replay_trades.jsonl", "w") as f:
    for t in trades:
        f.write(json.dumps(dict(
            sym=t["sym"], side="LONG" if t["side"] == 1 else "SHORT",
            entry_ms=int(t["entry_T"]), exit_ms=int(t["exit_T"]),
            entry_px=float(t["c0"]), exit_px=float(t["exit_px"]),
            kind=t["kind"], p=float(t["p"]),
            notional=round(float(t["notional"]), 4),
            gross_bp=float(t["gross_bp"]), fee_bp=float(t["fee_bp"]),
            fund_bp=float(t["fund_bp"]), net_bp=float(t["net_bp"]),
            win=bool(t["win"]), year=int(t["year"]),
            month=t["month_key"],
        ), default=float) + "\n")

print("\n=== ВОРОНКА ===")
print(f"кандидатов {funnel['cand']:,} | -мета {funnel['meta_veto']:,} | "
      f"-тихая {funnel['quiet_cut']:,} | -честный {funnel['honest_skip']:,} | "
      f"-слоты {funnel['slot_skip']:,} | -кап {funnel['cap_block']:,} | "
      f"заявок {funnel['orders']:,} | не взялись {funnel['unfilled']:,} | "
      f"сделок {n:,}")
print(f"мета включилась: {report['meta_on']} | карта включилась: "
      f"{report['map_on']}")
print("\n=== ИТОГИ ===")
tt = report["totals"]
print(f"сделок {tt['trades']} | win {tt['win']}% | avg {tt['avg_bp']}бп | "
      f"медиана {tt['median_bp']}бп | сумма {tt['sum_bp']:.0f}бп | "
      f"стопов {tt['stops']} ({tt['stop_pct']}%) | фандинг "
      f"{tt['funding_bp']:.0f}бп")
print("\n=== ПО ГОДАМ ===")
for y, a in report["by_year"].items():
    print(f"{y} | {a['n']} | {a['win']}% | {a['avg']}бп | "
          f"{a['sum_bp']:.0f} | стопов {a['stops']}")
mm = report["money"]
print("\n=== ДЕНЬГИ (компаунд, старт 100) ===")
print(f"итог {mm['final']:.2f} (x{mm['growth_x']}) за {mm['days']} дней | "
      f"красных {mm['red_days']} ({mm['red_pct']}%) | худшая серия "
      f"{mm['worst_streak']} | maxDD {mm['maxdd_pct']}% | худший день "
      f"{mm['worst_day']:+.2f}")
print(f"месяцев {mm['months']}, положительных {mm['months_pos']} "
      f"({round(mm['months_pos'] / max(1, mm['months']) * 100, 1)}%)")
mf_ = report["money_fixed"]
print(f"фикс-сотня: {mf_['final']:+.2f} | maxDD {mf_['maxdd']:.2f} | "
      f"красных {mf_['red_days']}")
log("РЕПЕТИЦИЯ БОЯ ГОТОВА")
