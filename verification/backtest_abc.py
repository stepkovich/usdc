"""САМЫЙ ЧЕСТНЫЙ БЭКТЕСТ пакетов A+B+C (26.09).

Данные: edge_lab/data_cache/ml_klines_15m/*.csv — 10 ликвидных пар,
15-минутки 2020-2026 (USDT-M как прокси USDC-M — динамика та же).

Зеркало живого бота (демо-VPS конфиг, все пороги ЗАМОРОЖЕНЫ, ничего
не оптимизируется):
  - Дончиан 48 TF-баров (12ч), shift(1); сигнал закрытием за каналом;
  - вход: лимитка на уровне, ждать 4 TF-бара (60 мин), отмена при
    close за уровнем на 0.5*ATR; fill только при ПРОТОРГОВКЕ уровня
    (touch-through 1 тик = 1 бп), гэп сквозь уровень -> исполнение по
    open (без гэп-кредита);
  - стоп 12 x ATR_frac(8ч=32 TF-бара), TP +0.5% (исполняется по цели);
  - same-bar: строгая конвенция (бар входа выхода не даёт);
    внутри бара при достижимости и стопа и TP считаем СТОП (худший);
  - издержки USDC-M промо: вход мейкер 0%, TP мейкер 0%, стоп тейкер
    0.045% + слиппедж 0.05%;
  - гейт BTC(30д): лонг >+3%, шорт <-3%, болото — не торгуем;
  - фильтры C: ADX(14)@15м < 20 -> skip; фандинг-окно 5 мин до границ
    00/08/16 UTC -> skip; VPVR(48 баров, 48 бинов) -> HVN-стена skip.

Отчёт в R-мультипликаторах (риск на сделку = 0.15% баланса) и
компаундом на 4858 USDC (баланс демо). Порог подозрения владельца:
всё, что >+200% за 20 месяцев — ошибка, пока не доказано обратное.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/home/iek/PycharmProjects/edge_lab/data_cache/ml_klines_15m")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "FILUSDT",
           "NEARUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT", "ZECUSDT"]

# замороженные параметры живого бота
DONCH = 48          # 12ч в 15м барах
ATR_N = 32          # 8ч в 15м барах
STOP_MULT = 12.0
TARGET = 0.005
WAIT_BARS = 4       # 60 мин
CANCEL_RATIO = 0.5
RISK = 0.0015
TAKER_FEE = 0.00045
STOP_SLIP = 0.0005
THR_BP = 1.0        # touch-through: проторговка 1 бп
ADX_N = 14
ADX_THR = 20.0
FUNDING_MIN = 5
VPVR_BINS = 48

BALANCE_DEMO = 4858.0


def load(sym: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{sym}_15m.csv",
                     usecols=["open_time", "open", "high", "low", "close",
                              "volume"])
    df["ot"] = df["open_time"].astype(np.int64)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df.sort_values("ot").reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr_frac"] = tr.rolling(ATR_N).mean() / c
    df["d_high"] = h.rolling(DONCH).max().shift(1)
    df["d_low"] = l.rolling(DONCH).min().shift(1)
    # ADX(14) Уайлдера: ewm(alpha=1/n) == сглаживание Уайлдера
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_s = tr.ewm(alpha=1 / ADX_N, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index)\
        .ewm(alpha=1 / ADX_N, adjust=False).mean() / atr_s
    mdi = 100 * pd.Series(minus_dm, index=df.index)\
        .ewm(alpha=1 / ADX_N, adjust=False).mean() / atr_s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / ADX_N, adjust=False).mean()
    return df


def regime_map() -> pd.Series:
    """BTC 30д доходность по дням (из 15м BTC)."""
    btc = load("BTCUSDT")
    btc["date"] = pd.to_datetime(btc["ot"], unit="ms", utc=True).dt.date
    daily = btc.groupby("date")["close"].last()
    ret30 = daily / daily.shift(30) - 1
    return ret30


def funding_skip(ot: np.int64) -> bool:
    """Бар закрывается в окне <=5 мин до границы фандинга 00/08/16 UTC."""
    close_s = (ot + 900_000) / 1000
    sod = close_s % 86400
    return min(abs(sod - 0), abs(sod - 28800), abs(sod - 57600),
               86400 - sod) <= FUNDING_MIN * 60


def vpvr_zone(bars: np.ndarray, price: float) -> str:
    """bars: последние DONCH баров [high, low, volume]; профиль по бинам."""
    hi, lo = bars[:, 0].max(), bars[:, 1].min()
    if hi <= lo:
        return "mid"
    width = (hi - lo) / VPVR_BINS
    hist = np.zeros(VPVR_BINS)
    for bh, bl, bv in bars:
        if bv <= 0:
            continue
        i0 = max(0, int((bl - lo) / width))
        i1 = min(VPVR_BINS - 1, int((bh - lo) / width))
        hist[i0:i1 + 1] += bv / (i1 - i0 + 1)
    nz = hist[hist > 0]
    if len(nz) == 0:
        return "mid"
    med = float(np.median(nz))
    idx = min(VPVR_BINS - 1, max(0, int((price - lo) / width)))
    if hist[idx] >= 2 * med:
        return "hvn"
    if hist[idx] <= 0.3 * med:
        return "lvn"
    return "mid"


def simulate(sym: str, df: pd.DataFrame, reg: pd.Series,
             use_adx: bool, use_vpvr: bool, use_funding: bool,
             use_gate: bool = True) -> dict:
    """Пошаговая симуляция одной пары. Возвращает сделки (список dict)."""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    ot = df["ot"].values
    atrf = df["atr_frac"].values
    dh = df["d_high"].values; dl = df["d_low"].values
    adx = df["adx"].values
    vol = df["volume"].values
    date_of = pd.to_datetime(ot, unit="ms", utc=True).date
    reg_arr = reg.reindex(date_of).values

    trades = []
    n = len(df)
    warm = max(DONCH + 2, ATR_N + 2, 4 * ADX_N)
    i = warm
    stats = {"signals": 0, "adx_skip": 0, "funding_skip": 0,
             "vpvr_skip": 0, "gate_skip": 0, "no_fill": 0}
    pending = None          # dict(side, level, atr, deadline, i0)
    pos = None              # dict(side, entry, stop, tp, atr, i_in)
    cooldown = 0

    def close_trade(pos, exit_px, kind, i):
        entry = pos["entry"]
        dist = pos["dist"]
        if pos["side"] == 1:
            pnl = exit_px - entry
        else:
            pnl = entry - exit_px
        cost = 0.0
        if kind == "stop":
            cost = (TAKER_FEE + STOP_SLIP) * exit_px
        r = (pnl - cost) / dist
        trades.append({"sym": sym, "i": i, "ot": ot[i], "side": pos["side"],
                       "entry": entry, "exit": exit_px, "kind": kind,
                       "r": r})

    while i < n:
        # 1) позиция: выходы (бар входа выхода не даёт — strict)
        if pos is not None:
            if i > pos["i_in"]:
                if pos["side"] == 1:
                    stop_hit = l[i] <= pos["stop"]
                    tp_hit = h[i] >= pos["tp"]
                else:
                    stop_hit = h[i] >= pos["stop"]
                    tp_hit = l[i] <= pos["tp"]
                if stop_hit:                      # стоп-приоритет (худший)
                    gap = min(o[i], pos["stop"]) if pos["side"] == 1 \
                        else max(o[i], pos["stop"])
                    close_trade(pos, gap, "stop", i)
                    pos = None
                    cooldown = i + WAIT_BARS * 3   # пауза после выхода
                elif tp_hit:
                    close_trade(pos, pos["tp"], "tp", i)   # TP по цели
                    pos = None
                    cooldown = i + WAIT_BARS * 3
            if pos is None:
                i += 1
                continue
            i += 1
            continue

        # 2) pending: ждать отката к уровню
        if pending is not None:
            p = pending
            side, level = p["side"], p["level"]
            thr = THR_BP / 10000 * level
            if side == 1:
                if o[i] < level:                   # гэп сквозь уровень вниз
                    fill = o[i]
                elif l[i] <= level - thr:
                    fill = level
                else:
                    fill = None
                cancel = c[i] < level * (1 - CANCEL_RATIO * p["atr"])
            else:
                if o[i] > level:
                    fill = o[i]
                elif h[i] >= level + thr:
                    fill = level
                else:
                    fill = None
                cancel = c[i] > level * (1 + CANCEL_RATIO * p["atr"])
            if fill is not None:
                dist = STOP_MULT * p["atr"] * fill
                pos = {"side": side, "entry": fill, "dist": dist,
                       "stop": fill - dist if side == 1 else fill + dist,
                       "tp": fill * (1 + TARGET) if side == 1
                       else fill * (1 - TARGET),
                       "i_in": i}
                pending = None
            elif cancel or i > p["deadline"]:
                stats["no_fill"] += 1
                pending = None
            i += 1
            continue

        # 3) новый сигнал
        if i < cooldown or np.isnan(atrf[i]) or np.isnan(dh[i]) \
                or atrf[i] <= 0:
            i += 1
            continue
        side = 0
        if c[i] > dh[i]:
            side = 1
        elif c[i] < dl[i]:
            side = -1
        if side == 0:
            i += 1
            continue
        stats["signals"] += 1
        if use_gate:
            r30 = reg_arr[i]
            if np.isnan(r30) or (side == 1 and r30 <= 0.03) or \
                    (side == -1 and r30 >= -0.03):
                stats["gate_skip"] += 1
                i += 1
                continue
        if use_adx and (np.isnan(adx[i]) or adx[i] < ADX_THR):
            stats["adx_skip"] += 1
            i += 1
            continue
        if use_funding and funding_skip(ot[i]):
            stats["funding_skip"] += 1
            i += 1
            continue
        level = dh[i] if side == 1 else dl[i]
        if use_vpvr:
            window = np.column_stack([h[i - DONCH + 1:i + 1],
                                      l[i - DONCH + 1:i + 1],
                                      vol[i - DONCH + 1:i + 1]])
            if vpvr_zone(window, float(c[i])) == "hvn":
                stats["vpvr_skip"] += 1
                i += 1
                continue
        pending = {"side": side, "level": level, "atr": atrf[i],
                   "deadline": i + WAIT_BARS, "i0": i}
        i += 1

    return {"trades": trades, "stats": stats}


def run(config_name: str, use_adx: bool, use_vpvr: bool, use_funding: bool,
        use_gate: bool = True) -> dict:
    reg = regime_map() if use_gate else pd.Series(dtype=float)
    all_trades = []
    agg_stats = {}
    for sym in SYMBOLS:
        df = add_indicators(load(sym))
        if not use_gate:
            reg_empty = pd.Series(np.nan, index=reg.index)
            res = simulate(sym, df, reg_empty, use_adx, use_vpvr,
                           use_funding, use_gate=False)
        else:
            res = simulate(sym, df, reg, use_adx, use_vpvr,
                           use_funding, use_gate=True)
        all_trades.extend(res["trades"])
        for k, v in res["stats"].items():
            agg_stats[k] = agg_stats.get(k, 0) + v
    tdf = pd.DataFrame(all_trades).sort_values("ot").reset_index(drop=True)
    tdf["dt"] = pd.to_datetime(tdf["ot"], unit="ms", utc=True)
    tdf["year"] = tdf["dt"].dt.year

    rs = tdf["r"].values
    n = len(rs)
    wr = float((rs > 0).mean()) * 100 if n else float("nan")
    avg_r = float(rs.mean()) if n else float("nan")
    sum_r = float(rs.sum()) if n else float("nan")
    eq = np.cumsum(rs)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if n else 0.0
    # компаунд на демо-баланс (риск 0.15% баланса на сделку)
    bal = BALANCE_DEMO
    for r in rs:
        bal *= (1 + RISK * r)
    growth = (bal / BALANCE_DEMO - 1) * 100
    # по годам
    by_year = tdf.groupby("year")["r"].agg(["count", "sum", "mean"])
    print(f"\n=== {config_name} ===")
    print(f"сделок {n}, WR {wr:.1f}%, ср.сделка {avg_r:+.4f}R, "
          f"сумма {sum_r:+.1f}R, макс.DD {dd:.1f}R")
    print(f"компаунд на {BALANCE_DEMO:.0f}: {bal:.0f} ({growth:+.1f}%)")
    print("фильтры:", {k: v for k, v in agg_stats.items()})
    print("по годам:")
    print(by_year.round(3).to_string())
    return {"name": config_name, "n": n, "wr": wr, "avg_r": avg_r,
            "sum_r": sum_r, "dd_r": dd, "final": bal, "growth": growth,
            "stats": agg_stats, "trades": tdf}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = []
    if which in ("all", "nogate"):
        results.append(run("без гейта (сырой пробойник)", False, False, False,
                           use_gate=False))
    if which in ("all", "base"):
        results.append(run("база: 15м + гейт BTC (= демо-VPS сейчас)",
                           False, False, False))
    if which in ("all", "adx"):
        results.append(run("база + ADX(14)<20 skip", True, False, False))
    if which in ("all", "vpvr"):
        results.append(run("база + VPVR (HVN-стена skip)", False, True, False))
    if which in ("all", "adxpvr"):
        results.append(run("база + ADX + VPVR", True, True, False))
    if which in ("all", "full"):
        results.append(run("база + ADX + VPVR + фандинг-окно (полный C)",
                           True, True, True))
    print("\n================= СВОДКА =================")
    print(f"{'конфиг':38s} {'сделок':>7s} {'WR%':>6s} {'ср.R':>9s} "
          f"{'сумма R':>9s} {'DD R':>7s} {'компаунд':>10s}")
    for r in results:
        print(f"{r['name']:38s} {r['n']:7d} {r['wr']:6.1f} "
              f"{r['avg_r']:+9.4f} {r['sum_r']:+9.1f} {r['dd_r']:7.1f} "
              f"{r['growth']:+9.1f}%")
