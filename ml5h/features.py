"""Признаки ML-5Ч: 1:1 с тренировкой (широкая модель на Кегле:
база + G-cross + sym_id). ПАРИТЕТ ЖИВОГО И БЭКТЕСТНОГО РАСЧЁТА
КРИТИЧЕН — функции копируются, не переписываются."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


def sym_id(symbol: str) -> int:
    """Метка монеты — та же формула, что в обучении (кегл-кернел)."""
    return zlib.crc32(symbol.encode()) & 0xFFFF


def make_features(df: pd.DataFrame, btc_close, symbol: str) -> pd.DataFrame:
    """База: признаки из run_ml_honest.py (30м бары)."""
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
    out["clv"] = ((close - low) / rng - 0.5).rolling(20, min_periods=10).mean()
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
        out["btc_corr"] = close.rolling(50, min_periods=20).corr(btc)
    # 24ч импульс прямо на 30м барах: 48 баров x 30мин = 24ч — тот же
    # смысл, что 6 x 4h, но БЕЗ утечки будущего (старый resample-вариант
    # давал бару 08:00 цену 11:30 — корзина накрывала 4 часа вперёд).
    out["h4_momentum"] = close.pct_change(48)
    ts = pd.to_datetime(df["open_time"], unit="ms")
    out["hour"] = ts.dt.hour.values
    out["dow"] = ts.dt.dayofweek.values
    return out


def g_cross(df: pd.DataFrame, btc_close, symbol: str) -> pd.DataFrame:
    """Группа G-cross (удержана шагом 2): контекст соседей."""
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


def feature_row(df: pd.DataFrame, btc_close, symbol: str,
                feats: list[str]) -> dict | None:
    """Последняя ЗАКРЫТАЯ строка признаков (NaN -> None -> пропуск)."""
    X = make_features(df, btc_close, symbol).replace([np.inf, -np.inf], np.nan)
    Xg = g_cross(df, btc_close, symbol)
    for c in Xg.columns:
        X[c] = Xg[c]
    # БАРЫ В df ТОЛЬКО ЗАКРЫТЫЕ (прогрев и поток отбрасывают формирующуюся
    # свечу) — свежая закрытая = iloc[-1]. Исторический урок 29.09: iloc[-2]
    # оставшийся от старой конструкции давал модели данные на бар старее.
    last = X.iloc[-1]
    row = {f: float(last[f]) for f in feats if f in X.columns}
    if "sym_id" in feats:
        row["sym_id"] = float(sym_id(symbol))
    missing = [f for f in feats if f not in row]
    if missing:
        return None
    return row
