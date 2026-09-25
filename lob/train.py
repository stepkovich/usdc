"""Тренеровщик стаканной модели + честная оценка (walk-forward по дням).

Таргет: mid через 5 минут выше текущего? (двоичный)
Фичи: строка рекордера (дисбалансы 5/10/20 уровней, микро-цена, спред,
поток 10с/60с, vpin) + изменения миды за 30с и 120с.
Модель: LightGBM (данных пока мало; нейросеть — когда накопится месяц+).

Честность:
- walk-forward: день k предсказываем моделью, обученной на днях < k;
- дрейф-контроль: сравниваем «купить в топ-квартиле уверенности» против
  «купить в BOTTOM-квартиле» — разница (spread) не зависит от дрейфа
  рынка и показывает чистую силу сигнала;
- издержки двумя сценариями: мейкер круг ~4 бп, тейкер круг ~19 бп.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

import glob
import sqlite3

LOB_DB = Path(__file__).resolve().parent.parent / "data" / "lob" / "lob.db"
PARQUET_DIR = Path(__file__).resolve().parent.parent / "data" / "lob" / "archive"
BOOKSHELF = Path("/home/iek/PycharmProjects/zcodetrading/data/bookshelf.db")
LAG_TOL_MS = 2500
HORIZON_S = 300
SEED = 42


def load_new() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(PARQUET_DIR / "feat_*.parquet"))):
        frames.append(pd.read_parquet(f))
    c = sqlite3.connect(LOB_DB)
    frames.append(pd.read_sql("select * from feat", c))
    c.close()
    df = pd.concat(frames, ignore_index=True) \
        .drop_duplicates(["ts", "symbol"]) \
        .sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df


def load_bookshelf(days: int = 10) -> pd.DataFrame:
    import sqlite3
    c = sqlite3.connect(BOOKSHELF)
    df = pd.read_sql(
        "select ts, symbol, best_bid, best_ask, bid_sum, ask_sum from book",
        c)
    c.close()
    df["mid"] = (df["best_bid"] + df["best_ask"]) / 2
    tot = df["bid_sum"] + df["ask_sum"]
    df["imb10"] = (df["bid_sum"] - df["ask_sum"]) / tot.replace(0, np.nan)
    df["spread_bp"] = (df["best_ask"] - df["best_bid"]) / df["mid"] * 10000
    df["source"] = "bookshelf"
    return df


def build_dataset(df: pd.DataFrame, rich: bool) -> tuple[pd.DataFrame, list]:
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    # будущая мида через ~HORIZON_S: ближайшая строка того же символа
    parts = []
    for sym, g in df.groupby("symbol"):
        g = g.copy()
        idx = g["ts"].searchsorted(g["ts"] + HORIZON_S * 1000)
        idx = np.clip(idx, 0, len(g) - 1)
        g["mid_fut"] = g["mid"].values[idx]
        ok = g["ts"].values[idx] >= g["ts"].values + HORIZON_S * 1000 - 1500
        g.loc[~ok, "mid_fut"] = np.nan
        parts.append(g)
    df = pd.concat(parts)
    df["y"] = (df["mid_fut"] / df["mid"] > 1).astype(float)
    df.loc[df["mid_fut"].isna(), "y"] = np.nan
    # лаги по ВРЕМЕНИ (строки 500мс, но ищем ближайший бар к t-lag)
    for lag_ms, name in ((30_000, "d30"), (120_000, "d120")):
        parts_l = []
        for sym, g in df.groupby("symbol"):
            g = g.copy()
            ts = g["ts"].values
            midv = g["mid"].values
            idx = ts.searchsorted(ts - lag_ms)
            idx = np.clip(idx, 0, len(g) - 1)
            ok = np.abs(ts[idx] - (ts - lag_ms)) <= LAG_TOL_MS
            past = np.where(ok, midv[idx], np.nan)
            g[name] = midv / past - 1
            parts_l.append(g)
        df = pd.concat(parts_l)
    if rich:
        feats = ["mid", "spread_bp", "microprice", "imb5", "imb10", "imb20",
                 "flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
                 "ntr10", "vpin10", "d30", "d120"]
    else:
        feats = ["mid", "spread_bp", "imb10", "d30", "d120"]
    df = df.dropna(subset=feats + ["y"])
    # нормировка относительных фич на символ не нужна: imb/vpin/спред
    # безразмерны, потоки делим на их среднее по символу
    for c in ("flow10_buy", "flow10_sell", "flow60_buy", "flow60_sell",
              "ntr10"):
        if c in df.columns:
            df[c] = df[c] / df.groupby("symbol")[c] \
                .transform("median").replace(0, np.nan)
    return df, feats


def evaluate(df: pd.DataFrame, feats: list, n_test_days: int = 5) -> None:
    df["day"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    days = sorted(df["day"].unique())
    test_days = days[-n_test_days:]
    print(f"дней всего {len(days)}, тестовых {len(test_days)}: "
          f"{test_days[0]}..{test_days[-1]}, строк {len(df)}")
    rows = []
    for k, day in enumerate(test_days):
        tr = df[df["day"] < day]
        te = df[df["day"] == day]
        if len(te) < 500 or len(tr) < 5000:
            continue
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05,
                               max_depth=4, subsample=0.8, colsample_bytree=0.8,
                               random_state=SEED, n_jobs=-1, verbosity=-1)
        m.fit(tr[feats], tr["y"])
        p = m.predict_proba(te[feats])[:, 1]
        te = te.assign(p=p)
        fwd = te["mid_fut"] / te["mid"] - 1
        q75, q25 = np.quantile(p, 0.75), np.quantile(p, 0.25)
        top = fwd[p >= q75].mean() * 10000
        bot = fwd[p <= q25].mean() * 10000
        acc = ((p > 0.5) == (te["y"] == 1)).mean() * 100
        rows.append({"day": day, "acc%": acc, "top_bp": top, "bot_bp": bot,
                     "spread_bp": top - bot})
        print(f"  {day}: acc {acc:.1f}% | топ-кварт {top:+.1f} бп | "
              f"боттом {bot:+.1f} бп | ЧИСТЫЙ СИГНАЛ {top-bot:+.1f} бп/5мин")
    if not rows:
        print("недостаточно данных для оценки")
        return
    r = pd.DataFrame(rows)
    print(f"\nСРЕДНЕЕ: чистый сигнал {r['spread_bp'].mean():+.1f} бп за 5 мин; "
          f"издержки: мейкер круг ~4 бп, тейкер ~19 бп")
    print("вывод: сигнал живой, если чистый спред устойчиво > издержек "
          "в БОЛЬШИНСТВЕ тестовых дней")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "bookshelf"
    save = "--save" in sys.argv
    if src == "bookshelf":
        df = load_bookshelf()
        rich = False
        print("источник: старый bookshelf (10с, топ-10, 20 символов) — "
              "проверка осуществимости")
    else:
        df = load_new()
        rich = True
        print("источник: новый рекордер (2с, 20 уровней + поток)")
    ds, feats = build_dataset(df, rich)
    evaluate(ds, feats)
    if save:
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05,
                               max_depth=4, subsample=0.8, colsample_bytree=0.8,
                               random_state=SEED, n_jobs=-1, verbosity=-1)
        m.fit(ds[feats], ds["y"])
        out = Path(__file__).resolve().parent.parent / "models" / "lob.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        m.booster_.save_model(str(out))
        print(f"модель сохранена -> {out}")
