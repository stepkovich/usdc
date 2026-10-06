"""USDC-МАЙКЕР-ЭКЗАМЕН (предрегистрация 05.10, до запуска) — идея владельца:
перевести аналитика на USDC-пары с мейкер-комиссией ~0.

ГИПОТЕЗА: свечной сигнал 1ч/30м умирал от ИЗДЕРЖЕК (20бп круг на USDT).
Если вход мейкером 0% и выход мейкером 0% (промо USDC-M) — круг ~0бп,
и даже тонкий навык может остаться деньгами.

ДАННЫЕ: 38 USDC-перпетуалов платформы USDT-M, 30м свечи из архива биржи
(месячные + дневные за последние дни). История молодых пар короче —
прогон адаптивный: тест = все кварталы после первых 4 обучающих.

ПРОТОКОЛ (записан до прогона):
- те же 24 признака, те же параметры модели, лонг p>gate / шорт p<1-gate,
  cooldown 10 баров, walk-forward по кварталам (тест никогда не в обучении);
- вход: лимит по закрытию сигнального бара, исполняется если следующий
  бар коснулся цены (как живой GTX);
- СЦЕНАРИИ ИЗДЕРЖЕК (круг, бп):
    S20  тейкер-мир USDT (5+5 slip вход, 5+5 выход)  — база для сравнения
    S10  мейкер-вход 0 + тейкер-выход 5 (+5 slip)
    S00  мейкер-вход 0 + мейкер-выход 0, выход лимиткой: заполняется,
         если за следующий бар цена коснулась; иначе тейкер через бар
- ПОРОГИ: 0.55 / 0.60 / 0.65 / 0.70 на общих предсказаниях.
ВЕРДИКТ (по S00 — главная гипотеза): жив, если avg >= +2бп И
плюс-кварталов >= 60% И нуль-перцентиль >= 95 на двух последних фолдах.
Ничего не деплоится.
"""
import io, json, time, zipfile
import urllib.request
import numpy as np, pandas as pd
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor

t0 = time.time()
H = 10
COOLDOWN_MS = 10 * 3600_000
SLIP = 0.0005
GATES = [0.55, 0.60, 0.65, 0.70]
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")
BASE = "https://data.binance.vision/data/futures/um"

# --- вселенная: живые USDC-перпетуалы (вшито: fapi с Кегла 451) ---
SYMS = ["1000BONKUSDC", "1000PEPEUSDC", "1000SHIBUSDC", "AAVEUSDC",
        "ADAUSDC", "ARBUSDC", "AVAXUSDC", "BCHUSDC", "BIOUSDC",
        "BNBUSDC", "BOMEUSDC", "BTCUSDC", "CRVUSDC", "DATAIPUSDC",
        "DOGEUSDC", "ENAUSDC", "ETHFIUSDC", "ETHUSDC", "FILUSDC",
        "HBARUSDC", "KAITOUSDC", "LINKUSDC", "LTCUSDC", "NEARUSDC",
        "NEOUSDC", "ORDIUSDC", "PENGUUSDC", "PNUTUSDC", "SOLUSDC",
        "SUIUSDC", "TIAUSDC", "TRUMPUSDC", "UNIUSDC", "WIFUSDC",
        "WLDUSDC", "WLFIUSDC", "XRPUSDC", "ZECUSDC"]
print(f"USDC-перпетуалов: {len(SYMS)}", flush=True)

MONTHS = [f"{y}-{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
          if not (y == 2026 and m > 9)]
DAILY = [f"2026-{m:02d}-{d:02d}" for m in (9, 10)
         for d in range(1, 32)
         if not (m == 9 and d < 1) and not (m == 10 and d > 4)]
COLS = ["open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "tbb", "tbq", "ig"]

STATS = {"200": 0, "404": 0, "other": 0, "exc": 0}
def fetch(url):
    """("ok", байты) | ("404", None) | ("err", None). Обрыв сети отличаем
    от отсутствия файла: обрыв -> перезапрос, 404 -> пропускаем."""
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            STATS["200"] += 1
            body = r.read()
            if STATS["200"] <= 2:
                print("HEADER Content-Encoding:",
                      r.headers.get("Content-Encoding"),
                      "| первые байты:", body[:16].hex(), flush=True)
            return "ok", body
    except urllib.error.HTTPError as e:
        if e.code == 404:
            STATS["404"] += 1
            return "404", None
        STATS["other"] += 1
        if STATS["other"] <= 3:
            print("HTTP", e.code, url[:90], flush=True)
    except Exception as e:
        STATS["exc"] += 1
        if STATS["exc"] <= 3:
            print("EXC", str(e)[:90], flush=True)
    return "err", None

def parse(content):
    # content — это ZIP-архив (PK\x03\x04), внутри которого CSV:
    # сначала распаковываем, потом читаем (без этого read_csv видит
    # бинарные байты ZIP-заголовка и падает на декодировании)
    zf2 = zipfile.ZipFile(io.BytesIO(content))
    raw = zf2.read(zf2.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLS,
                     low_memory=False)
    if df["open_time"].dtype == object:     # свежие дампы со строкой-шапкой
        df = df.iloc[1:].reset_index(drop=True)
    ot = pd.to_numeric(df["open_time"]).astype("int64")
    while ot.iloc[0] > 10**14:              # микросекунды -> мс
        ot = ot // 1000
    df["open_time"] = ot
    return df

PARSE_ERR = []
def fetch_parse(url):
    """Качаем с проверкой разбора: обрыв сети ловим перезапросом (до 3)."""
    for _ in range(3):
        st, content = fetch(url)
        if st == "404":
            return None
        if st == "ok":
            try:
                return parse(content)
            except Exception as e:
                if len(PARSE_ERR) < 3:
                    PARSE_ERR.append(f"{type(e).__name__}: {str(e)[:120]}")
                    print("PARSE ОШИБКА:", PARSE_ERR[-1], flush=True)
                continue
        time.sleep(1)
    return None

def get_symbol(sym):
    parts = []
    jobs = [(f"{BASE}/monthly/klines/{sym}/30m/"
             f"{sym}-30m-{mo}.zip", mo) for mo in MONTHS] + \
           [(f"{BASE}/daily/klines/{sym}/30m/"
             f"{sym}-30m-{d}.zip", d) for d in DAILY]
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(lambda j: fetch_parse(j[0]), jobs):
            if res is not None:
                parts.append(res)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "tbb"):
        df[c] = df[c].astype("float32")
    df = df.rename(columns={"tbb": "taker_buy_base"})
    df = df[["open_time", "high", "low", "close", "volume", "quote_volume",
             "taker_buy_base"]]
    return df

t_btc = time.time()
test_r = fetch(f"{BASE}/monthly/klines/BTCUSDC/30m/"
               "BTCUSDC-30m-2026-09.zip")
print("пробный запрос:", "OK" if test_r else "ПУСТО", STATS, flush=True)
BTC_df = get_symbol("BTCUSDC")
print(f"BTCUSDC: {'ПУСТО' if BTC_df is None else len(BTC_df)} баров | "
      f"STATS {STATS} | {time.time()-t_btc:.0f}с", flush=True)
if BTC_df is not None:
    print("BTCUSDC период:", pd.to_datetime(BTC_df['open_time'].iloc[0],
          unit='ms'), "..", pd.to_datetime(BTC_df['open_time'].iloc[-1],
          unit='ms'), flush=True)

def make_feats(df, btc_close, sym):
    close = df["close"]; high, low = df["high"], df["low"]
    rets = close.pct_change()
    X = pd.DataFrame(index=df.index)
    for h in (1, 4, 10, 20):
        X[f"ret_{h}"] = close.pct_change(h).astype("float32")
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    X["atr_pct"] = (tr.rolling(14).mean() / close).astype("float32")
    X["std_20"] = rets.rolling(20).std().astype("float32")
    X["vol_ratio"] = (df["volume"] / df["volume"].rolling(20).mean()
                      .replace(0, np.nan)).astype("float32")
    tb = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    X["taker_buy"] = tb.rolling(20, min_periods=10).mean().astype("float32")
    rng = (high - low).replace(0, np.nan)
    X["clv"] = ((close - low) / rng - 0.5).rolling(20, min_periods=10) \
        .mean().astype("float32")
    sma = close.rolling(200, min_periods=50).mean()
    std = close.rolling(200, min_periods=50).std()
    X["zscore"] = ((close - sma) / std).replace([np.inf, -np.inf], np.nan) \
        .astype("float32")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    X["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) \
        .astype("float32")
    X["skew"] = rets.rolling(20, min_periods=10).skew().astype("float32")
    sma50, sma20 = close.rolling(50).mean(), close.rolling(20).mean()
    X["sma_cross"] = ((close > sma50).astype("float32")
                      - (close > sma20).astype("float32"))
    if sym != "BTCUSDC" and btc_close is not None:
        b = btc_close.reindex(df.index).ffill()
        X["btc_ret_10"] = b.pct_change(10).astype("float32")
        X["btc_ret_20"] = b.pct_change(20).astype("float32")
        bret = b.pct_change()
        X["btc_corr"] = rets.rolling(50, min_periods=20).corr(bret) \
            .astype("float32")
        X["rel_ret_20"] = (rets.rolling(20).sum()
                           - bret.rolling(20).sum()).astype("float32")
        X["rel_ret_60"] = (rets.rolling(60).sum()
                           - bret.rolling(60).sum()).astype("float32")
        cov = rets.rolling(480).cov(bret); var = bret.rolling(480).var()
        X["beta_20"] = (cov / var.replace(0, np.nan)).astype("float32")
        X["corr_20"] = rets.rolling(480, min_periods=100).corr(bret) \
            .astype("float32")
    X["h4_momentum"] = close.pct_change(48).astype("float32")
    X["hour"] = df.index.hour.astype("float32")
    X["dow"] = df.index.dayofweek.astype("float32")
    X["sym_id"] = np.uint16(abs(hash(sym)) % 65536)
    return X

import zlib
frames = []
btc_close = (BTC_df.set_index(
    pd.to_datetime(BTC_df["open_time"], unit="ms"))["close"]
    if BTC_df is not None else None)
empty_syms = []
for k, sym in enumerate(SYMS, 1):
    df = get_symbol(sym)
    if df is None or len(df) < 3000:
        empty_syms.append(sym)
        continue
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    X = make_feats(df, btc_close, sym)
    close = df["close"].values
    fut = np.full(len(df), np.nan, dtype="float32")
    fut[:-H] = close[H:]
    X["y"] = np.where(np.isnan(fut), np.nan,
                      (fut / close > 1).astype("float32"))
    X["fwd"] = (fut / close - 1).astype("float32")
    X["close"] = close
    X["ot"] = df["open_time"].values.astype(np.int64)
    X["low"] = df["low"].values
    X["high"] = df["high"].values
    X["sym_id"] = np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=[c for c in X.columns
                                   if c not in ("fwd", "close")]))
    if k % 10 == 0:
        print(f"{k}/{len(SYMS)} символов | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
print("пустых символов:", len(empty_syms), empty_syms[:8], flush=True)
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
test_q = quarters[4:] if len(quarters) >= 8 else quarters[3:]
FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
print(f"строк {len(data)}, кварталов {len(quarters)}, "
      f"тест {len(test_q)}: {test_q[0]}..{test_q[-1]} "
      f"| {time.time()-t0:.0f}с", flush=True)

def simulate(d, gate, scen):
    """Возврат списка pnl (доля). scen: S20/S10/S00."""
    pools = {int(fi): gg.sort_values("ot") for fi, gg in d.groupby("sym_id")}
    rows = []
    for side in (1, -1):
        sel = d[(d["p"] > gate) if side == 1 else (d["p"] < 1 - gate)] \
            .sort_values(["sym_id", "ot"])
        last = {}
        for row in sel.itertuples():
            if row.ot - last.get(row.sym_id, -10**18) < COOLDOWN_MS:
                continue
            last[row.sym_id] = row.ot
            pool = pools.get(int(row.sym_id))
            if pool is None:
                continue
            ots = pool["ot"].values
            j = int(np.searchsorted(ots, row.ot))
            if j >= len(ots) or ots[j] != row.ot or j + H + 2 >= len(pool):
                continue
            c = row.close
            nxt_lo = pool["low"].values[j + 1]
            nxt_hi = pool["high"].values[j + 1]
            filled = (nxt_lo <= c) if side == 1 else (nxt_hi >= c)
            if not filled:
                continue                       # живой GTX: цена ушла - пропуск
            exit_close = pool["close"].values[j + H]
            if np.isnan(exit_close):
                continue
            if scen == "S20":                  # тейкер-мир USDT
                e = c * (1 + side * SLIP)
                x = exit_close * (1 - side * SLIP)
                fee = 0.001
            elif scen == "S10":                # мейкер-вход 0, тейкер-выход
                e = c
                x = exit_close * (1 - side * SLIP)
                fee = 0.0005
            else:                              # S00: оба мейкер 0
                e = c
                x = exit_close
                # выход лимиткой по close[t+H]: заполняется если следующий
                # бар коснулся; иначе тейкер на close[t+H+1]
                px_lo = pool["low"].values[j + H + 1]
                px_hi = pool["high"].values[j + H + 1]
                touch = (px_hi >= x) if side == 1 else (px_lo <= x)
                if not touch:
                    x = pool["close"].values[j + H + 1]
                    x = x * (1 - side * SLIP)
                    fee = 0.0005
                else:
                    fee = 0.0
            pnl = side * (x / e - 1) - fee
            rows.append({"pnl": pnl, "quarter": row.quarter,
                         "side": "L" if side == 1 else "S"})
    return pd.DataFrame(rows)

def null_pct(te, n_real, seeds=50):
    rng = np.random.default_rng(7)
    sub = te[["close", "fwd"]].dropna()
    if n_real < 20 or len(sub) < n_real:
        return None
    out = []
    for _ in range(seeds):
        s = sub.sample(n=n_real, random_state=rng.integers(1 << 30))
        e = s["close"].values * (1 + SLIP)
        x = s["close"].values * (1 + s["fwd"].values) * (1 - SLIP)
        out.append((x / e - 1 - 0.001).mean() * 100)
    return out

acc = {g: {s: [] for s in ("S20", "S10", "S00")} for g in GATES}
nulls = {g: [] for g in GATES}
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 2000 or len(trd) < 50_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(trd[FEATS], trd["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    for g in GATES:
        for scen in ("S20", "S10", "S00"):
            T = simulate(d, g, scen)
            acc[g][scen].append(T)
    if k >= len(test_q) - 2:
        for g in GATES:
            T = pd.concat(acc[g]["S20"], ignore_index=True)
            Tq = T[T["quarter"] == qq]
            nl = null_pct(ted, len(Tq))
            if nl and len(Tq):
                nulls[g].append(float(np.mean(np.array(nl)
                                              < Tq["pnl"].mean() * 100) * 100))
    print(f"фолд {qq} готов | {time.time()-t0:.0f}с", flush=True)

report = {"pre_reg": "S00 жив: avg>=+2бп, плюс-кв>=60%, нуль>=95",
          "quarters": test_q, "gates": {}}
for g in GATES:
    row = {}
    for scen in ("S20", "S10", "S00"):
        T = pd.concat(acc[g][scen], ignore_index=True)
        if not len(T):
            row[scen] = {"n": 0}
            continue
        avg = float(T["pnl"].mean() * 10000)
        by_q = {k: round(v * 10000, 2) for k, v in
                T.groupby("quarter")["pnl"].mean().items()}
        pos_q = sum(1 for v in by_q.values() if v > 0)
        row[scen] = {"n": int(len(T)), "avg_bp": round(avg, 2),
                     "pos_quarters": pos_q, "counted": len(by_q),
                     "by_quarter": by_q}
    nl_pct = int(np.mean(nulls[g])) if nulls[g] else None
    row["null_pct"] = nl_pct
    s00 = row["S00"]
    alive = (s00.get("n", 0) > 0 and s00.get("avg_bp", -99) >= 2.0
             and s00.get("pos_quarters", 0) / max(1, s00.get("counted", 1))
             >= 0.6 and (nl_pct or 0) >= 95)
    row["alive"] = bool(alive)
    report["gates"][str(g)] = row
    print(f"gate {g}: S20 {row['S20'].get('avg_bp')}бп ({row['S20'].get('n')}) "
          f"| S10 {row['S10'].get('avg_bp')}бп | S00 {s00.get('avg_bp')}бп "
          f"({s00.get('pos_quarters')}/{s00.get('counted')}) нуль {nl_pct}% "
          f"-> {'ЖИВ' if alive else 'нет'}", flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1)
print("USDC-мейкер-экзамен завершён", flush=True)
