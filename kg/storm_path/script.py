"""ПУТЬ СДЕЛОК БУРИ: ТЕЙК/СТОП В КРАТНЫХ ATR (предрегистрация 07.10).

ВОПРОС: боевая модель-буря выходит будильником (close[t+10], обе ноги
тейкер). Но мы знаем: её край живёт на бурях — и не мерили ПУТЬ:
сколько выигрышных сделок коснулись бы +k×ATR по дороге, сколько
проигрышных можно обрезать раньше?

ВАРИАНТЫ (записаны до прогона, симметричные, на ОДНИХ сделках):
  BASE    как в бою: будильник, close[t+10];
  TP_SL_05  тейк +0.5×ATR, стоп -0.5×ATR;
  TP_SL_10  +1.0×ATR / -1.0×ATR;
  TP_SL_15  +1.5×ATR / -1.5×ATR;
  TP_SL_20  +2.0×ATR / -2.0×ATR.
МЕХАНИКА ГОНКИ (по 30м барам удержания): тейк — лоу/хай коснулся
уровня; стоп — коснулся; ОБА уровня внутри одного бара -> считаем
СТОП ПЕРВЫМ (консервативно, порядок внутри бара невидим). Результат:
тейк -> +k×ATR (выход мейкер, 0); стоп -> -k×ATR (выход рынок);
таймаут -> side×fwd (выход рынок). Вход тейкер у всех (5бп).
ИТОГО: тейк = +k×ATR − 5бп; стоп = −k×ATR − 5бп; таймаут =
side×fwd − 10бп. BASE = side×fwd − 10бп.
ВЕРДИКТ (до прогона): k — кандидат, если пул (7 валидных фолдов)
avg(k) − avg(BASE) >= +3 бп при n >= 1000. Оговорка: стоп в ATR
меняет характер сигнала (буря любит отскоки) — потому честная гонка
с консервативным правилом и полная таблица, без выбора лучшего
постфактум как ответа. Ничего не деплоится.
"""
import glob, io, json, time, zipfile, zlib
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
H = 10
GATE = 0.65
COOLDOWN_MS = 10 * 1800_000
KS = [0.5, 1.0, 1.5, 2.0]
FEE_IN = 0.0005
FEE_OUT_TAKER = 0.0005
PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              n_jobs=4, verbosity=-1, class_weight="balanced")

zips = glob.glob("/kaggle/input/**/panel.zip", recursive=True)
if not zips:
    json.dump({"error": "panel.zip not found"},
              open("/kaggle/working/report.json", "w"), indent=1)
    raise SystemExit(0)
zf = zipfile.ZipFile(zips[0])
members = [n for n in zf.namelist() if n.endswith("_30m.csv")]
print("монет:", len(members), flush=True)
btc_pre = pd.read_csv(io.BytesIO(zf.read("BTCUSDT_30m.csv")),
                      usecols=["open_time", "close"])
btc_pre["open_time"] = btc_pre["open_time"].astype(np.int64)
btc_pre.index = pd.to_datetime(btc_pre["open_time"], unit="ms")
btc = btc_pre["close"].astype("float32")

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
    X["zscore"] = ((close - sma) / std).replace([np.inf, -np.inf],
                                                np.nan).astype("float32")
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    X["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) \
        .astype("float32")
    X["skew"] = rets.rolling(20, min_periods=10).skew().astype("float32")
    sma50, sma20 = close.rolling(50).mean(), close.rolling(20).mean()
    X["sma_cross"] = ((close > sma50).astype("float32")
                      - (close > sma20).astype("float32"))
    if sym != "BTCUSDT" and btc_close is not None:
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
    X["sym_id"] = np.uint16(zlib.crc32(sym.encode()) & 0xFFFF)
    return X, tr

frames = []
for i, name in enumerate(members, 1):
    sym = name.replace("_30m.csv", "")
    df = pd.read_csv(io.BytesIO(zf.read(name)),
                     usecols=["open_time", "high", "low", "close",
                              "volume", "quote_volume", "taker_buy_base"])
    for c in df.columns:
        if c != "open_time":
            df[c] = df[c].astype("float32")
    df["open_time"] = df["open_time"].astype(np.int64)
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    X, tr = make_feats(df, btc, sym)
    close = df["close"].values
    atr_rel = (tr.rolling(14).mean() / close).values.astype("float64")
    fut = np.full(len(df), np.nan, dtype="float64")
    fut[:-H] = close[H:]
    fwd = fut / close - 1
    thr = 0.5 * np.nan_to_num(atr_rel, nan=np.nan)
    X["y"] = np.where(np.isnan(fwd) | np.isnan(thr), np.nan,
                      np.where(fwd > thr, 1.0,
                               np.where(fwd < -thr, 0.0, np.nan)))
    X["fwd"] = fwd.astype("float32")
    X["atr_rel"] = atr_rel.astype("float32")
    X["close"] = close
    X["high"] = df["high"].values
    X["low"] = df["low"].values
    X["ot"] = df["open_time"].values.astype(np.int64)
    X = X.replace([np.inf, -np.inf], np.nan)
    frames.append(X.dropna(subset=["atr_pct"]))
    if i % 100 == 0:
        print(f"{i}/{len(members)} | {time.time()-t0:.0f}с", flush=True)

data = pd.concat(frames, ignore_index=True)
del frames
q = pd.PeriodIndex(pd.to_datetime(data["ot"], unit="ms"), freq="Q")
data["quarter"] = q.astype(str)
quarters = sorted(data["quarter"].unique())
idx = np.linspace(0, len(quarters) - 1, 8).round().astype(int)
test_q = [quarters[i] for i in idx]
FEATS = ["ret_1", "ret_4", "ret_10", "ret_20", "h4_momentum",
         "atr_pct", "std_20", "skew", "vol_ratio", "taker_buy", "clv",
         "zscore", "rsi", "sma_cross", "btc_ret_10", "btc_ret_20",
         "btc_corr", "rel_ret_20", "rel_ret_60", "beta_20", "corr_20",
         "hour", "dow", "sym_id"]
print(f"строк {len(data)}, фолды {test_q} | {time.time()-t0:.0f}с",
      flush=True)

# сбор сделок с ПУТЁМ (высоты/низы баров удержания)
def collect(d):
    pools = {int(fi): gg.sort_values("ot") for fi, gg in d.groupby("sym_id")}
    trades = []
    for side in (1, -1):
        sel = d[(d["p"] > GATE) if side == 1 else (d["p"] < 1 - GATE)] \
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
            if j >= len(ots) or ots[j] != row.ot or j + H >= len(ots):
                continue
            seg_h = pool["high"].values[j + 1: j + 1 + H].astype("float64")
            seg_l = pool["low"].values[j + 1: j + 1 + H].astype("float64")
            if len(seg_h) < H:
                continue
            trades.append({"side": side, "quarter": row.quarter,
                           "entry": float(row.close),
                           "atr_rel": float(row.atr_rel),
                           "fwd": float(row.fwd) if row.fwd == row.fwd
                           else np.nan,
                           "seg_h": seg_h, "seg_l": seg_l})
    return trades

COOLDOWN_MS = COOLDOWN_MS

def race_pnl(t, atr_rel, side, seg_h, seg_l, k):
    """Гонка уровней в барам: тейк +k×atr, стоп -k×atr.
    Оба в одном баре = СТОП (консервативно). Нет касаний = None."""
    tp = k * atr_rel
    if side == 1:
        hit_tp = np.nonzero(seg_h >= t + tp)[0]
        hit_sl = np.nonzero(seg_l <= t - tp)[0]
    else:
        hit_tp = np.nonzero(seg_l <= t - tp)[0]
        hit_sl = np.nonzero(seg_h >= t + tp)[0]
    i_tp = int(hit_tp[0]) if len(hit_tp) else 10**9
    i_sl = int(hit_sl[0]) if len(hit_sl) else 10**9
    if i_tp == 10**9 and i_sl == 10**9:
        return None, None, None
    if i_sl <= i_tp:                        # оба в одном баре = стоп
        return -tp - FEE_OUT_TAKER, "sl", i_sl
    return tp - FEE_OUT_TAKER, "tp", i_tp

COOLDOWN_MS = 10 * 1800_000
FEE_OUT_TAKER = 0.0005
FEE_ALL_TAKER = 0.0010

acc = {("BASE"): [], **{f"K{k}": [] for k in KS}}
fills = {f"K{k}": {"tp": 0, "sl": 0, "to": 0} for k in KS}
all_folds = []
for k, qq in enumerate(test_q):
    trd = data[data["quarter"] < qq]
    ted = data[data["quarter"] == qq]
    if len(ted) < 20000 or len(trd) < 500_000:
        continue
    m = lgb.LGBMClassifier(**PARAMS)
    trd_ok = trd.dropna(subset=["y"])
    m.fit(trd_ok[FEATS], trd_ok["y"], categorical_feature=["sym_id"])
    d = ted.assign(p=m.predict_proba(ted[FEATS])[:, 1])
    trades = collect(d)
    for t in trades:
        base = t["side"] * t["fwd"] - FEE_ALL_TAKER
        acc["BASE"].append(base)
        for k in KS:
            pnl, out, _ = race_pnl(t["entry"], t["atr_rel"], t["side"],
                                   t["seg_h"], t["seg_l"], k)
            if pnl is None:
                pnl = t["side"] * t["fwd"] - FEE_ALL_TAKER
                out = "timeout"
            fills[f"K{k}"][out] = fills[f"K{k}"].get(out, 0) + 1
            acc[f"K{k}"].append(pnl)
    all_folds.append(qq)
    print(f"фолд {qq}: сделок {len(trades)} | {time.time()-t0:.0f}с",
          flush=True)

report = {"folds": [str(f) for f in all_folds],
          "pre_reg": "кандидат: пул avg(k×ATR) − avg(BASE) >= +3 бп "
                     "при n >= 1000; консервативное правило гонки "
                     "(оба в баре = стоп)",
          "variants": {}}
base_avg = float(np.mean(acc["BASE"]) * 10000)
nb = len(acc["BASE"])
report["variants"]["BASE"] = {"n": nb, "avg_bp": round(base_avg, 2)}
print(f"BASE: {nb} сделок, {base_avg:+.2f} бп", flush=True)
for k in KS:
    a = np.array(acc[f"K{k}"], dtype="float64")
    avg = float(np.nanmean(a) * 10000)
    n_f = fills[f"K{k}"]
    tot = max(1, sum(n_f.values()))
    delta = avg - base_avg
    report["variants"][f"K{k}"] = {
        "avg_bp": round(avg, 2), "delta_vs_base": round(delta, 2),
        "tp_share": round(n_f["tp"] / tot * 100, 1),
        "sl_share": round(n_f["sl"] / tot * 100, 1),
        "timeout_share": round(n_f["timeout"] / tot * 100, 1),
        "candidate": bool(delta >= 3.0 and nb >= 1000)}
    print(f"K{k}: avg {avg:+.2f} бп (Δ{delta:+.2f}), тейк-первых "
          f"{n_f['tp']/tot*100:.0f}%, стоп-первых {n_f['sl']/tot*100:.0f}%, "
          f"таймаут {n_f['timeout']/tot*100:.0f}% -> "
          f"{'КАНДИДАТ' if report['variants'][f'K{k}']['candidate'] else ''}",
          flush=True)
json.dump(report, open("/kaggle/working/report.json", "w"), indent=1,
          ensure_ascii=False)
print("ПУТЬ БУРИ завершён", flush=True)
