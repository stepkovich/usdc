"""Кегл-кернел v3: панель из ДАМПОВ data.binance.vision (API Биненса
блокирует US-IP — HTTP 451; дампы открыты). Месячные zip 30м свечей,
список монет вшит с сервера. open_time в свежих дампах — микросекунды,
детектируем и приводим к миллисекундам."""
import io
import urllib.request
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

META_PATH = "/kaggle/input/usdc-panel-meta/panel_meta.json"
if not os.path.exists(META_PATH):
    print("input-дерево:", flush=True)
    for root, dirs, fs in os.walk("/kaggle/input"):
        print(" ", root, len(fs), flush=True)
    raise SystemExit(1)
META = json.load(open(META_PATH))
SYMS = META["syms"]
MONTHS = META["months5y"]
OUT = "/kaggle/working"
BASE = "https://data.binance.vision/data/futures/um/monthly/klines/{sym}/30m/{sym}-30m-{month}.zip"


errs = []

def fetch(sym, month):
    url = BASE.format(sym=sym, month=month)
    try:
        r = urllib.request.urlopen(url, timeout=30)
        return r.read()
    except Exception as e:
        if len(errs) < 3:
            errs.append(f"{type(e).__name__}: {str(e)[:120]}")
            if len(errs) == 1:
                print("ПЕРВАЯ ОШИБКА СЕТИ:", errs[0], flush=True)
        return None


def get_symbol(args):
    """Возвращает (sym, текст CSV) — панель собирается в ОДИН zip
    (лимит Кегла: 500 файлов в output, у нас 527 CSV)."""
    sym, i, total = args
    rows = []
    for month in MONTHS:
        data = fetch(sym, month)
        if not data:
            continue
        z = zipfile.ZipFile(io.BytesIO(data))
        for name in z.namelist():
            rows.extend(z.read(name).decode().strip().split("\n"))
    if not rows:
        print(f"{i}/{total} {sym}: пусто", flush=True)
        return None
    clean = [r for r in rows if r and not r.startswith("open_time")]
    out = ["open_time,open,high,low,close,volume,close_time,quote_volume,"
           "n,taker_buy_base,taker_buy_quote,ignore"]
    for r in clean:
        c = r.split(",")
        ot = int(float(c[0]))
        if ot > 10 ** 14:            # микросекунды -> миллисекунды
            ot //= 1000
        out.append(",".join([str(ot)] + c[1:]))
    print(f"{i}/{total} {sym}: {len(clean)} строк", flush=True)
    return (sym, "\n".join(out))


t0 = time.time()
with ThreadPoolExecutor(16) as ex:
    results = list(ex.map(get_symbol,
                          [(s, i, len(SYMS)) for i, s in enumerate(SYMS, 1)]))
ok = [r for r in results if r]
with zipfile.ZipFile(f"{OUT}/panel.zip", "w",
                     compression=zipfile.ZIP_DEFLATED) as z:
    for sym, text in ok:
        z.writestr(f"{sym}_30m.csv", text)
print(f"ГОТОВО за {time.time() - t0:.0f}с: {len(ok)}/{len(SYMS)} монет -> "
      f"panel.zip", flush=True)
if errs:
    print("ошибки сети:", errs, flush=True)
