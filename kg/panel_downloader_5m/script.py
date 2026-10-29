"""Кегл-кернел: панель 5м свечей из ДАМПОВ data.binance.vision
(те же монеты и месяцы, что у боевой панели 30м — months5y + свежие дни).
Нужна для раунда 2 «шаг свечи»: переучивание связки на 5м с перебором
горизонтов (15м/30м/1ч дальше собираются ресемплом).
Пишет streams-ом: монета за монетой в ОДИН zip, не держит всё в памяти
(5м-панель в 6 раз жирнее 30м — ~25ГБ текстом, в zip ~6-9ГБ).
"""
import io
import urllib.request
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

# Кегль менял пути подключения (то /kaggle/input/<имя>, то
# /kaggle/input/datasets/<юзер>/<имя>) — ищем файл по всему дереву
META_PATH = None
for root, _d, fs in os.walk("/kaggle/input"):
    if "panel_meta.json" in fs:
        META_PATH = os.path.join(root, "panel_meta.json")
        break
assert META_PATH, "panel_meta.json не найден"
META = json.load(open(META_PATH))
SYMS = META["syms"]
MONTHS = META.get("months5y") or META.get("months") or []
EXTRA_DAYS = META.get("extra_days", [])  # дневные пакеты свежих дней
OUT = "/kaggle/working"
BASE = "https://data.binance.vision/data/futures/um/monthly/klines/{sym}/5m/{sym}-5m-{month}.zip"
BASE_D = "https://data.binance.vision/data/futures/um/daily/klines/{sym}/5m/{sym}-5m-{day}.zip"

errs = []


def fetch(url):
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
    """Возвращает (sym, текст CSV) или None; результат сразу пишется в zip
    главным потоком (ленивый ex.map), память не копится."""
    sym, i, total = args
    rows = []
    urls = [BASE.format(sym=sym, month=m) for m in MONTHS] + \
           [BASE_D.format(sym=sym, day=d) for d in EXTRA_DAYS]
    for url in urls:
        data = fetch(url)
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
    if i % 50 == 0:
        print(f"{i}/{total} {sym}: {len(clean)} строк | {time.time()-T0:.0f}с",
              flush=True)
    return (sym, "\n".join(out))


T0 = time.time()
n_ok = 0
with zipfile.ZipFile(f"{OUT}/panel_5m.zip", "w",
                     compression=zipfile.ZIP_DEFLATED) as z, \
     ThreadPoolExecutor(16) as ex:
    for res in ex.map(get_symbol,
                      [(s, i, len(SYMS)) for i, s in enumerate(SYMS, 1)]):
        if res:
            sym, text = res
            z.writestr(f"{sym}_5m.csv", text)
            n_ok += 1
print(f"ГОТОВО за {time.time() - T0:.0f}с: {n_ok}/{len(SYMS)} монет -> "
      f"panel_5m.zip ({os.path.getsize(OUT + '/panel_5m.zip')/2**30:.2f}ГБ)",
      flush=True)
if errs:
    print("ошибки сети:", errs, flush=True)
