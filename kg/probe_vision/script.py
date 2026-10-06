"""ЩУП: какие URL data.binance.vision видит Кегл (диагностика 404)."""
import urllib.request, urllib.error

URLS = [
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/30m/BTCUSDT-30m-2026-08.zip",
    "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDC/30m/BTCUSDC-30m-2026-09.zip",
    "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDC/30m/BTCUSDC-30m-2026-10-04.zip",
    "https://data.binance.vision/?prefix=data/futures/um/monthly/klines/BTCUSDC/",
    "https://data.binance.vision/",
]
for url in URLS:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            print(r.status, url[:100], flush=True)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, url[:100], flush=True)
    except Exception as e:
        print("EXC", str(e)[:80], url[:80], flush=True)
