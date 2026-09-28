import urllib.request
def get(url):
    try:
        r = urllib.request.urlopen(url, timeout=20)
        return f"{r.status} {len(r.read())} байт"
    except Exception as e:
        return f"ERR {type(e).__name__}: {str(e)[:160]}"
print("CDN object:", get("https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/30m/BTCUSDT-30m-2024-01.zip"), flush=True)
print("S3 direct:", get("https://s3.ap-northeast-1.amazonaws.com/data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/30m/BTCUSDT-30m-2024-01.zip"), flush=True)
print("S3 list:", get("https://s3.ap-northeast-1.amazonaws.com/data.binance.vision?list-type=2&prefix=data/futures/um/monthly/klines/BTCUSDT/30m/&max-keys=2")[:120], flush=True)
