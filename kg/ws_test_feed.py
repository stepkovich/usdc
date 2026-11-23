"""Живая проверка потокового слоя: 3 минуты слушать свечи всех монет,
затем сравнить пришедшие закрытые свечи с эталоном REST (источник правды)."""
import asyncio, json, sys, time
sys.path.insert(0, "/repo")
from bot.streams import MarketStreams

async def main():
    passport = json.load(open("/repo/models/ml5h_meta.json"))
    symbols = passport["symbols"]
    feed = MarketStreams(symbols)
    t0 = time.time()
    await feed.start()
    await asyncio.sleep(180)          # 3 минуты реального потока
    dt = time.time() - t0
    print(f"=== СТАТИСТИКА ПОТОКА ({dt:.0f} сек) ===")
    print(feed.status())
    print(f"монет подписано: {len(symbols)}")
    print(f"последнее событие: {time.time() - feed.last_event:.1f} сек назад")
    # сравнение с REST-эталоном по 5 монетам с барами
    from binance_common.configuration import ConfigurationRestAPI
    from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
    from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
        DerivativesTradingUsdsFutures, ConfigurationWebSocketStreams)
    client = DerivativesTradingUsdsFutures(config_rest_api=ConfigurationRestAPI(
        base_path=DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL))
    checked = ok = 0
    with_bars = [s for s in symbols if s in feed.bars]
    print(f"монет с закрытыми барами из потока: {len(with_bars)}")
    for sym in with_bars[:40:8] or with_bars[:5]:
        bar = feed.bars[sym]
        try:
            resp = await client.rest_api.klines(symbol=sym, interval="30m",
                                                limit=3)
            rows = getattr(resp, "actual_instance", resp) or []
            if not rows: continue
            r = rows[-2] if len(rows) >= 2 else rows[-1]  # предпоследняя = последняя закрытая
            to = r.model_dump(by_alias=True) if hasattr(r, "model_dump") else {}
            o = float(to.get("open", 0) or 0); c = float(to.get("close", 0) or 0)
            h = float(to.get("high", 0) or 0); l = float(to.get("low", 0) or 0)
            m = (abs(o - bar["open"]) < 1e-6 and abs(c - bar["close"]) < 1e-6
                 and abs(h - bar["high"]) < 1e-6 and abs(l - bar["low"]) < 1e-6)
            checked += 1
            ok += 1 if m else 0
            print(f"{sym}: поток o={bar['open']} c={bar['close']} | "
                  f"REST o={o} c={c} | {'СОВПАЛО' if m else 'РАСХОЖДЕНИЕ'}")
        except Exception as e:
            print(f"{sym}: REST-сверка не удалась ({str(e)[:60]})")
    print(f"\nСВЕРКА С ЭТАЛОНОМ: {ok}/{checked} совпало")

asyncio.run(main())
