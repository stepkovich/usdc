import asyncio, json, sys, time
sys.path.insert(0, "/repo")
from bot.streams import MarketStreams

async def main():
    passport = json.load(open("/repo/models/ml5h_meta.json"))
    feed = MarketStreams(passport["symbols"])
    await feed.start()
    await asyncio.sleep(45)
    print(f"=== событий {feed.events_total:,} | живых свечей {len(feed.live_bar)} ===",
          flush=True)
    from binance_common.configuration import ConfigurationRestAPI
    from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
    from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
        DerivativesTradingUsdsFutures)
    client = DerivativesTradingUsdsFutures(config_rest_api=ConfigurationRestAPI(
        base_path=DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL))
    coins = [s for s in passport["symbols"] if s in feed.live_bar][:5]
    print("монет для сверки:", len(coins), flush=True)
    checked = ok = 0
    for sym in coins:
        live = feed.live_bar[sym]
        resp = client.rest_api.kline_candlestick_data(symbol=sym,
                                                            interval="30m",
                                                            limit=2)
        rows = getattr(resp.data(), "root", None) or resp.data()
        if len(rows) < 2:
            print(f"{sym}: REST вернул {len(rows)} строк"); continue
        r = rows[-1]                      # сырой массив: [время, откр, макс, мин, закр, ...]
        o_rest = float(r[1])
        t_rest = int(r[0])
        m = abs(o_rest - live["open"]) < 1e-6 and t_rest == live["open_time"]
        checked += 1
        ok += 1 if m else 0
        print(f"{sym}: поток откр={live['open']} t={live['open_time']} | "
              f"REST откр={o_rest} t={t_rest} | "
              f"{'СОВПАЛО' if m else 'РАСХОЖДЕНИЕ'}", flush=True)
    print(f"\nСВЕРКА ЖИВЫХ СВЕЧЕЙ С ЭТАЛОНОМ: {ok}/{checked}", flush=True)

asyncio.run(main())
