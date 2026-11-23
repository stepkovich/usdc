"""Проверка живой свечи из потока против REST: цена открытия и время
начала текущего получаса совпадать обязаны (они фиксированы биржей)."""
import asyncio, json, sys, time
sys.path.insert(0, "/repo")
from bot.streams import MarketStreams

async def main():
    passport = json.load(open("/repo/models/ml5h_meta.json"))
    feed = MarketStreams(passport["symbols"])
    await feed.start()
    await asyncio.sleep(45)           # 45 сек потока — события точно придут
    print(f"=== ПОТОК: событий {feed.events_total:,} | живых свечей "
          f"{len(feed.live_bar)} | закрытых (через границы) "
          f"{len(feed.bars)} ===", flush=True)
    from binance_common.configuration import ConfigurationRestAPI
    from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
    from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
        DerivativesTradingUsdsFutures)
    client = DerivativesTradingUsdsFutures(config_rest_api=ConfigurationRestAPI(
        base_path=DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL))
    coins = [s for s in passport["symbols"] if s in feed.live_bar][:5]
    checked = ok = 0
    for sym in coins:
        live = feed.live_bar[sym]
        try:
            resp = await client.rest_api.klines(symbol=sym, interval="30m",
                                                limit=2)
            rows = getattr(resp, "actual_instance", resp) or []
            if len(rows) < 2: continue
            r = rows[-1].model_dump(by_alias=True)     # текущая незакрытая свеча
            o_rest = float(r.get("open", 0) or 0)
            t_rest = int(r.get("open_time", 0) or 0)
            m_open = abs(o_rest - live["open"]) < 1e-6
            m_time = (t_rest - 1800_000) == live["open_time"]
            checked += 1
            ok += 1 if (m_open and m_time) else 0
            print(f"{sym}: поток откр={live['open']} время={live['open_time']} | "
                  f"REST откр={o_rest} время={t_rest} | "
                  f"{'СОВПАЛО' if m_open and m_time else 'РАСХОЖДЕНИЕ'}",
                  flush=True)
        except Exception as e:
            print(f"{sym}: ошибка сверки {str(e)[:60]}", flush=True)
    print(f"\nСВЕРКА ЖИВЫХ СВЕЧЕЙ С ЭТАЛОНОМ: {ok}/{checked} совпало", flush=True)

asyncio.run(main())
