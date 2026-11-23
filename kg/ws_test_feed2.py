"""Диагностический прогон: подробный журнал + счётчики по типам событий."""
import asyncio, json, logging, sys, time
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
sys.path.insert(0, "/repo")
from bot.streams import MarketStreams

async def main():
    passport = json.load(open("/repo/models/ml5h_meta.json"))
    symbols = passport["symbols"]
    feed = MarketStreams(symbols)
    await feed.start()
    print(f"=== подписка отправлена для {len(symbols)} монет ===", flush=True)
    await asyncio.sleep(120)
    print(f"=== через 120 сек: событий {feed.events_total:,} | "
          f"монет с барами {len(feed.bars)} | "
          f"книга {len(feed.book)} | "
          f"последнее {time.time() - feed.last_event:.1f}с назад ===",
          flush=True)

asyncio.run(main())
