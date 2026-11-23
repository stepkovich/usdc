import asyncio, json, sys, time
sys.path.insert(0, "/repo")
from bot.streams import MarketStreams

async def main():
    passport = json.load(open("/repo/models/ml5h_meta.json"))
    print("паспорт: первые 3:", passport["symbols"][:3], flush=True)
    feed = MarketStreams(passport["symbols"])
    await feed.start()
    await asyncio.sleep(30)
    keys = list(feed.live_bar.keys())
    print("поток: живых свечей", len(keys), "| первые 3 ключа:", keys[:3], flush=True)
    print("пересечение имён:", len(set(passport["symbols"]) & set(keys)), flush=True)

asyncio.run(main())
