"""Единая система: два двигателя, один демо-счёт.

  ML-5Ч  (USDT-M, топ-20)  — 5-часовые позиции по уверенности модели.
  СТАКАН (USDC-M, топ-10)  — 5-минутные позиции по микроструктуре.

Запуск: python main.py  (в Docker: CMD). Данные — мейннет (публичные),
исполнение — демо. Каждая сделка и каждый сигнал — в журналы, отчёт
в Telegram каждый час.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bot.config import Config
from bot.executor import Executor
import bot.telegram as tg
import bot.timesync as timesync
from lob.engine import LobBot
from lob import recorder as lob_recorder
from ml5h.engine import Ml5hEngine

log = logging.getLogger("main")


async def hourly_report(cfg: Config, ml5h: Ml5hEngine, lob: LobBot) -> None:
    last_hour = datetime.now(timezone.utc).hour
    while True:
        await asyncio.sleep(30)
        now = datetime.now(timezone.utc)
        if now.hour == last_hour:
            continue
        last_hour = now.hour
        try:
            ml_open = len(ml5h.pos)
            ml_day = float(ml5h.day_pnl)
            lob_open = len(lob.pos)
            lob_day = float(lob.day_pnl)
            ml_model = "готова" if ml5h.model else "ждёт артефакт"
            lob_model = "готова" if lob.model else \
                f"копит данные ({lob.lob.first_train_h}ч)"
            tg.fire(
                f"📊 <b>{now.strftime('%H:%M UTC')} [DEMO]</b>\n"
                f"ML-5ч (USDT): позиций {ml_open}, день {ml_day:+.2f} USDT, "
                f"модель {ml_model}\n"
                f"Стакан (USDC): позиций {lob_open}, день {lob_day:+.2f} "
                f"USDC, модель {lob_model}")
        except Exception:
            log.exception("hourly report")


async def main() -> None:
    log_path = Path(os.environ.get("BOT_LOG_PATH", ROOT / "data" / "bot.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_path)])
    cfg = Config.from_env(ROOT)
    (cfg.data_dir / "lob" / "archive").mkdir(parents=True, exist_ok=True)
    if cfg.telegram_token and cfg.telegram_chat_id:
        tg.init(cfg.telegram_token, cfg.telegram_chat_id,
                env=cfg.mode.value)
        tg.fire("🤖 Единая система запущена: ML-5ч + стакан. Демо.")
    timesync.install()

    ex = Executor(cfg, filters={})
    await asyncio.to_thread(ex.ensure_position_mode, False)  # ONE-WAY
    off = await asyncio.to_thread(timesync.measure, ex.client.rest_api)
    log.info("время: офсет %+d мс", off)

    tasks = []
    ml5h: Ml5hEngine | None = None
    lob: LobBot | None = None
    if cfg.ml5h.enabled:
        ml5h = Ml5hEngine(cfg, ex)
        await ml5h.setup()
        tasks.append(ml5h.run())
        log.info("ML-5Ч: %d символов USDT", len(ml5h.symbols))
    if cfg.lob.enabled:
        lob = LobBot(cfg, ex)
        await asyncio.to_thread(lob.setup)
        lob.try_load_model()
        tasks.append(asyncio.create_task(lob_recorder.run(lob.SYMS)))
        tasks.append(lob.run_loop())
        log.info("СТАКАН: %d символов USDC", len(lob.SYMS))
    tasks.append(hourly_report(cfg, ml5h, lob))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("остановлено вручную")
