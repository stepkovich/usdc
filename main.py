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

from bot.config import Config, Mode
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
            # движок может быть выключен конфигом (реал = только Буря):
            # тогда его блок в отчёте просто не печатается
            ml_block = ""
            if ml5h is not None:
                ml_model = "готова" if ml5h.model else "ждёт артефакт"
                ml_block = (f"ML-5ч (USDT): позиций {len(ml5h.pos)}, "
                            f"день {float(ml5h.day_pnl):+.2f} USDT, "
                            f"модель {ml_model}\n")
            lob_block = ""
            if lob is not None:
                lob_model = "готова" if lob.model else \
                    f"копит данные ({lob.lob.first_train_h}ч)"
                lob_block = (f"Стакан (USDC): позиций {len(lob.pos)}, "
                             f"день {float(lob.day_pnl):+.2f} USDC, "
                             f"модель {lob_model}\n")
            env_tag = "DEMO" if not cfg.dry_run and \
                cfg.mode != Mode.MAINNET else \
                ("СУХОЙ РЕЖИМ MAINNET" if cfg.dry_run else "MAINNET")
            tg.fire(
                f"📊 <b>{now.strftime('%H:%M UTC')} [{env_tag}]</b>\n"
                f"{ml_block}{lob_block}")
        except Exception:
            log.exception("hourly report")


async def main() -> None:
    log_path = Path(os.environ.get("BOT_LOG_PATH", ROOT / "data" / "bot.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024,
                                      backupCount=2)])
    cfg = Config.from_env(ROOT)
    (cfg.data_dir / "lob" / "archive").mkdir(parents=True, exist_ok=True)
    if cfg.telegram_token and cfg.telegram_chat_id:
        tg.init(cfg.telegram_token, cfg.telegram_chat_id,
                env=cfg.mode.value)
        tg.fire("🤖 Единая система запущена: ML-5ч + стакан. Демо.")
    timesync.install()

    ex = Executor(cfg, filters={})
    # сверка часов ДО первого подписанного запроса: на VPS без RTC
    # и с плавающим NAT подпись с неотмеренным офсетом ловит -1021
    off = await asyncio.to_thread(timesync.measure, ex.client.rest_api)
    log.info("время: офсет %+d мс", off)
    await asyncio.to_thread(ex.ensure_position_mode, False)  # ONE-WAY

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
