"""Проверка ордерной обвязки на демо БЕЗ рыночного риска:
режим позиции -> плечо -> дальняя LIMIT заявка (не исполнится) ->
NEW/CANCELED события в user-data стриме -> дальний условный STOP -> отмена.
Запуск: PYTHONPATH=.:pylibs python3 -m bot.plumbing_test"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from decimal import Decimal
from pathlib import Path

from binance_common.configuration import ConfigurationRestAPI, ConfigurationWebSocketStreams
from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.config import BotConfig                       # noqa: E402
from bot.executor import Executor                      # noqa: E402
from bot.ledger import Ledger                          # noqa: E402
from bot.markets import parse_filters                  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("plumb")

SYM = "SOLUSDC"


async def main() -> None:
    cfg = BotConfig.from_env(ROOT)
    probe = DerivativesTradingUsdsFutures(config_rest_api=ConfigurationRestAPI(
        api_key=cfg.api_key, api_secret=cfg.api_secret, base_path=cfg.exec_rest_url))
    info = probe.rest_api.exchange_information().data().model_dump(by_alias=True)
    sym_info = next(s for s in info["symbols"] if s["symbol"] == SYM)
    f = parse_filters(SYM, sym_info["filters"])
    market_rest = DerivativesTradingUsdsFutures(config_rest_api=ConfigurationRestAPI(
        api_key=cfg.api_key, api_secret=cfg.api_secret, base_path=cfg.exec_rest_url))
    d = market_rest.rest_api.symbol_price_ticker(symbol=SYM).data()
    inst = getattr(d, "actual_instance", None) or d
    last = float(inst.price)
    log.info("режим=%s %s last=%.4f tick=%s step=%s minNotional=%s",
             cfg.mode.value, SYM, last, f.tick_size, f.step_size, f.min_notional)

    ex = Executor(cfg, {SYM: f})

    # 1) режим позиции (попытка установить хедж; при открытых позициях останется текущий)
    hedge = ex.ensure_position_mode(want_hedge=True)
    log.info("итоговый режим позиции: %s", "HEDGE" if hedge else "ONE-WAY")

    # 2) плечо: проверить и установить
    ex.verify_and_set_leverage([SYM])
    log.info("плечо %s: %sx", SYM, ex.leverage_set.get(SYM))

    # 3) user-data стрим (сырой WS: /ws/<listenKey> на демо-хосте)
    events: list[dict] = []

    def new_key():
        return probe.rest_api.start_user_data_stream().data().listen_key

    def keepalive():
        probe.rest_api.keepalive_user_data_stream()

    from bot.user_stream import RawUserStream
    raw = RawUserStream(new_key, keepalive,
                        lambda d: events.append(d))
    task = asyncio.create_task(raw.run())
    await asyncio.sleep(3)
    log.info("user-data стрим запущен, ждём события")

    def seen(status: str, cid: str) -> bool:
        return any(isinstance(e.get("o"), dict) and e["o"].get("X") == status
                   and str(e["o"].get("c", "")) == cid for e in events)

    async def wait_evt(status: str, cid: str, timeout_s: float) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if seen(status, cid):
                log.info("событие получено: X=%s cid=%s ", status, cid)
                return True
            await asyncio.sleep(0.5)
        log.error("событие НЕ получено: X=%s cid=%s", status, cid)
        return False

    # очистка наших хвостов прошлых запусков
    snap = ex.snapshot()
    for o in snap["orders"]:
        if str(o.get("clientOrderId", "")).startswith("tst-"):
            log.info("чистим хвост: заявка %s (%s)", o.get("orderId"), o.get("clientOrderId"))
            ex.cancel_order(SYM, int(o["orderId"]))
    for a in snap["algo"]:
        if str(a.get("clientAlgoId", a.get("clientOrderId", ""))).startswith("tst-"):
            ex.cancel_algo_order(SYM, int(a.get("algoId") or a.get("orderId")))

    stamp = int(time.time())
    cid_e = f"tst-E-{stamp}"
    cid_s = f"tst-S-{stamp}"
    try:
        # 4) дальняя LIMIT заявка (на 10% ниже рынка — не исполнится)
        entry_px = f.round_price(Decimal(str(last)) * Decimal("0.90"))
        qty = f.qty_for_notional(Decimal("50"), entry_px)   # тестовый объём
        log.info("ставим дальнюю заявку BUY %s @%s (рынок %.4f)", qty, entry_px, last)
        oid = ex.place_entry_limit(SYM, "BUY", entry_px, qty, cid_e)
        ok_new = await wait_evt("NEW", cid_e, 20)
        log.info("отменяем заявку id=%s", oid)
        ex.cancel_order(SYM, oid)
        ok_cxl = await wait_evt("CANCELED", cid_e, 20)

        # 5) дальний условный STOP_MARKET с quantity (триггер на 10% ниже рынка)
        stop_px = f.round_price(Decimal(str(last)) * Decimal("0.90"))
        log.info("ставим дальний STOP_MARKET SELL %s @%s (trigger)", qty, stop_px)
        algo = ex.place_stop_market(SYM, "SELL", stop_px, cid_s, qty=qty)
        log.info("условный ордер размещён id=%s; проверяем в списке algo-заявок", algo)
        snap = ex.snapshot()
        algo_ids = [str(a.get("algoId") or a.get("orderId")) for a in snap["algo"]
                    if a.get("symbol") == SYM]
        log.info("algo-заявки на %s: %s", SYM, algo_ids)
        if algo:
            ex.cancel_algo_order(SYM, algo)
        log.info("ИТОГ plumbing: NEW=%s CANCELED=%s algo-placed=%s",
                 ok_new, ok_cxl, bool(algo))
    finally:
        raw.stop()
    log.info("Готово. Событий user-data: %d", len(events))
    for e in events[:6]:
        o = e.get("o") or {}
        log.info("  событие: e=%s X=%s c=%s", e.get("e"),
                 o.get("X") if isinstance(o, dict) else o,
                 o.get("c") if isinstance(o, dict) else "")
    log.info("Готово. Событий user-data: %d", len(events))
    for e in events[:6]: log.info("RAW: %s", str(e)[:200])
    for e in events[:8]:
        o = e.get("o") or {}
        log.info("  событие: e=%s X=%s c=%s", e.get("e"),
                 o.get("X") if isinstance(o, dict) else o,
                 o.get("c") if isinstance(o, dict) else "")


async def wait_for(events: list, status: str, cid: str, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for e in events:
            o = e.get("o") or {}
            if isinstance(o, dict) and o.get("X") == status and str(o.get("c", "")) == cid:
                log.info("событие получено: X=%s cid=%s", status, cid)
                return True
        await asyncio.sleep(0.5)
    log.error("событие НЕ получено: X=%s cid=%s (стрим не доставляет?)", status, cid)
    return False


if __name__ == "__main__":
    asyncio.run(main())
