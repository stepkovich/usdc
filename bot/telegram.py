"""Telegram-уведомления: сделки, ошибки, дневные итоги.
Модуль не блокирует торговлю — отправка в фоне, ошибки глотаются.
Настройка через .env: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID"""
import asyncio
import logging

import aiohttp

log = logging.getLogger("telegram")

TOKEN = None
CHAT_ID = None
ENV = "demo"


def init(token: str, chat_id: str, env: str = "demo") -> None:
    global TOKEN, CHAT_ID, ENV
    TOKEN = token
    CHAT_ID = chat_id
    ENV = env.upper()
    log.info("telegram: токен и chat_id установлены, среда %s", ENV)


async def send(text: str) -> None:
    """Отправить сообщение. Фоновая задача — не блокирует торговлю."""
    if not TOKEN or not CHAT_ID:
        log.warning("telegram: TOKEN или CHAT_ID не установлены — пропускаю")
        return
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                res = await r.json()
                if not res.get("ok"):
                    log.warning("telegram send failed: %s", res)
                else:
                    log.info("telegram: отправлено (%d символов)", len(text))
    except Exception as e:
        log.warning("telegram send error: %s", e)


def _prefix() -> str:
    return f"[{ENV}] "


def fire(text: str) -> None:
    """Fire-and-forget: вызвать из sync/async контекста, не ждёт ответа."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(send(text))
        else:
            loop.run_until_complete(send(text))
    except RuntimeError:
        pass


# --- Форматированные уведомления ---

def notify_entry(symbol: str, side: str, size: float, price: str) -> None:
    emoji = "🟢" if side == "LONG" else "🔴"
    fire(f"{_prefix()}{emoji} <b>ВХОД {symbol} {side}</b>\n"
         f"Размер: {size:.2f} USDC @ {price}")


def notify_tp(symbol: str, pnl: float, dur_h: float) -> None:
    fire(f"{_prefix()}✅ <b>TP {symbol}</b> +{pnl:.2f} USDC\n"
         f"Время в позиции: {dur_h:.1f} ч")


def notify_stop(symbol: str, pnl: float, dur_h: float) -> None:
    fire(f"{_prefix()}🔴 <b>СТОП {symbol}</b> {pnl:.2f} USDC\n"
         f"Время в позиции: {dur_h:.1f} ч")


def notify_daily(n: int, wins: int, pnl: float) -> None:
    fire(f"{_prefix()}📊 <b>Итог дня</b>: {n} сделок, {wins} в плюс, "
         f"PnL {pnl:+.2f} USDC")


def notify_error(text: str) -> None:
    fire(f"{_prefix()}⚠️ {text}")
