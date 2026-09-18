"""Конфигурация бота. Режимы и URL — строго по константам репозитория
binance-connector-python (common/src/binance_common/constants.py)."""
from __future__ import annotations

import os
from decimal import Decimal
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from binance_common.constants import (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_DEMO_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL,
)


class Mode(str, Enum):
    DEMO = "demo"        # demo-fapi.binance.com — по умолчанию
    TESTNET = "testnet"  # testnet.binancefuture.com
    MAINNET = "mainnet"  # fapi.binance.com — боевой, использовать осознанно


# URL исполнения по режиму (REST). Демо-URL — из констант репозитория.
EXEC_REST_URL = {
    Mode.DEMO: DERIVATIVES_TRADING_USDS_FUTURES_REST_API_DEMO_URL,
    Mode.TESTNET: DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
    Mode.MAINNET: DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
}
# Рыночные WS-стримы: в репозитории демо-URL стримов для фьючерсов НЕ существует,
# therefore рыночные данные всегда берём с боевых стримов (публичные, ключи не нужны).
MARKET_STREAMS_URL = {
    Mode.DEMO: DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
    Mode.TESTNET: DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
    Mode.MAINNET: DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL,
}
# Рыночный REST (история свечей для тёплого старта) — тоже прод, публичный.
MARKET_REST_URL = DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL


def load_dotenv(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


class BotConfig(BaseModel):
    mode: Mode = Mode.DEMO
    api_key: str = ""
    api_secret: str = ""

    # стратегия (замороженные параметры бэктеста, Decimal)
    target_pct: Decimal = Decimal("0.005")     # цель +0.5%
    stop_atr_mult: Decimal = Decimal("12")     # стоп = 12 x ATR(480m) как доля цены
    atr_window: int = 480                      # 8 часов минутных баров
    wait_bars: int = 60                        # сколько минут ждём отката к уровню
    cancel_ratio: Decimal = Decimal("0.5")     # отмена заявки: close < level*(1-0.5*atr)
    cool_bars: int = 120                       # пауза после выхода (минут)
    warmup_bars: int = 1000                    # минут истории на старте

    # деньги: рейт-нормированный сайзинг (идея владельца)
    risk_pct: Decimal = Decimal("0.0015")      # риск на один стоп, доля баланса (0.15%)
    max_notional: Decimal = Decimal("1500")    # потолок нотионала позиции
    notional_buffer: Decimal = Decimal("1.1")  # пол = minNotional монеты x буфер
    leverage: int = 20                         # запас до ликвидации > 2x худшего стопа
    dry_run: bool = False                      # True: сигналы только в журнал

    # учёт «как будто комиссии нет» (ваш тариф) + реальность демо
    assume_maker_fee: Decimal = Decimal("0")       # тариф: мейкер 0
    assume_taker_fee: Decimal = Decimal("0.0005")  # стопы по рынку
    assume_stop_slip: Decimal = Decimal("0.0005")  # слиппедж стопа для учёта

    # базис демо/прод, выше которого пару не торгуем
    max_basis: Decimal = Decimal("0.001")

    # символы: None -> все USDC-пары демо, доступные по фильтрам
    symbols: list[str] | None = None

    @property
    def exec_rest_url(self) -> str:
        return EXEC_REST_URL[self.mode]

    @property
    def market_streams_url(self) -> str:
        return MARKET_STREAMS_URL[self.mode]

    @classmethod
    def from_env(cls, root: Path) -> "BotConfig":
        load_dotenv(root / ".env")
        cfg = cls(
            mode=Mode(os.environ.get("BOT_MODE", "demo")),
            api_key=os.environ.get("API_KEY", ""),
            api_secret=os.environ.get("API_SECRET", ""),
            target_pct=Decimal(os.environ.get("BOT_TARGET_PCT", "0.005")),
            risk_pct=Decimal(os.environ.get("BOT_RISK_PCT", "0.0015")),
            max_notional=Decimal(os.environ.get("BOT_MAX_NOTIONAL", "1500")),
            notional_buffer=Decimal(os.environ.get("BOT_NOTIONAL_BUFFER", "1.1")),
            leverage=int(os.environ.get("BOT_LEVERAGE", "20")),
            dry_run=os.environ.get("BOT_DRY_RUN", "1") == "1",
        )
        if os.environ.get("BOT_SYMBOLS"):
            cfg.symbols = [s.strip().upper()
                           for s in os.environ["BOT_SYMBOLS"].split(",") if s.strip()]
        if not cfg.api_key or not cfg.api_secret:
            raise ValueError("API_KEY/API_SECRET не заданы (.env)")
        return cfg
