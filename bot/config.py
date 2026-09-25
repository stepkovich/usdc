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
    """Файл .env обязателен только локально; в Docker переменные приходят
    через env_file compose прямо в окружение — файла может не быть.
    Инлайн-комментарии после значения («KEY=val  # пояснение») отрезаются:
    без этого Decimal(os.environ[...]) падает на ConversionSyntax."""
    if not path.exists():
        return
    import re
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            v = re.split(r"\s+#", v.strip(), maxsplit=1)[0].strip()
            os.environ.setdefault(k.strip(), v)


class BotConfig(BaseModel):
    mode: Mode = Mode.DEMO
    api_key: str = ""
    api_secret: str = ""

    # стратегия (замороженные параметры бэктеста, Decimal)
    target_pct: Decimal = Decimal("0.005")     # цель +0.5%
    stop_atr_mult: Decimal = Decimal("12")     # стоп = 12 x ATR(480m) как доля цены
    atr_window: int = 480                      # окно ATR, минутных баров (8ч)
    donchian_bars: int = 480                   # окно Дончиана, минутных баров (8ч)
    wait_bars: int = 60                        # сколько минут ждём отката к уровню
    cancel_ratio: Decimal = Decimal("0.5")     # отмена заявки: close < level*(1-0.5*atr)
    cool_bars: int = 120                       # пауза после выхода (минут)
    warmup_bars: int = 1000                    # минут истории на старте

    # деньги: рейт-нормированный сайзинг (идея владельца)
    risk_pct: Decimal = Decimal("0.0015")      # риск на один стоп, доля баланса (0.15%)
    max_notional: Decimal = Decimal("1500")    # потолок нотионала позиции
    notional_buffer: Decimal = Decimal("1.1")  # пол = minNotional монеты x буфер
    daily_loss_pct: Decimal = Decimal("0.02")  # дневной лимит убытка НА НАПРАВЛЕНИЕ
    total_daily_loss_pct: Decimal = Decimal("0.03")  # общий дневной кап на ОБЕ стороны
    
    # ---- ПАКЕТ A: Риск-контур (DD-guard, consecutive stops) ----
    dd_guard_pct: Decimal = Decimal("0.20")    # портфельный MaxDD 20% -> полная остановка входов
    max_consecutive_stops: int = 3             # стопов подряд -> кулдаун
    consecutive_cooldown_bars: int = 60        # баров кулдауна после серии стопов
    
    leverage: int = 20                         # запас до ликвидации > 2x худшего стопа
    dry_run: bool = False                      # True: сигналы только в журнал
    rearm: bool = False                        # перестановка заявки на новый экстремум
    smc_filter: bool = False                   # SMC-фильтр: не входить против структуры 15м
    tf_min: int = 1                            # зернистость торговых свечей (1=минутки)
    regime_gate: bool = False                  # гейт по режиму BTC(30д): лонги в быке, шорты в медведе
    
    # ---- ПАКЕТ B: Maker-исполнение ----
    maker_entry: bool = True                   # вход post-only GTX с re-peg/таймаутом
    maker_entry_repeg_bps: int = 2             # смещение цены при реквоте (bps)
    maker_entry_max_repeg: int = 5             # макс. попыток реквота
    maker_entry_ttl_sec: int = 180             # TTL лимитки входа (сек)
    maker_stop: bool = True                    # стоп: сначала лимитка, потом маркет
    maker_stop_trigger_bps: int = 2            # за сколько bps до стопа ставим лимитку
    maker_stop_fallback_bps: int = 2           # если лимитка не исполнилась за bps -> маркет
    
    # ---- ПАКЕТ C: Фильтры качества входа ----
    funding_guard_min: int = 5                 # не входить за N минут до/после фандинга
    adx_filter: bool = True                    # ADX-фильтр: не входить в CHOP (ADX<20)
    adx_window: int = 14                       # окно ADX (минутных баров)
    adx_threshold: Decimal = Decimal("20")     # порог ADX для тренда
    vpvr_filter: bool = False                  # VPVR фильтр (требует тиковых данных)

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
            daily_loss_pct=Decimal(os.environ.get("BOT_DAILY_LOSS_PCT", "0.02")),
            total_daily_loss_pct=Decimal(os.environ.get("BOT_TOTAL_DAILY_LOSS_PCT", "0.03")),
            # ПАКЕТ A: Риск-контур
            dd_guard_pct=Decimal(os.environ.get("BOT_DD_GUARD_PCT", "0.20")),
            max_consecutive_stops=int(os.environ.get("BOT_MAX_CONSECUTIVE_STOPS", "3")),
            consecutive_cooldown_bars=int(os.environ.get("BOT_CONSECUTIVE_COOLDOWN_BARS", "60")),
            leverage=int(os.environ.get("BOT_LEVERAGE", "20")),
            dry_run=os.environ.get("BOT_DRY_RUN", "1") == "1",
            rearm=os.environ.get("BOT_REARM", "0") == "1",
            smc_filter=os.environ.get("BOT_SMC_FILTER", "0") == "1",
            tf_min=int(os.environ.get("BOT_TF_MIN", "1")),
            donchian_bars=int(os.environ.get("BOT_DONCHIAN_BARS", "480")),
            regime_gate=os.environ.get("BOT_REGIME_GATE", "0") == "1",
            # ПАКЕТ B: Maker-исполнение
            maker_entry=os.environ.get("BOT_MAKER_ENTRY", "1") == "1",
            maker_entry_repeg_bps=int(os.environ.get("BOT_MAKER_ENTRY_REPEG_BPS", "2")),
            maker_entry_max_repeg=int(os.environ.get("BOT_MAKER_ENTRY_MAX_REPEG", "5")),
            maker_entry_ttl_sec=int(os.environ.get("BOT_MAKER_ENTRY_TTL_SEC", "180")),
            maker_stop=os.environ.get("BOT_MAKER_STOP", "1") == "1",
            maker_stop_trigger_bps=int(os.environ.get("BOT_MAKER_STOP_TRIGGER_BPS", "2")),
            maker_stop_fallback_bps=int(os.environ.get("BOT_MAKER_STOP_FALLBACK_BPS", "2")),
            # ПАКЕТ C: Фильтры
            funding_guard_min=int(os.environ.get("BOT_FUNDING_GUARD_MIN", "5")),
            adx_filter=os.environ.get("BOT_ADX_FILTER", "1") == "1",
            adx_window=int(os.environ.get("BOT_ADX_WINDOW", "14")),
            adx_threshold=Decimal(os.environ.get("BOT_ADX_THRESHOLD", "20")),
            vpvr_filter=os.environ.get("BOT_VPVR_FILTER", "0") == "1",
        )
        if os.environ.get("BOT_SYMBOLS"):
            cfg.symbols = [s.strip().upper()
                           for s in os.environ["BOT_SYMBOLS"].split(",") if s.strip()]
        if not cfg.api_key or not cfg.api_secret:
            raise ValueError("API_KEY/API_SECRET не заданы (.env)")
        return cfg
