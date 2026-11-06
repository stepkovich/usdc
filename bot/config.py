"""Конфигурация единой системы.

  Буря (ML-5ч): все живые USDT-перпетуалы (~515), сигнал каждые 30
  минут; гейт 0.65 берётся из паспорта модели, холд 10 баров, вход/выход
  мейкером. Сайзинг владельца (v3, 02.11): номинал = баланс × 50% пула ×
  плечо 3 × 10% (= 15% баланса), пол = мин-ордер монеты × 1.1; слоты 10
  (полный стол = 50% баланса маржей); дневной кап 4% баланса.
  Двигатель «Стакан» (lob) выключен в бою (LOB_ENABLED=0).

Принципы: биржа — источник правды; Decimal; тик/шаг с биржи; дневной кап.
"""
from __future__ import annotations

import os
from decimal import Decimal
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from binance_common.constants import (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_DEMO_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
)

ROOT = Path(__file__).resolve().parent.parent


class Mode(str, Enum):
    DEMO = "demo"
    TESTNET = "testnet"
    MAINNET = "mainnet"


EXEC_REST_URL = {
    Mode.DEMO: DERIVATIVES_TRADING_USDS_FUTURES_REST_API_DEMO_URL,
    Mode.TESTNET: DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
    Mode.MAINNET: DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
}


def load_dotenv(path: Path) -> None:
    """Инлайн-комментарии отрезаются: KEY=val  # пояснение."""
    if not path.exists():
        return
    import re
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            v = re.split(r"\s+#", v.strip(), maxsplit=1)[0].strip()
            os.environ.setdefault(k.strip(), v)


class ML5HConfig(BaseModel):
    enabled: bool = True
    symbols: list[str] = []
    model_path: Path = ROOT / "data" / "models" / "ml5h.txt"   # на томе данных: обновление без пересборки
    meta_path: Path = ROOT / "data" / "models" / "ml5h_meta.json"
    gate: float = 0.55               # заморожено на этапе исследования
    hold_bars: int = 10              # 10 x 30м = 5 часов
    bar_minutes: int = 30
    warmup_bars: int = 1000          # 30м-баров истории на старте (1500 = двойной вес запроса -> бан)
    risk_pct: Decimal = Decimal("0.0015")          # (не используется — см. notional_pct)
    buffer_pct: Decimal = Decimal("0.02")     # (не используется — см. notional_pct)
    notional_cap_pct: Decimal = Decimal("0.02")     # (не используется — см. notional_pct)
    universe_top: int = 200                        # вселенная владельца 04.11
    universe_min_age_days: int = 180               # монеты старше 180 дней
    universe_exclude: list[str] = ["BTCUSDT", "ETHUSDT"]
    notional_pct: Decimal = Decimal("0.10")   # доля от ПЛЕЧЕВОЙ ЁМКОСТИ пула (v3 02.11)
    margin_pool_pct: Decimal = Decimal("0.5") # пул = половина баланса
    min_notional_buffer: Decimal = Decimal("1.1")  # пол = мин-ордер монеты × 1.1
    max_slots: int = 10             # v3: 10 × маржа 5% баланса = ровно 50% пула
    entry_ttl_s: int = 60
    exit_ttl_s: int = 90
    leverage: int = 3
    emergency_stop_pct: Decimal = Decimal("0.15")  # аварийный выход (MAE-исследование 28.09)
    risk_usdc: Decimal = Decimal("0")              # риск НА СДЕЛКУ в деньгах (0 = процентный)
    day_cap_usdc: Decimal = Decimal("0")           # дневной кап в деньгах (0 = процентный)
    quiet_filter: bool = True                      # фильтр «тихая треть» (рентген 12.10)
    meta_gate: bool = False                        # мета-подтверждение (12.10, идея владельца)
    taker_fallback: bool = False                   # демо: мейкер не взялся за TTL -> вход по рынку
    meta_model_path: Path = ROOT / "data" / "models" / "meta.txt"


class LobConfig(BaseModel):
    enabled: bool = True
    symbols: list[str] = []   # пусто = ВСЕ живые USDC-перпетуалы (авто с биржи)
    model_path: Path = ROOT / "models" / "lob.txt"
    gate: float = 0.62
    hold_s: int = 300
    row_ms: int = 500                # частота пачек биржи — пишем каждую
    hot_hours: int = 6
    archive_days: int = 2
    first_train_h: int = 6           # часов данных до первой тренировки
    retrain_at_utc: tuple[int, int] = (0, 15)
    risk_pct: Decimal = Decimal("0.0015")
    buffer_spread_mult: int = 3      # оценочный убыток = 3 x спред
    min_buffer_bp: int = 3
    max_slots: int = 3
    entry_ttl_s: int = 60
    exit_ttl_s: int = 45
    leverage: int = 3
    emergency_stop_pct: Decimal = Decimal("0.15")  # аварийный выход (флеш-краш защита)
    risk_usdc: Decimal = Decimal("0")              # риск НА СДЕЛКУ в деньгах (0 = процентный)
    day_cap_usdc: Decimal = Decimal("0")           # дневной кап в деньгах (0 = процентный)
    quiet_filter: bool = True                      # фильтр «тихая треть» (рентген 12.10)
    meta_gate: bool = False                        # мета-подтверждение (12.10, идея владельца)
    taker_fallback: bool = False                   # демо: мейкер не взялся за TTL -> вход по рынку
    meta_model_path: Path = ROOT / "data" / "models" / "meta.txt"


class Config(BaseModel):
    mode: Mode = Mode.DEMO
    api_key: str = ""
    api_secret: str = ""
    telegram_token: str = ""
    telegram_chat_id: str = ""
    data_dir: Path = ROOT / "data"
    dry_run: bool = False

    ml5h: ML5HConfig = ML5HConfig()
    lob: LobConfig = LobConfig()

    daily_cap_pct: Decimal = Decimal("0.04")   # дневной кап 4% баланса (02.11: ~2 стопа/день)

    @property
    def exec_rest_url(self) -> str:
        return EXEC_REST_URL[self.mode]

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Config":
        load_dotenv((root or ROOT) / ".env")
        cfg = cls(
            mode=Mode(os.environ.get("BOT_MODE", "demo")),
            api_key=os.environ.get("API_KEY", ""),
            api_secret=os.environ.get("API_SECRET", ""),
            telegram_token=os.environ.get("TELEGRAM_TOKEN", ""),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
            dry_run=os.environ.get("BOT_DRY_RUN", "0") == "1",
        )
        # денежный сайзинг (малый профиль): риск и дневной кап в деньгах,
        # гибрид с процентным потолком
        ml5h_risk = os.environ.get("ML5H_RISK_USDC", "")
        cfg.ml5h.risk_usdc = Decimal(ml5h_risk) if ml5h_risk else Decimal("0")
        ml5h_day = os.environ.get("ML5H_DAY_CAP_USDC", "")
        cfg.ml5h.day_cap_usdc = Decimal(ml5h_day) if ml5h_day else Decimal("0")
        lob_risk = os.environ.get("LOB_RISK_USDC", "")
        cfg.lob.risk_usdc = Decimal(lob_risk) if lob_risk else Decimal("0")
        lob_day = os.environ.get("LOB_DAY_CAP_USDC", "")
        cfg.lob.day_cap_usdc = Decimal(lob_day) if lob_day else Decimal("0")
        cfg.ml5h.enabled = os.environ.get("ML5H_ENABLED", "1") == "1"
        cfg.ml5h.quiet_filter = os.environ.get("ML5H_QUIET_FILTER", "1") == "1"
        if os.environ.get("ML5H_META"):
            cfg.ml5h.meta_gate = os.environ["ML5H_META"] == "1"
        if os.environ.get("ML5H_TAKER_FALLBACK"):
            cfg.ml5h.taker_fallback = os.environ["ML5H_TAKER_FALLBACK"] == "1"
        if os.environ.get("ML5H_MAX_SLOTS"):
            cfg.ml5h.max_slots = int(os.environ["ML5H_MAX_SLOTS"])
        if os.environ.get("ML5H_UNIVERSE_TOP"):
            cfg.ml5h.universe_top = int(os.environ["ML5H_UNIVERSE_TOP"])
        if os.environ.get("ML5H_UNIVERSE_AGE"):
            cfg.ml5h.universe_min_age_days = int(
                os.environ["ML5H_UNIVERSE_AGE"])
        cfg.lob.enabled = os.environ.get("LOB_ENABLED", "1") == "1"
        if os.environ.get("ML5H_SYMBOLS"):
            cfg.ml5h.symbols = [s.strip().upper() for s in
                                os.environ["ML5H_SYMBOLS"].split(",") if s.strip()]
        if os.environ.get("LOB_SYMBOLS"):
            cfg.lob.symbols = [s.strip().upper() for s in
                               os.environ["LOB_SYMBOLS"].split(",") if s.strip()]
        if not cfg.api_key or not cfg.api_secret:
            raise ValueError("API_KEY/API_SECRET не заданы (.env)")
        return cfg
