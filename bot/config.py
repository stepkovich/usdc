"""Конфигурация единой системы: два двигателя на одном демо-счёте.

  ДВИГАТЕЛЬ ML-5Ч  — топ-20 USDT-монет: каждые 30 минут модель отвечает
  «вырастет ли через 5 часов?»; порог уверенности 0.55 (заморожен на
  этапе исследования), холд 10 баров, вход/выход мейкером.
  ДВИГАТЕЛЬ СТАКАН — топ-10 USDC-монет: строка стакана каждые 500 мс,
  «вырастет ли через 5 минут?», порог 0.62, холд 300 секунд.

Принципы: биржа — источник правды; Decimal; сайзинг от потери;
тик/шаг с биржи; дневной кап; данные с мейннета, исполнение — демо.
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
    meta_path: Path = ROOT / "models" / "ml5h_meta.json"
    gate: float = 0.55               # заморожено на этапе исследования
    hold_bars: int = 10              # 10 x 30м = 5 часов
    bar_minutes: int = 30
    warmup_bars: int = 1000          # 30м-баров истории на старте
    risk_pct: Decimal = Decimal("0.0015")
    buffer_pct: Decimal = Decimal("0.02")     # оценочный убыток 5ч позиции
    notional_cap_pct: Decimal = Decimal("0.02")
    max_slots: int = 6
    entry_ttl_s: int = 60
    exit_ttl_s: int = 90
    leverage: int = 3


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

    daily_cap_pct: Decimal = Decimal("0.005")   # на двигатель, доля баланса

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
        cfg.ml5h.enabled = os.environ.get("ML5H_ENABLED", "1") == "1"
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
