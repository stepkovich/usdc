"""Принудительная синхронизация времени с биржей (защита от -1021
«Timestamp outside recvWindow»).

SDK подписывает запросы локальными часами: binance_common.utils.get_timestamp()
= time.time()*1000, синхронизации и хука у него нет. Модуль меряет офсет
до сервера биржи (check_server_time — публичный эндпоинт) с компенсацией
полуреттинга: offset = serverTime − (t0+t1)/2, и подменяет get_timestamp
в SDK на «локальные часы + офсет».

Подмена функции SDK — осознанное отступление («единственный стрим, где
отступаем» №2): у SDK нет конфиг-хука на время. Патч точечный и задокументирован;
чистится заменой одной строки, если SDK добавит поддержку.

Порядок: install() ДО первого подписанного запроса, measure() в setup и
раз в 15 мин в watchdog; при -1021 в ретраях ордеров — внеплановый measure().
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger("timesync")

OFFSET_MS = 0          # сервер − локальные часы (компенсированный полуреттинг)
LAST_RTT_MS = 0.0      # последний round-trip замера — для контроля качества


def install() -> None:
    """Подменяет binance_common.utils.get_timestamp на «локальное время + офсет».
    Имя разрешается в момент вызова внутри send_request, поэтому патч
    действует на все последующие подписанные запросы без пересоздания клиента."""
    import binance_common.utils as _u

    def _synced_timestamp() -> int:
        return int(time.time() * 1000) + OFFSET_MS

    _u.get_timestamp = _synced_timestamp


def measure(rest_api) -> int:
    """Замеряет офсет к серверу биржи и обновляет глобальный OFFSET_MS."""
    global OFFSET_MS, LAST_RTT_MS
    t0 = time.time() * 1000
    d = rest_api.check_server_time().data()
    t1 = time.time() * 1000
    server = d.model_dump(by_alias=True).get("serverTime") \
        if hasattr(d, "model_dump") else getattr(d, "server_time", None)
    if server is None:
        raise RuntimeError("check_server_time: в ответе нет serverTime")
    LAST_RTT_MS = t1 - t0
    OFFSET_MS = int(server - (t0 + t1) / 2)
    return OFFSET_MS
