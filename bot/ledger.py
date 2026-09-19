"""Журнал (SQLite): события, заявки, исполнения, сделки с двойным учётом.
1) pnl_exchange — как отчитывается биржа (на демо есть комиссии);
2) pnl_assumed — пересчёт «как будто комиссии нет» (ваш тариф):
   мейкер-исполнения бесплатны, стопы по рынку — тейкер + слиппедж."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from decimal import Decimal
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  ts REAL, kind TEXT, symbol TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS fills(
  ts REAL, symbol TEXT, role TEXT, order_id TEXT, client_id TEXT,
  side TEXT, price TEXT, qty TEXT, commission TEXT, realized_pnl TEXT);
CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT,
  entry_ts REAL, exit_ts REAL, entry_px TEXT, exit_px TEXT, qty TEXT,
  exit_kind TEXT, fees_exchange TEXT, fees_assumed TEXT,
  pnl_exchange TEXT, pnl_assumed TEXT);
"""


class Ledger:
    def __init__(self, path: Path, env: str = "demo"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.env = env                    # DEMO/MAINNET: флаг в каждой записи
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        for col in ("env TEXT", "c_asset TEXT", "z_qty TEXT"):
            try:                          # миграция старых баз
                self._db.execute(f"ALTER TABLE fills ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        self._db.commit()
        self._lock = threading.Lock()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._db.execute(sql, params)
            self._db.commit()

    def event(self, kind: str, symbol: str, payload: dict) -> None:
        payload = {"env": self.env, **payload}
        self._exec("INSERT INTO events(ts,kind,symbol,payload) VALUES(?,?,?,?)",
                   (time.time(), kind, symbol, json.dumps(payload, default=str)))

    def fill(self, symbol: str, role: str, o: dict) -> None:
        """o — словарь ORDER_TRADE_UPDATE['o'] (алиасы биржи)."""
        self._exec(
            "INSERT INTO fills(ts,symbol,role,order_id,client_id,side,price,qty,"
            "commission,realized_pnl,env,c_asset,z_qty) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time() / 1000, symbol, role, str(o.get("i", "")), str(o.get("c", "")),
             str(o.get("S", "")), str(o.get("L", "") or o.get("ap", "")),
             str(o.get("l", "")), str(o.get("n", "")), str(o.get("rp", "")), self.env,
             str(o.get("N", "")), str(o.get("z", ""))))

    def trade_closed(self, symbol: str, side: str, entry_ts: float, exit_ts: float,
                     entry_px: Decimal, exit_px: Decimal, qty: Decimal,
                     exit_kind: str, fees_exchange: Decimal, fees_assumed: Decimal,
                     pnl_exchange: Decimal) -> None:
        pnl_assumed = pnl_exchange + fees_exchange - fees_assumed
        self._exec(
            "INSERT INTO trades(symbol,side,entry_ts,exit_ts,entry_px,exit_px,qty,"
            "exit_kind,fees_exchange,fees_assumed,pnl_exchange,pnl_assumed) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (symbol, side, entry_ts, exit_ts, str(entry_px), str(exit_px), str(qty),
             exit_kind, str(fees_exchange), str(fees_assumed),
             str(pnl_exchange), str(pnl_assumed)))

    def summary(self) -> dict:
        cur = self._db.execute(
            "SELECT COUNT(*), SUM(pnl_exchange), SUM(pnl_assumed) FROM trades")
        n, px, pa = cur.fetchone()
        return {"trades": n or 0,
                "pnl_exchange": float(px or 0),
                "pnl_assumed": float(pa or 0)}
