"""Стратегия «тренд-ретест» (порт бэктеста, Decimal):
пробой 8-часового Дончиана -> лимитка на откате к уровню (wait_bars),
TP-лимитка на +target%, STOP_MARKET на 12xATR, отмена заявки при провале ретеста."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from bot.feed import SymbolHistory

log = logging.getLogger("strategy")
ATR_MIN = Decimal("0.0003")   # фильтр «мёртвого рынка» из бэктеста


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class PlaceEntry:
    symbol: str
    side: Side
    price: Decimal
    qty: Decimal
    client_id: str
    atr0: Decimal
    level: Decimal


@dataclass
class CancelEntry:
    symbol: str
    order_id: int
    reason: str
    gap_bp: float = 0.0        # мин. расстояние цены до уровня за ожидание, бп


@dataclass
class PlaceTp:
    symbol: str
    side: Side            # сторона ЗАКРЫВАЮЩЕГО ордера
    price: Decimal
    qty: Decimal
    client_id: str


@dataclass
class PlaceStop:
    symbol: str
    side: Side            # сторона закрывающего ордера
    stop_price: Decimal
    client_id: str


@dataclass
class CancelExit:
    symbol: str
    kind: str             # 'tp' | 'stop'
    reason: str


Action = PlaceEntry | CancelEntry | PlaceTp | PlaceStop | CancelExit


@dataclass
class SymState:
    symbol: str
    bars_seen: int = 0
    cooldown_until: int = -1
    pending: dict | None = None       # {order_id, side, level, atr0, deadline, qty, client_id}
    position: dict | None = None      # {side, qty, entry, tp_id, stop_id, tp_price}
    closes: list = field(default_factory=list)


class Strategy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.states: dict[str, SymState] = {}

    def state(self, symbol: str) -> SymState:
        return self.states.setdefault(symbol, SymState(symbol))

    # ---------- сигнал по закрытому бару ----------
    def on_closed_bar(self, h: SymbolHistory, bar) -> list[Action]:
        st = self.state(h.symbol)
        st.bars_seen += 1
        st.closes.append(bar.close)
        acts: list[Action] = []
        cfg = self.cfg

        # 1) позиция есть -> сопровождение выхода не требует баров (стоп/TP стоят на бирже)
        # 2) ожидание отката: проверка отмены заявки
        if st.pending:
            p = st.pending
            # насколько близко цена подходила к уровню (для статистики упущенных входов)
            gap = abs(bar.close - p["level"]) / p["level"]
            p["min_gap"] = min(p.get("min_gap", 1.0), gap)
            if st.bars_seen > p["deadline"]:
                acts.append(CancelEntry(h.symbol, p["order_id"], "timeout",
                                        p.get("min_gap", 1.0) * 10000))
                st.pending = None
            else:
                if p["side"] is Side.LONG and bar.close <= p["level"] * (1 - cfg.cancel_ratio * p["atr0"]):
                    acts.append(CancelEntry(h.symbol, p["order_id"], "retest_failed",
                                            p.get("min_gap", 1.0) * 10000))
                    st.pending = None
                elif p["side"] is Side.SHORT and bar.close >= p["level"] * (1 + cfg.cancel_ratio * p["atr0"]):
                    acts.append(CancelEntry(h.symbol, p["order_id"], "retest_failed",
                                            p.get("min_gap", 1.0) * 10000))
                    st.pending = None
            return acts

        # 3) кулдаун / позиция -> не сигналим
        if st.position or st.bars_seen < st.cooldown_until:
            return acts

        # 4) новый сигнал: пробой Дончиана (окно без последнего бара — shift(1))
        try:
            d_high, d_low = h.donchian()
            atr0 = h.atr_frac
        except ValueError:
            return acts
        if atr0 < ATR_MIN:
            return acts

        if bar.close > d_high:
            side, level = Side.LONG, d_high
        elif bar.close < d_low:
            side, level = Side.SHORT, d_low
        else:
            return acts

        cid = f"scr-E-{h.symbol}-{st.bars_seen}"
        log.info("СИГНАЛ %s %s уровень=%s atr=%s", h.symbol, side.value, level, atr0)
        acts.append(PlaceEntry(h.symbol, side, level, Decimal(0), cid, atr0, level))
        st.pending = {"client_id": cid, "order_id": None, "side": side,
                      "level": level, "atr0": atr0,
                      "deadline": st.bars_seen + cfg.wait_bars, "qty": None}
        return acts

    # ---------- заявка на вход подтверждена биржей ----------
    def entry_placed(self, symbol: str, order_id: int, qty: Decimal) -> None:
        st = self.state(symbol)
        if st.pending:
            st.pending["order_id"] = order_id
            st.pending["qty"] = qty

    # ---------- вход исполнился ----------
    def entry_filled(self, symbol: str, qty: Decimal, avg_price: Decimal,
                     order_id: int) -> list[Action]:
        st = self.state(symbol)
        if not st.pending:
            return []                        # хвост после реконсиляции — игнор
        p = st.pending
        side = p["side"]
        exit_side = Side.SHORT if side is Side.LONG else Side.LONG
        tp_price = (avg_price * (1 + self.cfg.target_pct) if side is Side.LONG
                    else avg_price * (1 - self.cfg.target_pct))
        stop_price = (avg_price * (1 - self.cfg.stop_atr_mult * p["atr0"]) if side is Side.LONG
                      else avg_price * (1 + self.cfg.stop_atr_mult * p["atr0"]))
        st.position = {"side": side, "qty": qty, "entry": avg_price,
                       "tp_id": None, "stop_id": None, "tp_price": tp_price}
        st.pending = None
        cid_tp = f"scr-T-{symbol}-{order_id}"
        cid_stop = f"scr-S-{symbol}-{order_id}"
        log.info("ВХОД %s %s qty=%s avg=%s TP=%s STOP=%s",
                 symbol, side.value, qty, avg_price, tp_price, stop_price)
        return [PlaceTp(symbol, exit_side, tp_price, qty, cid_tp),
                PlaceStop(symbol, exit_side, stop_price, cid_stop)]

    # ---------- выход исполнился ----------
    def exit_filled(self, symbol: str, kind: str) -> list[Action]:
        st = self.state(symbol)
        if not st.position:
            return []
        st.cooldown_until = st.bars_seen + self.cfg.cool_bars
        other = "stop" if kind == "tp" else "tp"
        st.position = None
        log.info("ВЫХОД %s по %s -> кулдаун до бара %s", symbol, kind, st.cooldown_until)
        return [CancelExit(symbol, other, f"{kind}_filled")]

    # ---------- реконсиляция: биржа видит позицию/заявки иначе, чем мы ----------
    def force_flat(self, symbol: str) -> None:
        st = self.state(symbol)
        st.pending = None
        st.position = None
