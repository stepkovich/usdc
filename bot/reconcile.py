"""Сверка журнала с биржей (источник правды). Запуск:
PYTHONPATH=.:pylibs python3 -m bot.reconcile [часов_назад=4]"""
import sys, time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from binance_common.configuration import ConfigurationRestAPI            # noqa: E402
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (  # noqa: E402
    DerivativesTradingUsdsFutures,
)
from bot.config import BotConfig                                         # noqa: E402
from bot.executor import Executor, unwrap                                # noqa: E402

def main() -> None:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    cfg = BotConfig.from_env(Path(__file__).resolve().parent.parent)
    ex = Executor(cfg, {})
    start_ms = int(time.time() * 1000) - int(hours * 3600 * 1000)
    rows = unwrap(ex.client.rest_api.get_income_history(
        income_type="REALIZED_PNL", start_time=start_ms, limit=1000).data())
    rows = getattr(rows, "root", None) or rows
    comm = unwrap(ex.client.rest_api.get_income_history(
        income_type="COMMISSION", start_time=start_ms, limit=1000).data())
    comm = getattr(comm, "root", None) or comm
    agg = {}
    for r in list(rows) + list(comm):
        d = r.model_dump(by_alias=True)
        v = agg.setdefault(d.get("symbol", "?"), {"pnl": Decimal(0), "comm": Decimal(0)})
        if d.get("incomeType") == "REALIZED_PNL":
            v["pnl"] += Decimal(str(d.get("income", "0")))
        elif d.get("incomeType") == "COMMISSION":
            v["comm"] += Decimal(str(d.get("income", "0")))
    tp = tc = Decimal(0)
    print(f"{'символ':<16}{'PnL':>12}{'комиссии':>12}")
    for s in sorted(agg):
        p, c = agg[s]["pnl"], agg[s]["comm"]
        if p == 0 and c == 0:
            continue
        tp += p; tc += c
        print(f"{s:<16}{p:+12.6f}{c:+12.6f}")
    print(f"{'ИТОГО':<16}{tp:+12.6f}{tc:+12.6f}")
    print(f"\nЧистыми (PnL+комиссии): {tp + tc:+.6f} USDC")
    print("PnL вашего тарифа = ИТОГО PnL + |мейкерские комиссии| (см. журнал, роли E/T)")

if __name__ == "__main__":
    main()
