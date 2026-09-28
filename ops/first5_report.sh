#!/bin/bash
# ОТЧЁТ о закрытии первых 5 позиций (23:59-00:05 местного).
# Собирает всё с сервера, пишет человекочитаемый отчёт в файл + Telegram.
OUT=/home/iek/lob_archive/first5_report.txt
{
echo "=== ЗАКРЫТИЯ В ЛОГАХ ==="
ssh -o ConnectTimeout=20 root@159.65.133.26 'docker logs unified --since 6h 2>&1 | grep -E "ЗАКРЫТО|ВХОД" | grep -E "AIOT|ASTER|4USDT|APTUSDT|1000CAT" | tail -20'
echo ""
echo "=== БИРЖА: PnL и комиссии по 5 позициям ==="
ssh -o ConnectTimeout=20 root@159.65.133.26 'docker exec unified python3 -c "
import sys, os; sys.path.insert(0, \"/app\")
from binance_common.configuration import ConfigurationRestAPI
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import DerivativesTradingUsdsFutures
import datetime
c = DerivativesTradingUsdsFutures(config_rest_api=ConfigurationRestAPI(
    api_key=os.environ[\"API_KEY\"], api_secret=os.environ[\"API_SECRET\"],
    base_path=\"https://demo-fapi.binance.com\", timeout=5000, retries=2))
day0 = int(datetime.datetime.now(datetime.UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()*1000)
inc = c.rest_api.get_income_history(start_time=day0, limit=1000).data()
rows = getattr(inc, \"root\", None) or inc
pnl, comm, n = 0.0, 0.0, 0
syms = (\"AIOTUSDT\",\"ASTERUSDT\",\"4USDT\",\"APTUSDT\",\"1000CATUSDT\")
for r in rows:
    d = r.model_dump(by_alias=True) if hasattr(r, \"model_dump\") else r
    if d.get(\"symbol\") in syms:
        v = float(d.get(\"income\", 0) or 0)
        if d.get(\"incomeType\") == \"REALIZED_PNL\": pnl += v; n += 1
        elif d.get(\"incomeType\") == \"COMMISSION\": comm += v
print(f\"сделок-исполнений: {n} | PnL: {pnl:+.4f} | комиссии: {comm:+.4f} | чистыми: {pnl+comm:+.4f}\")
bal = c.rest_api.futures_account_balance_v3().data()
br = getattr(bal, \"root\", None) or bal
for a in br:
    d = a.model_dump(by_alias=True) if hasattr(a, \"model_dump\") else a
    if d.get(\"asset\") == \"USDT\": print(\"баланс USDT:\", d.get(\"balance\"))
" 2>&1 | grep -v INFO'
echo ""
echo "=== ОШИБКИ ЗА 6 ЧАСОВ ==="
ssh -o ConnectTimeout=20 root@159.65.133.26 'docker logs unified --since 6h 2>&1 | grep -c "Traceback"; docker logs unified --since 6h 2>&1 | grep -A2 "Traceback" | head -6'
echo ""
echo "=== ОТКРЫТЫЕ ПОЗИЦИИ (хвост дня) ==="
ssh -o ConnectTimeout=20 root@159.65.133.26 'docker logs unified --since 6h 2>&1 | grep -cE "ВХОД"; docker logs unified --since 6h 2>&1 | grep -E "СИГНАЛ.*p=0\.[6-9]" | tail -3'
} > "$OUT" 2>&1
source /home/iek/.tg_creds 2>/dev/null
SUMMARY=$(grep -E "чистыми|баланс USDT|Traceback" "$OUT" | head -3 | tr '\n' ' ')
curl -s -o /dev/null -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
  -d chat_id="${TELEGRAM_CHAT_ID}" --data-urlencode "text=📊 Отчёт о первых 5 позициях собран в чат-файл. Кратко: ${SUMMARY:-см. файл first5_report.txt}"
echo "$(date -u +%FT%TZ) first5 report done" >> /home/iek/lob_archive/pull.log
