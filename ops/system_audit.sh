#!/bin/bash
# АУДИТ каждые 4 часа: unified-real + amount_bot на этом же сервере + диск.
set +e
exec 9>/tmp/system_audit.lock
flock -n 9 || exit 0
source /root/usdc/.tg_env 2>/dev/null
send() { curl -s -o /dev/null -m 20 -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" -d chat_id="${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1"; }
FIX=""

# 1) РЕАЛ (локальный контейнер)
REAL_ST=$(docker inspect -f {{.State.Status}} unified-real 2>/dev/null); REAL_ST=${REAL_ST:-нет}
if [ "$REAL_ST" != "running" ]; then docker start unified-real >/dev/null 2>&1; sleep 5; REAL_ST=$(docker inspect -f {{.State.Status}} unified-real 2>/dev/null); FIX="$FIX реал-поднят;"; fi
ERR_R=$(docker logs unified-real --since 4h 2>&1 | grep -c Traceback)

# 2) БАЛАНС (IP в белом списке, напрямую)
BAL=$(python3 - <<PYEOF
import os, time, hashlib, hmac, json, urllib.request
from urllib.parse import urlencode
from datetime import datetime, timezone
env = {}
for line in open("/root/usdc-real/.env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); env[k] = v
KEY = env["API_KEY"]; SEC = env["API_SECRET"]
def signed(path, params=None):
    p = dict(params or {}); p.update({"recvWindow": 20000, "timestamp": int(time.time()*1000)})
    q = urlencode(p); sig = hmac.new(SEC.encode(), q.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(f"https://fapi.binance.com{path}?{q}&signature={sig}", headers={"X-MBX-APIKEY": KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())
for b in signed("/fapi/v3/balance"):
    if b["asset"] == "USDT":
        bal = b["balance"]
mid = int(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()*1000)
# торговый день = сделки+комиссии+фандинг (переводы не считаем — это не торговля)
day = sum(float(i["income"]) for i in signed("/fapi/v1/income", {"startTime": mid, "limit": 1000})
          if i["incomeType"] != "TRANSFER")
pos = [p for p in signed("/fapi/v2/positionRisk") if abs(float(p["positionAmt"])) > 0]
print(f"Баланс: {bal} USDT | день: {day:+.2f} | позиций: {len(pos)}")
for p in pos[:5]:
    d = "ШОРТ" if float(p["positionAmt"]) < 0 else "ЛОНГ"
    print(f"  {p['symbol']} {d} | PnL {float(p['unRealizedProfit']):+.2f}")
PYEOF
) 2>/dev/null
if [ -z "$BAL" ]; then
  send "⚠️ АУДИТ: биржа не ответила о балансе — сводка отложена. НЕ значит, что позиций нет."
  exit 1
fi

# 3) amount_bot
AMT_ST=$(docker inspect -f {{.State.Status}} amount_bot-amountbot-1 2>/dev/null); AMT_ST=${AMT_ST:-нет}

# 4) Диск
DF=$(df --output=avail -BM / | tail -1 | tr -dc "0-9")
if [ "${DF:-9999}" -lt 400 ]; then
  docker image prune -af >/dev/null 2>&1; docker builder prune -af >/dev/null 2>&1
  DF=$(df --output=avail -BM / | tail -1 | tr -dc "0-9"); FIX="$FIX диск-почищен;"
fi

MSG="🕐 АУДИТ $(TZ=Europe/Moscow date +%H:%M) МСК
▪️ Реал: $REAL_ST | ошибки 4ч: $ERR_R
▪️ $BAL
▪️ Amount_bot: $AMT_ST
▪️ Диск: ${DF}МБ
${FIX:+▪️ Действия: $FIX}"
echo "$MSG" > /tmp/last_audit.txt
send "$MSG"
