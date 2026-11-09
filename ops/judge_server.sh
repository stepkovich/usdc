#!/bin/bash
# СУДЬЯ НА СЕРВЕРЕ: вчерашний день -> модель -> исходы -> сводка в ТГ.
# Контейнер: тот же образ, что у бота; репо и том данных смонтированы.
exec 9>/tmp/judge.lock
flock -n 9 || { echo "уже идёт"; exit 0; }
cd /root/usdc
docker run --rm \
  -v /root/usdc:/repo -w /repo \
  -v /root/usdc-real/data/models:/repo/models:ro \
  -v /root/usdc-real/data:/app/data \
  -e JUDGE_ROOT=/repo \
  --entrypoint python3 usdc-unified /repo/ops/daily_judge.py \
  >> /root/usdc/ops/judge/cron.log 2>&1
SUM=$(cat /root/usdc/ops/judge/summary.txt 2>/dev/null)
source /root/usdc/.tg_env 2>/dev/null
if [ -n "$SUM" ]; then
  curl -s -o /dev/null -m 20 -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" --data-urlencode "text=$SUM"
fi
# ФЛАГ ПЕРЕОБУЧЕНИЯ: бот просит пересмотр (30 сделок в минусе)
if [ -f /root/usdc-real/data/signals/retrain_needed ]; then
  curl -s -o /dev/null -m 20 -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" --data-urlencode "text=🧠⚠️ Флаг переобучения активен (30 сделок в минусе) — решение за владельцем"
fi
