#!/bin/bash
# Уведомление в Telegram из локальных скриптов. $1 = текст.
[ -f /home/iek/.tg_creds ] || exit 0
source /home/iek/.tg_creds
[ -n "$TELEGRAM_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ] || exit 0
curl -s -o /dev/null -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
  -d chat_id="${TELEGRAM_CHAT_ID}" -d parse_mode=HTML --data-urlencode "text=$1"
