#!/bin/bash
# Дневной теневой судья Бури: вчера -> модель -> исходы -> сводка.
# Cron 06:50 по локальному времени (дневные пакеты биржи уже Published).
exec 9>/tmp/daily_judge.lock
flock -n 9 || { echo "уже идёт"; exit 0; }
cd /home/iek/PycharmProjects/USDC
python3 ops/daily_judge.py >> ops/judge/cron.log 2>&1
SUM=$(cat ops/judge/summary.txt 2>/dev/null)
if [ -n "$SUM" ]; then
  bash ops/tg_notify.sh "$SUM"
fi
