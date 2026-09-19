#!/bin/bash
# Супервизор v2: не создаёт дубликатов (проверяет живость перед стартом),
# рестартует при падении, каждый старт/выход — в журнал с причиной.
DIR="$(cd "$(dirname "$0")/.." && pwd)"
while true; do
  RUNNING=$(ps -eo cmd | grep "python3 -m bot.bot" | grep -v grep | grep -cv AppImage)
  if [ "$RUNNING" -eq 0 ]; then
    echo "$(date '+%F %T') START bot (не было живого экземпляра)" >> "$DIR/bot/supervisor.log"
    cd "$DIR"
    PYTHONPATH=.:pylibs python3 -m bot.bot >> "$DIR/bot/live.log" 2>> "$DIR/bot/stderr.log"
    echo "$(date '+%F %T') EXIT code=$? (краш/стоп — причина в stderr.log)" >> "$DIR/bot/supervisor.log"
    sleep 10
  fi
  sleep 15
done
