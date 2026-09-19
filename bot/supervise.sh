#!/bin/bash
# Супервизор бота без sudo: рестарт при падении + журнал причин по построению.
# Каждое завершение процесса логируется с exit-кодом и временем -> инициатор
# любого рестарта известен из этого файла, а не расследуется.
DIR="$(cd "$(dirname "$0")/.." && pwd)"
while true; do
  echo "$(date '+%F %T') START bot" >> "$DIR/bot/supervisor.log"
  cd "$DIR"
  PYTHONPATH=.:pylibs python3 -m bot.bot >> "$DIR/bot/live.log" 2>> "$DIR/bot/stderr.log"
  RC=$?
  echo "$(date '+%F %T') EXIT code=$RC (краш/стоп — причина в stderr.log)" >> "$DIR/bot/supervisor.log"
  sleep 10
done
