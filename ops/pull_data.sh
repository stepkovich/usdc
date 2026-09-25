#!/bin/bash
# ПРИЁМ данных с сервера (архив стакана + журналы сделок) на локальную машину.
# Запускается по cron каждый час (и вручную). Машина выключена — пропустит,
# догонит при следующем включении.
set -e
VPS=root@159.65.133.26
DEST=/home/iek/lob_archive
mkdir -p "$DEST/ml5h"
if ! rsync -az --quiet "$VPS:usdc/data/lob/archive/" "$DEST/"; then
  /home/iek/PycharmProjects/USDC/ops/tg_notify.sh "⚠️ Приём данных стакана с сервера НЕ УДАЛСЯ — проверь связь"
fi
rsync -az --quiet "$VPS:usdc/data/ml5h_trades.csv" "$DEST/ml5h/" 2>/dev/null || true
rsync -az --quiet "$VPS:usdc/data/lob/trades.csv" "$DEST/lob_trades.csv" 2>/dev/null || true
echo "$(date -u +%FT%TZ) данные получены" >> /home/iek/lob_archive/pull.log
