#!/bin/bash
# ПРИЁМ данных с сервера (архив стакана + журналы сделок) на локальную машину.
# Запускается по cron каждый час (и вручную). Машина выключена — пропустит,
# догонит при следующем включении.
set -e
# один запуск за раз: параллельный (cron + вручную) молча пропускает
exec 9>/tmp/pull_data.lock
flock -n 9 || { echo "pull_data уже идёт — пропуск"; exit 0; }
VPS=root@159.65.133.26
DEST=/home/iek/lob_archive
mkdir -p "$DEST/ml5h"
# *.tmp исключаем: рекордер пишет часовой файл через tmp и переименовывает —
# rsync ловил "file has vanished" (код 24) и будил ТГ ложной тревогой.
# Код 24 = штатная гонка, недоехавший файл довозит следующий час.
set +e
rsync -az --quiet --exclude='*.tmp' "$VPS:usdc/data/lob/archive/" "$DEST/"
rc=$?
set -e
if [ "$rc" -ne 0 ] && [ "$rc" -ne 24 ]; then
  /home/iek/PycharmProjects/USDC/ops/tg_notify.sh "⚠️ Приём данных стакана с сервера НЕ УДАЛСЯ — проверь связь"
fi
rsync -az --quiet "$VPS:usdc/data/ml5h_trades.csv" "$DEST/ml5h/" 2>/dev/null || true
rsync -az --quiet "$VPS:usdc/data/lob/trades.csv" "$DEST/lob_trades.csv" 2>/dev/null || true
echo "$(date -u +%FT%TZ) данные получены" >> /home/iek/lob_archive/pull.log
# Выгрузка на Кегл вынесена в ночную (push_lob_dataset.sh, 04:00):
# датасет дорос до 12ГБ и часовой полный перезалив не помещался ни в какой таймаут.
# Архив стакана растёт ~1.1ГБ/день (сырые уровни с 01.10): если на
# локальном диске остаётся меньше 100ГБ — кричим, пока не поздно
FREE_GB=$(df --output=avail -BG /home | tail -1 | tr -dc '0-9')
if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt 100 ]; then
  /home/iek/PycharmProjects/USDC/ops/tg_notify.sh "⚠️ Локальный диск: свободно ${FREE_GB}ГБ — архив стакана пора подрезать"
fi
