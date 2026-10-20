#!/bin/bash
# Ночная выгрузка стаканного датасета на Кегл (12+ГБ, ~полчаса на 6МБ/с).
# Раз в сутки: тренеры (лоб + нейронки) читают данные отсюда.
set -e
exec 9>/tmp/push_lob.lock
flock -n 9 || { echo "уже идёт"; exit 0; }
STAGE=/home/iek/lob_archive/dataset
# свежая порция еды в staging: лестница + лента (лента мелкая, мегабайты)
rsync -az --quiet /home/iek/lob_archive/feat_*.parquet "$STAGE/" 2>/dev/null || true
# ЛИМИТ КЕГЛЯ: 500 файлов на датасет. Лента (сотни мелких файлов) идёт
# ОДНИМ архивом, паркет старше 30 дней из выгрузки убираем (полная
# история остаётся на локальной машине и во Франции).
rsync -az --quiet /home/iek/lob_archive/tape_*.csv.gz "$STAGE/" 2>/dev/null || true
tar czf "$STAGE/tapes_latest.tar.gz" -C /home/iek/lob_archive $(cd /home/iek/lob_archive && ls tape_*.csv.gz) 2>/dev/null
find "$STAGE" -name "tape_*.csv.gz" -delete 2>/dev/null
find "$STAGE" -name "tapes_*.tar.gz" ! -name "tapes_$(date +%Y%m%d).tar.gz" -delete 2>/dev/null
find "$STAGE" -name "feat_*.parquet" -mtime +30 -delete 2>/dev/null
find "$STAGE" -name "*.tmp" -delete 2>/dev/null
cd "$STAGE"
OK=0
for try in 1 2 3; do
  if timeout 3600 python3 -m kaggle datasets version -p . -m "daily" >> /home/iek/lob_archive/kaggle_push.log 2>&1; then OK=1; break; fi
  echo "попытка $try не удалась, пауза 5 мин" >> /home/iek/lob_archive/kaggle_push.log
  sleep 300
done
if [ "$OK" = "1" ]; then
  echo "$(date -u +%FT%TZ) кагл-версия создана" >> /home/iek/lob_archive/kaggle_push.log
else
  /home/iek/PycharmProjects/USDC/ops/tg_notify.sh "⚠️ Ночная выгрузка стакана на Кегл не удалась — тренеры получат старые данные"
  exit 1
fi
