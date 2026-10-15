#!/bin/bash
# Ночная выгрузка стаканного датасета на Кегл (12+ГБ, ~полчаса на 6МБ/с).
# Раз в сутки: тренеры (лоб + нейронки) читают данные отсюда.
set -e
exec 9>/tmp/push_lob.lock
flock -n 9 || { echo "уже идёт"; exit 0; }
STAGE=/home/iek/lob_archive/dataset
# свежая порция еды в staging: лестница + лента (лента мелкая, мегабайты)
rsync -az --quiet /home/iek/lob_archive/feat_*.parquet "$STAGE/" 2>/dev/null || true
rsync -az --quiet /home/iek/lob_archive/tape_*.csv.gz "$STAGE/" 2>/dev/null || true
find "$STAGE" -name "*.tmp" -delete 2>/dev/null
cd "$STAGE"
if timeout 3600 python3 -m kaggle datasets version -p . -m "daily" >> /home/iek/lob_archive/kaggle_push.log 2>&1; then
  echo "$(date -u +%FT%TZ) кагл-версия создана" >> /home/iek/lob_archive/kaggle_push.log
else
  /home/iek/PycharmProjects/USDC/ops/tg_notify.sh "⚠️ Ночная выгрузка стакана на Кегл не удалась — тренеры получат старые данные"
  exit 1
fi
