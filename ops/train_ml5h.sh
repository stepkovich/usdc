#!/bin/bash
# ПЕРЕОБУЧЕНИЕ ML-5Ч на локальной машине + отправка модели на сервер.
# Каждые ~10 дней по cron (окно walk-forward 20 дней — с запасом).
# Свет погас / машина выключена — сервер продолжает торговать старой
# моделью, обучение догонит при следующем включении.
set -e
cd /home/iek/PycharmProjects/USDC
export PYTHONPATH=/home/iek/.zcode/workspace/default/scalp_research/pylibs
python3 verification/ml5h_train_final.py
# отправка модели на сервер (том данных, движок подхватит без рестарта)
scp -q models/ml5h.txt models/ml5h_meta.json root@159.65.133.26:usdc/data/models/
# фиксация в репозитории + публикация в GitHub через VPS
git add models/
git -c user.name=iek -c user.email=iek@local commit -m "ml5h model retrain $(date -u +%F)" || echo "нет изменений"
git bundle create /tmp/ml5h_bundle.bundle main
scp -q /tmp/ml5h_bundle.bundle root@159.65.133.26:/tmp/
ssh root@159.65.133.26 'cd ~/usdc && git pull -q /tmp/ml5h_bundle.bundle main && git push -q origin main'
echo "$(date -u +%FT%TZ) модель переобучена и отправлена" >> /home/iek/lob_archive/pull.log
