#!/bin/bash
# ПЕРЕОБУЧЕНИЕ ML-5Ч на локальной машине + отправка модели на сервер.
# Каждые ~10 дней по cron (окно walk-forward 20 дней — с запасом).
# Свет погас / машина выключена — сервер продолжает торговать старой
# моделью, обучение догонит при следующем включении.
set -e
cd /home/iek/PycharmProjects/USDC
export PYTHONPATH=/home/iek/.zcode/workspace/default/scalp_research/pylibs
if ! python3 verification/ml5h_train_final.py; then
  ./ops/tg_notify.sh "🧠❌ Переобучение ML-5ч ПРОВАЛИЛОСЬ — модель не обновлена, сервер торгует старой"
  exit 1
fi
# отправка модели на сервер (том данных, движок подхватит без рестарта)
scp -q models/ml5h.txt models/ml5h_meta.json root@159.65.133.26:usdc/data/models/
# фиксация в репозитории + публикация в GitHub через VPS
git add models/
git -c user.name=iek -c user.email=iek@local commit -m "ml5h model retrain $(date -u +%F)" || echo "нет изменений"
git bundle create /tmp/ml5h_bundle.bundle main
scp -q /tmp/ml5h_bundle.bundle root@159.65.133.26:/tmp/
ssh root@159.65.133.26 'cd ~/usdc && git pull -q /tmp/ml5h_bundle.bundle main && git push -q origin main'
echo "$(date -u +%FT%TZ) модель переобучена и отправлена" >> /home/iek/lob_archive/pull.log
/home/iek/PycharmProjects/USDC/ops/tg_notify.sh "🧠✅ ML-5ч переобучена и отправлена на сервер ($(date -u +%F)); движок подхватит без рестарта"
