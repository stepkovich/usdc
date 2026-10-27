#!/bin/bash
# МЕСЯЧНОЕ ПЕРЕОБУЧЕНИЕ (1-е число, 10:00 МСК):
# 1) panel_downloader на Кегле качает свечи с зеркала биржи (интернет там)
# 2) ml5h_wide на Кегле GPU: тренировка + экзамен 8 кварталов
#    Сдал -> ml5h.txt деплоится на DO-сервер, бот перезапускается
# 3) storm_meta: мета переучивается на сигналах новой Бури -> на DO
# Владелец 27.10: реал на DigitalOcean, Франция упразднена.
set -e
cd /home/iek/PycharmProjects/USDC
source ops/tg_env.sh 2>/dev/null || true
notify() { ./ops/tg_notify.sh "$1"; }

notify "🧠 МЕСЯЧНОЕ ПЕРЕОБУЧЕНИЕ: старт (1-го числа)"

# --- 1. Панель: свежий месяц качается на Кегле ---
(cd kg/panel_downloader && python3 -m kaggle kernels push -p . >/dev/null 2>&1)
for i in $(seq 1 40); do
  sleep 60
  S=$(python3 -m kaggle kernels status sewerted/usdc-panel-downloader 2>&1)
  echo "$i: $S"
  echo "$S" | grep -q COMPLETE && break
  echo "$S" | grep -qE "ERROR|CANCEL" && { notify "❌ Месячный цикл: загрузчик панели упал"; exit 1; }
done

# --- 2. Буря: тренировка + экзамен на Кегле GPU ---
(cd kg/ml5h_wide && python3 -m kaggle kernels push -p . >/dev/null 2>&1)
for i in $(seq 1 60); do
  sleep 60
  S=$(python3 -m kaggle kernels status sewerted/usdc-ml5h-wide 2>&1)
  echo "$i: $S"
  echo "$S" | grep -q COMPLETE && break
  echo "$S" | grep -qE "ERROR|CANCEL" && { notify "❌ Месячный цикл: тренировка Бури упала"; exit 1; }
done

# --- 3. Результат экзамена ---
rm -rf /tmp/wide_out && mkdir -p /tmp/wide_out
python3 -m kaggle kernels output sewerted/usdc-ml5h-wide -p /tmp/wide_out >/dev/null 2>&1
VERDICT=$(python3 -c "import json; print(json.load(open('/tmp/wide_out/report.json')).get('verdict', False))" 2>/dev/null || echo False)
if [ "$VERDICT" = "True" ] && [ -f /tmp/wide_out/ml5h.txt ]; then
  cp /tmp/wide_out/ml5h.txt /tmp/wide_out/ml5h_meta.json /home/iek/PycharmProjects/USDC/models/
  scp -q /home/iek/PycharmProjects/USDC/models/ml5h.txt /home/iek/PycharmProjects/USDC/models/ml5h_meta.json root@159.65.133.26:/root/usdc-real/data/models/
  ssh root@159.65.133.26 'docker restart unified-real' >/dev/null 2>&1
  notify "🧠✅ Буря сдала месячный экзамен — новая модель УЖЕ на сервере"
else
  notify "🧠⚠️ Буря НЕ сдала месячный экзамен — остаётся текущая модель"
fi

# --- 4. Мета: переобучение на сигналах свежей Бури ---
(cd kg/storm_meta && python3 -m kaggle kernels push -p . >/dev/null 2>&1)
for i in $(seq 1 40); do
  sleep 60
  S=$(python3 -m kaggle kernels status sewerted/usdc-storm-meta 2>&1)
  echo "$S" | grep -q COMPLETE && break
  echo "$S" | grep -qE "ERROR|CANCEL" && { notify "⚠️ Мета: переобучение упало — работает предыдущая"; exit 1; }
done
rm -rf /tmp/meta_cyc && mkdir -p /tmp/meta_cyc
python3 -m kaggle kernels output sewerted/usdc-storm-meta -p /tmp/meta_cyc >/dev/null 2>&1
if [ -f /tmp/meta_cyc/meta.txt ]; then
  cp /tmp/meta_cyc/meta.txt /home/iek/PycharmProjects/USDC/models/meta.txt
  scp -q /home/iek/PycharmProjects/USDC/models/meta.txt root@159.65.133.26:/root/usdc-real/data/models/meta.txt
  ssh root@159.65.133.26 'docker restart unified-real' >/dev/null 2>&1
  notify "🧠✅ Мета переучена на свежей истории — развезена, бот перезапущен"
else
  notify "⚠️ Мета: файл не получен — работает предыдущая"
fi
notify "🏁 МЕСЯЧНЫЙ ЦИКЛ ЗАВЕРШЁН"
