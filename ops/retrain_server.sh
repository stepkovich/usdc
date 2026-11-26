#!/bin/bash
# МЕСЯЧНОЕ ПЕРЕОБУЧЕНИЕ НА СЕРВЕРЕ (1-го числа): кагл -> экзамен ->
# деплой моделей в бот -> рестарт -> синк репо. Полная копия логики
# monthly_retrain.sh, но пути серверные и деплой локальный.
set -e
cd /root/usdc
source /root/usdc/.tg_env 2>/dev/null
# ключ Кегля из .env (один источник ключей на сервере); файлы ~/.kaggle
# остаются как запасной вариант, если .env не заведён
if [ -f /root/usdc/.env ]; then
  set -a; source /root/usdc/.env; set +a
fi
notify() { curl -s -o /dev/null -m 20 -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" -d chat_id="${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1"; }
notify "🧠 МЕСЯЧНОЕ ПЕРЕОБУЧЕНИЕ (сервер): старт"
(cd kg/panel_downloader && python3 -m kaggle kernels push -p . >/dev/null 2>&1)
for i in $(seq 1 40); do sleep 60; S=$(python3 -m kaggle kernels status sewerted/usdc-panel-downloader 2>&1); echo "$S" | grep -q COMPLETE && break; echo "$S" | grep -qE "ERROR|CANCEL" && { notify "❌ Конвейер: panel упал"; exit 1; }; done
(cd kg/ml5h_wide && python3 -m kaggle kernels push -p . >/dev/null 2>&1)
for i in $(seq 1 60); do sleep 60; S=$(python3 -m kaggle kernels status sewerted/usdc-ml5h-wide 2>&1); echo "$S" | grep -q COMPLETE && break; echo "$S" | grep -qE "ERROR|CANCEL" && { notify "❌ Конвейер: Буря упала"; exit 1; }; done
rm -rf /tmp/wide_out && mkdir -p /tmp/wide_out
python3 -m kaggle kernels output sewerted/usdc-ml5h-wide -p /tmp/wide_out >/dev/null 2>&1
VERDICT=$(python3 -c "import json; print(json.load(open('/tmp/wide_out/report.json')).get('verdict', False))" 2>/dev/null || echo False)
if [ "$VERDICT" = "True" ] && [ -f /tmp/wide_out/ml5h.txt ]; then
  cp /tmp/wide_out/ml5h.txt /tmp/wide_out/ml5h_meta.json /root/usdc-real/data/models/
  [ -f /tmp/wide_out/calibration.json ] && cp /tmp/wide_out/calibration.json /root/usdc-real/data/models/
  # (г) git-архив всеми файлами — иначе декабрьский коммит собрал бы «франкенштейна»
  cp /tmp/wide_out/ml5h.txt /tmp/wide_out/ml5h_meta.json /root/usdc/models/
  [ -f /tmp/wide_out/calibration.json ] && cp /tmp/wide_out/calibration.json /root/usdc/models/
  docker restart unified-real >/dev/null 2>&1 || notify "⚠️ рестарт unified-real не удался"
  notify "🧠✅ Буря сдала — новая модель на сервере"
else
  notify "🧠⚠️ Буря НЕ сдала — остаётся текущая"
fi
notify "🧠 Мета: старт переобучения"
(cd kg/storm_meta && python3 -m kaggle kernels push -p . >/dev/null 2>&1)
for i in $(seq 1 180); do sleep 60; S=$(python3 -m kaggle kernels status sewerted/usdc-storm-meta 2>&1); echo "$S" | grep -q COMPLETE && break; echo "$S" | grep -qE "ERROR|CANCEL" && { notify "⚠️ Мета упала — работает предыдущая"; exit 1; }; done
rm -rf /tmp/meta_cyc && mkdir -p /tmp/meta_cyc
python3 -m kaggle kernels output sewerted/usdc-storm-meta -p /tmp/meta_cyc >/dev/null 2>&1
if [ -f /tmp/meta_cyc/meta.txt ]; then
  cp /tmp/meta_cyc/meta.txt /root/usdc-real/data/models/meta.txt
  cp /tmp/meta_cyc/meta.txt /root/usdc/models/meta.txt
  # отчёт кёрнела (вердикт/фолды) в архив — OOS-различение код-проверяем каждый месяц
  [ -f /tmp/meta_cyc/report.json ] && cp /tmp/meta_cyc/report.json /root/usdc/models/meta_report.json
  docker restart unified-real >/dev/null 2>&1 || notify "⚠️ рестарт unified-real не удался"
  notify "🧠✅ Мета переучена — бот и судья синхронны"
else
  notify "⚠️ Мета: файл не получен — работает предыдущая"
fi
git add models/ && git -c user.name=iek -c user.email=iek@local commit -q -m "модели $(date -u +%F) (серверный конвейер)" || true
git push -q origin main && notify "📦 Модели синкнуты в репозитарий" || notify "⚠️ Синк моделей в репо не прошёл"
notify "🏁 МЕСЯЧНЫЙ ЦИКЛ ЗАВЕРШЁН"
