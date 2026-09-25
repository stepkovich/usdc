#!/bin/bash
# НОЧНОЙ КОНВЕЙЕР: ждёт окончания загрузки панели -> широкое переобучение
# -> вердикт по жёстким критериям -> авто-деплой или авто-отказ -> Telegram.
set -e
cd /home/iek/PycharmProjects/USDC
FLAG=/home/iek/lob_archive/wide_done.flag
[ -f "$FLAG" ] && exit 0
# ждём загрузку (максимум 4 часа)
for i in $(seq 1 48); do
  grep -q "ГОТОВО" ops/download.log && break
  sleep 300
done
if ! grep -q "ГОТОВО" ops/download.log; then
  ./ops/tg_notify.sh "🌙 Широкая панель НЕ докачалась за 4ч — конвейер отложен, история качается дальше"
  exit 1
fi
export PYTHONPATH=/home/iek/.zcode/workspace/default/scalp_research/pylibs
./ops/tg_notify.sh "🌙 Панель готова: стартует широкое переобучение (527 монет, walk-forward, нуль-тест). Итог через 1.5-3ч"
if python3 verification/ml5h_train_wide.py; then
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('models/wide_report.json'))['verdict'] else 1)"; then
    scp -q models/ml5h.txt models/ml5h_meta.json root@159.65.133.26:usdc/data/models/
    git add models/ verification/ml5h_wide_trades.csv 2>/dev/null || true
    git -c user.name=iek -c user.email=iek@local commit -m "wide model deployed $(date -u +%F)" || true
    git bundle create /tmp/wide.bundle main && scp -q /tmp/wide.bundle root@159.65.133.26:/tmp/
    ssh root@159.65.133.26 'cd ~/usdc && git pull -q /tmp/wide.bundle main && git push -q origin main'
    ./ops/tg_notify.sh "🧠✅ ШИРОКАЯ МОДЕЛЬ ПРИНЯТА по критериям и УЖЕ НА СЕРВЕРЕ (подхватится сама)"
  else
    ./ops/tg_notify.sh "🧠⚠️ Широкая модель НЕ прошла критерии — оставлена текущая (20 монет). Отчёт в models/wide_report.json"
  fi
else
  ./ops/tg_notify.sh "🧠❌ Широкое переобучение упало — смотри логи, работаем на старой модели"
fi
touch "$FLAG"
