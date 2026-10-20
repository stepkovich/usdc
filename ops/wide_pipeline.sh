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
    # ----- МЕТА: переобучение на исходах сигналов свежей модели -----
    # (владелец 20.10: обе модели переучиваются вместе, мета учится
    #  на том, угадала ли. После деплоя новой бури — её сигналы,
    #  накопленные ходьбой по истории на Кегле, кормят мету.)
    ./ops/tg_notify.sh "🧮 Этап 2: переобучение меты на свежей истории сигналов (~15 мин)"
    (cd kg/storm_meta && python3 -m kaggle kernels push -p . >/dev/null 2>&1)
    META_ST=""
    for i in $(seq 1 40); do
      sleep 60
      META_ST=$(python3 -m kaggle kernels status sewerted/usdc-storm-meta 2>&1)
      echo "$META_ST" | grep -q "COMPLETE" && break
      echo "$META_ST" | grep -qE "ERROR|CANCEL" && break
    done
    rm -rf /tmp/meta_out && mkdir -p /tmp/meta_out
    python3 -m kaggle kernels output sewerted/usdc-storm-meta -p /tmp/meta_out >/dev/null 2>&1 || true
    if [ -f /tmp/meta_out/meta.txt ]; then
      cp /tmp/meta_out/meta.txt models/meta.txt
      scp -q models/meta.txt root@159.65.133.26:usdc/data/models/meta.txt
      timeout 60 scp -qP 9252 models/meta.txt root@VPS-872366.ssh.vps1euro.fr:/root/usdc-real/data/models/meta.txt \
        && timeout 60 ssh -p 9252 root@VPS-872366.ssh.vps1euro.fr "systemctl restart usdc-real" || true
      docker restart unified >/dev/null 2>&1 || true
      git add models/meta.txt 2>/dev/null || true
      git -c user.name=iek -c user.email=iek@local commit -m "meta retrained $(date -u +%F)" >/dev/null 2>&1 || true
      ./ops/tg_notify.sh "🧠✅ Мета переобучена на свежей истории сигналов и уже на обоих серверах (боты перезапущены)"
    else
      ./ops/tg_notify.sh "🧠⚠️ Мета: файл модели не получен с Кегля — работает старая"
    fi
  else
    ./ops/tg_notify.sh "🧠⚠️ Широкая модель НЕ прошла критерии — оставлена текущая (20 монет). Отчёт в models/wide_report.json"
  fi
else
  ./ops/tg_notify.sh "🧠❌ Широкое переобучение упало — смотри логи, работаем на старой модели. Повтор: вручную или при перезагрузке"
fi
