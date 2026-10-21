#!/bin/bash
# МЕСЯЧНОЕ ПЕРЕОБУЧЕНИЕ МЕТЫ (2-е число месяца, 10:00 МСК).
# Идея владельца 20.10: мета учится с задержкой месяц — в ноябре смотрит,
# как отработали октябрь и её собственные решения. Биржа/панель к этому
# моменту уже опубликовали полный месяц.
set -e
cd /home/iek/PycharmProjects/USDC

# 1) панель-мета: месяцы через последний завершённый + дни текущего месяца
python3 - <<'PYEOF'
import json
from datetime import date, timedelta
p = 'kg/panel_downloader/panel_meta.json'
m = json.load(open(p))
today = date.today()
first = today.replace(day=1)
last_month = first - timedelta(days=1)
lm = last_month.strftime('%Y-%m')
months = m.get('months5y') or m.get('months') or []
if not months or months[-1] != lm:
    months = sorted(set(months + [lm]))
    m['months'] = months
    if 'months5y' in m:
        m['months5y'] = months
m['extra_days'] = [(first + __import__('datetime').timedelta(days=i)).strftime('%Y-%m-%d')
                   for i in range((today - first).days)]
json.dump(m, open(p, 'w'), ensure_ascii=False)
print('панель-мета: месяцы до', months[-1], '| доп. дни:', len(m['extra_days']))
PYEOF

# 2) версия датасета меты + пуш загрузчика
python3 -m kaggle datasets version -p kg/panel_downloader -m "monthly meta refresh" --dir-mode skip 2>&1 | tail -1
python3 -m kaggle kernels push -p kg/panel_downloader 2>&1 | tail -1

# 3) ждём загрузчик (максимум 30 мин)
for i in $(seq 1 30); do
  sleep 60
  S=$(python3 -m kaggle kernels status sewerted/usdc-panel-downloader 2>&1)
  echo "$S" | grep -q COMPLETE && break
  echo "$S" | grep -qE "ERROR|CANCEL" && { ./ops/tg_notify.sh "⚠️ Мета-цикл: загрузчик панели упал"; exit 1; }
done

# 4) пуш меты и ожидание
python3 -m kaggle kernels push -p kg/storm_meta 2>&1 | tail -1
for i in $(seq 1 40); do
  sleep 60
  S=$(python3 -m kaggle kernels status sewerted/usdc-storm-meta 2>&1)
  echo "$S" | grep -q COMPLETE && break
  echo "$S" | grep -qE "ERROR|CANCEL" && { ./ops/tg_notify.sh "⚠️ Мета-цикл: переобучение меты упало на Кегле"; exit 1; }
done

# 5) забираем meta.txt и развозим на оба сервера
rm -rf /tmp/meta_cycle && mkdir -p /tmp/meta_cycle
python3 -m kaggle kernels output sewerted/usdc-storm-meta -p /tmp/meta_cycle >/dev/null 2>&1
if [ -f /tmp/meta_cycle/meta.txt ]; then
  cp /tmp/meta_cycle/meta.txt models/meta.txt
  scp -q models/meta.txt root@159.65.133.26:usdc/data/models/meta.txt
  scp -qP 9252 models/meta.txt root@VPS-872366.ssh.vps1euro.fr:/root/usdc-real/data/models/meta.txt
  ssh root@159.65.133.26 'docker restart unified' >/dev/null 2>&1
  ssh -p 9252 root@VPS-872366.ssh.vps1euro.fr 'systemctl restart usdc-real' >/dev/null 2>&1
  ./ops/tg_notify.sh "🧠✅ Мета переобучена на закрытом месяце и развезена на оба сервера (боты перезапущены)"
else
  ./ops/tg_notify.sh "⚠️ Мета-цикл: файл meta.txt не получен с Кегля — работает предыдущая мета"
  exit 1
fi
