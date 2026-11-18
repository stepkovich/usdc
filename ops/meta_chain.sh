#!/bin/bash
# ЦЕПОЧКА ТЕТРАДИ (владелец 16.11 «не буду постоянно спрашивать»):
# панель COMPLETE -> пуш storm_meta v10 -> COMPLETE -> деплой meta.txt
# (hot-reload) + meta.txt/notebook.csv в репо + ТГ. Маркеры в
# /root/usdc-real/.chain/ — каждый этап выполняется один раз.
set -a; source /root/usdc/.env 2>/dev/null; set +a
source /root/usdc/.tg_env 2>/dev/null
exec 9>/tmp/meta_chain.lock
flock -n 9 || exit 0
LOG=/root/usdc/ops/meta_chain.log
say() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }
tg() { [ -n "$TELEGRAM_TOKEN" ] && curl -s -o /dev/null -m 15 -X POST \
  "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
  -d chat_id="${TELEGRAM_CHAT_ID}" --data-urlencode "text=$1"; }
mkdir -p /root/usdc-real/.chain

PS=$(python3 -m kaggle kernels status sewerted/usdc-panel-downloader 2>&1 | tail -1)
case "$PS" in
  *COMPLETE*)
    if [ ! -f /root/usdc-real/.chain/panel_done ]; then
      touch /root/usdc-real/.chain/panel_done
      say "панель COMPLETE — пуш storm_meta v10"
      (cd /root/usdc/kg/storm_meta && \
        python3 -m kaggle kernels push -p . >> "$LOG" 2>&1) && \
        tg "🧠 мета v10 запущена (полный месячный проход + тетрадь)"
    fi
    ;;
  *ERROR*|*CANCEL*)
    if [ ! -f /root/usdc-real/.chain/panel_done ]; then
      touch /root/usdc-real/.chain/panel_done
      say "панель упала ($PS) — не зацикливаемся"
      tg "❌ панель упала — мета не запущена, разберёмся"
    fi
    ;;
  *) exit 0 ;;
esac

MS=$(python3 -m kaggle kernels status sewerted/usdc-storm-meta 2>&1 | tail -1)
case "$MS" in
  *COMPLETE*)
    if [ ! -f /root/usdc-real/.chain/meta_done ]; then
      rm -rf /tmp/mout && mkdir -p /tmp/mout
      python3 -m kaggle kernels output sewerted/usdc-storm-meta -p /tmp/mout >> "$LOG" 2>&1
      if [ -f /tmp/mout/meta.txt ]; then
        cp /tmp/mout/meta.txt /root/usdc-real/data/models/meta.txt
        cp /tmp/mout/meta.txt /root/usdc/models/meta.txt
        [ -f /tmp/mout/notebook.csv ] && \
          cp /tmp/mout/notebook.csv /root/usdc/models/notebook.csv
        (cd /root/usdc && git add models/ && \
          git -c user.name=iek -c user.email=iek@local commit -q \
          -m "мета v10 (непрерывное окно, тетрадь с фандингом)" && \
          git push -q origin main)
        touch /root/usdc-real/.chain/meta_done
        say "мета v10 деплоена (hot-reload), файлы в репо"
        tg "🧠✅ мета v10 в бою (непрерывное окно, тетрадь сохранена в репо)"
      else
        say "meta.txt не найден в выходе v10"
        tg "⚠️ мета v10: файл не получен — старая работает"
      fi
    fi
    ;;
  *ERROR*|*CANCEL*)
    if [ ! -f /root/usdc-real/.chain/meta_done ]; then
      touch /root/usdc-real/.chain/meta_done
      say "мета v10 упала ($MS)"
      tg "❌ мета v10 упала — в бою осталась предыдущая"
    fi
    ;;
esac
