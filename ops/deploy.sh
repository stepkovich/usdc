#!/bin/bash
# ДЕПЛОЙ кода (дискобезопасный): коммит -> сервер -> GitHub -> пересборка.
# Порядок важен: на VPS 8.7ГБ, два образа сразу не помещаются —
# сначала down+снос старого образа, потом сборка нового.
set -e
cd /home/iek/PycharmProjects/USDC
[ -f ops/vps.env ] && . ops/vps.env
: ${VPS_SSH:?"адрес сервера: создай ops/vps.env со строкой VPS_SSH=root@хост"}
git add -A
git -c user.name=iek -c user.email=iek@local commit -m "deploy $(date -u +%FT%TZ)" || echo "нечего коммитить"
git bundle create /tmp/usdc_deploy.bundle main
scp -q /tmp/usdc_deploy.bundle ${VPS_SSH:?задай VPS_SSH в ops/vps.env}:/tmp/
ssh ${VPS_SSH:?задай VPS_SSH в ops/vps.env} 'cd ~/usdc && git pull -q /tmp/usdc_deploy.bundle main && git push -q origin main; # модели в каждом коммите — история растёт на ~5МБ/деплой, жмём сразу
git reflog expire --expire=now --all >/dev/null 2>&1; git gc --prune=now --quiet >/dev/null 2>&1; for d in ~/usdc/kg/*/; do n=$(basename "$d"); mkdir -p ~/kg/$n; cp -f "$d"script.py "$d"kernel-metadata.json ~/kg/$n/ 2>/dev/null; done; docker compose down && docker image prune -af >/dev/null 2>&1; docker compose up -d --build && docker image prune -af >/dev/null 2>&1; df -h / | tail -1'
ssh ${VPS_SSH:?задай VPS_SSH в ops/vps.env} 'docker builder prune -af >/dev/null 2>&1' || true
rm -f /tmp/usdc_deploy.bundle
echo "задеплоено (дискобезопасно)"
