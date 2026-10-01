#!/bin/bash
# ДЕПЛОЙ кода (дискобезопасный): коммит -> сервер -> GitHub -> пересборка.
# Порядок важен: на VPS 8.7ГБ, два образа сразу не помещаются —
# сначала down+снос старого образа, потом сборка нового.
set -e
cd /home/iek/PycharmProjects/USDC
git add -A
git -c user.name=iek -c user.email=iek@local commit -m "deploy $(date -u +%FT%TZ)" || echo "нечего коммитить"
git bundle create /tmp/usdc_deploy.bundle main
scp -q /tmp/usdc_deploy.bundle root@159.65.133.26:/tmp/
ssh root@159.65.133.26 'cd ~/usdc && git pull -q /tmp/usdc_deploy.bundle main && git push -q origin main; for d in ~/usdc/kg/*/; do n=$(basename "$d"); mkdir -p ~/kg/$n; cp -f "$d"script.py "$d"kernel-metadata.json ~/kg/$n/ 2>/dev/null; done; docker compose down && docker image prune -af >/dev/null 2>&1; docker compose up -d --build && docker image prune -af >/dev/null 2>&1; df -h / | tail -1'
rm -f /tmp/usdc_deploy.bundle
echo "задеплоено (дискобезопасно)"
