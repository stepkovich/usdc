#!/bin/bash
# ДЕПЛОЙ кода: локальный коммит -> сервер (pull из bundle) -> GitHub -> пересборка.
set -e
cd /home/iek/PycharmProjects/USDC
git add -A
git -c user.name=iek -c user.email=iek@local commit -m "deploy $(date -u +%FT%TZ)" || echo "нечего коммитить"
git bundle create /tmp/usdc_deploy.bundle main
scp -q /tmp/usdc_deploy.bundle root@159.65.133.26:/tmp/
ssh root@159.65.133.26 'cd ~/usdc && git pull -q /tmp/usdc_deploy.bundle main && git push -q origin main && docker compose up -d --build'
echo "задеплоено"
