# USDC Scalper — тренд-ретест бот (Binance USDC-M фьючерсы)

Стратегия: пробой 8-часового канала Дончиана -> лимитка на откате к уровню ->
TP-лимитка (+0.5%) + STOP_MARKET (12xATR) -> кулдаун 120 минут.
Сайзинг: риск на один стоп = RISK_PCT баланса / дистанция стопа,
пол = minNotional x 1.1 (меньше — пропуск), потолок = MAX_NOTIONAL.
Биржа — единственный источник правды: реконсиляция каждую минуту,
восстановление TP/стопов после сбоев, двойной учёт PnL (биржа / тариф).

## Запуск на сервере (Docker)

```bash
git clone <ваш-репо> && cd USDC
cp .env.example .env
nano .env                # вписать API_KEY/API_SECRET, проверить BOT_MODE
docker compose up -d --build
docker compose logs -f   # смотреть логи (Ctrl+C не останавливает)
```

Контейнер перезапускается автоматически (restart: always) при падении
и перезагрузке VPS. Журнал сделок и логи лежат в ./data и переживают
пересоздание контейнера.

## Полезные команды

```bash
docker compose logs --tail 100 scalper   # последние строки
docker compose restart scalper           # мягкий перезапуск
docker compose down                      # остановить
docker compose exec scalper python -m bot.reconcile 24   # сверка с биржей за 24ч
docker compose exec scalper python -m bot.plumbing_test  # тест ордерной обвязки
```

## Режимы и безопасность

- `BOT_MODE=demo` — демо-счёт (demo-fapi.binance.com) — ПО УМОЛЧАНИЮ
- `BOT_MODE=testnet` — тестнет (тот же бэкенд)
- `BOT_MODE=mainnet` — РЕАЛЬНЫЕ ДЕНЬГИ. Сначала 1-2 недели на демо,
  затем реалнет с малым балансом.
- `BOT_DRY_RUN=1` — торговые сигналы только в журнал, ордера не ставятся.

**Важно:** файл `.env` с ключами не должен попадать в репозиторий
(он в .gitignore). Если ключи засветились — перевыпустить на бирже.
Ключам на реалнете: запрет вывода средств, ограничение по IP.

## Мониторинг

- `docker compose logs` — сигналы (СИГНАЛ/ВХОД/ВЫХОД/САЙЗИНГ), реконсиляция
- `./data/journal.db` — SQLite: события, исполнения, сделки (таблица trades)
- `python -m bot.reconcile <часов>` — авторитетная сверка PnL с биржей
# usdc
