FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot/ ./bot/

# журнал и лог идут в /app/data (том из docker-compose)
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    BOT_JOURNAL_PATH=/app/data/journal.db \
    BOT_LOG_PATH=/app/data/bot.log

CMD ["python", "-m", "bot.bot"]
