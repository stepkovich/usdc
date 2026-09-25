FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY bot/ ./bot/
COPY ml5h/ ./ml5h/
COPY lob/ ./lob/
COPY main.py .
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    BOT_DATA_DIR=/app/data

CMD ["python", "main.py"]
