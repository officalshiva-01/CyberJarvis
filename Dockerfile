FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data /app/logs \
    && useradd --create-home --shell /usr/sbin/nologin cyberjarvis \
    && chown -R cyberjarvis:cyberjarvis /app /data

ENV CYBERJARVIS_DB_PATH=/data/cyberjarvis.db

USER cyberjarvis

CMD ["python", "main.py", "--no-tts"]
