FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app/main.py
COPY swatches.json /app/swatches.json

ENV PORT=8080
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
