FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    flask==3.0.3 \
    gunicorn==22.0.0 \
    google-genai==1.64.0 \
    google-cloud-firestore==2.23.0 \
    google-cloud-aiplatform==1.138.0 \
    numpy==2.4.2 \
    scipy==1.17.0 \
    python-dotenv==1.2.1 \
    requests==2.32.5

COPY server.py .
COPY dashboard.html .
COPY core/ ./core/

ENV PORT=8080

CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 server:app
