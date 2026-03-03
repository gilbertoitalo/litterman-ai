FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit==1.45.1 \
    plotly==6.1.1 \
    google-genai==1.64.0 \
    google-cloud-firestore==2.23.0 \
    google-cloud-aiplatform==1.138.0 \
    numpy==2.4.2 \
    scipy==1.17.0 \
    python-dotenv==1.2.1 \
    requests==2.32.5

COPY dashboard.py .
COPY core/ ./core/

RUN mkdir -p /root/.streamlit
COPY streamlit_config.toml /root/.streamlit/config.toml

ENV PORT=8080

CMD streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0
