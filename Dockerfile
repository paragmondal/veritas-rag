# ==============================================================================
# Veritas — Backend Dockerfile
# ==============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for compiling native extensions if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, configurations, data, and scripts
COPY .env.example .env
COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data/
COPY eval/ ./eval/
COPY scripts/ ./scripts/

# Expose FastAPI port
EXPOSE 8000

# Entrypoint: Build indexes if missing and start uvicorn server
CMD ["sh", "-c", "python -m src.embed_index && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
