# Production Dockerfile for Google Cloud Run (CPU / Cloud Run with GPU)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HOST=0.0.0.0 \
    JAX_PLATFORMS=cpu \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY app/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application source code and configurations
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY app/ /app/app/

# Expose port (Cloud Run sets PORT automatically, defaults to 8080)
EXPOSE 8080

# Run FastAPI with Uvicorn
CMD ["sh", "-c", "uvicorn app.backend.main:app --host $HOST --port $PORT --workers 1"]
