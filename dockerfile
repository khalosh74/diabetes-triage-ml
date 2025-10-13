# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (build tools + curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first for better caching
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY src ./src
COPY api ./api
COPY README.md .
COPY CHANGELOG.md .

# Make artifacts dir and train during build (deterministic)
RUN mkdir -p artifacts && python src/diaclinic/train.py && ls -lah artifacts

# Runtime environment
ENV PORT=8000
EXPOSE 8000

# Default command: serve API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
