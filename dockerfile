# syntax=docker/dockerfile:1.6

############################
# Base (pinned Python + OS)
############################
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages: build tools (for some wheels), curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl \
    && rm -rf /var/lib/apt/lists/*

############################
# Deps layer (cache-friendly)
############################
FROM base AS deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

############################
# Build/training layer
############################
FROM deps AS build
# Copy only the code we need
COPY src ./src
COPY api ./api
COPY README.md .
COPY CHANGELOG.md .

# Train during build (deterministic: seeds in code)
RUN mkdir -p artifacts \
 && python src/diaclinic/train.py \
 && ls -lah artifacts

############################
# Runtime (lean, non-root)
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install only runtime deps (same pins)
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy app + trained artifacts
COPY --from=build /app/api ./api
COPY --from=build /app/artifacts ./artifacts
COPY --from=build /app/README.md .
COPY --from=build /app/CHANGELOG.md .

# Create a non-root user
RUN useradd -ms /bin/bash appuser
USER appuser

EXPOSE 8000

# Optional container-level healthcheck (FastAPI /ready)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -sf http://127.0.0.1:${PORT}/ready || exit 1

# Start the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
