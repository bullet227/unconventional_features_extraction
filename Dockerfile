# Dockerfile for Unconventional Features ML Pipeline
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: Builder - Install dependencies and build wheels
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create wheels for all dependencies
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ============================================================================
# Stage 2: Runtime - Final lightweight image
# ============================================================================
FROM python:3.11-slim AS runtime

# Labels
LABEL maintainer="Forex ML Pipeline"
LABEL description="Unconventional Features Extraction and ML Pipeline for Forex Trading"
LABEL version="0.1.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Default database configuration
    POSTGRES_HOST=postgres \
    POSTGRES_PORT=5432 \
    POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=forex_ml_2024 \
    FOREX_DB=forex_trading_data \
    UNCONVENTIONAL_DB=unconventional_features \
    # Pipeline configuration
    BATCH_SIZE=250000 \
    PARALLEL_WORKERS=4 \
    ENABLE_IMAGE_FEATURES=false \
    ENABLE_PROMETHEUS=false \
    # Model defaults
    MODEL_TYPE=xgboost \
    N_SPLITS=5 \
    MAX_FEATURES=100

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Create directories for data, models, and reports
RUN mkdir -p /app/data /app/models /app/reports /app/logs

# Copy application code
COPY features/ ./features/
COPY ml_pipeline/ ./ml_pipeline/
COPY unconventional_features.py .

# Create non-root user for security
RUN groupadd -r mlpipeline && useradd -r -g mlpipeline mlpipeline \
    && chown -R mlpipeline:mlpipeline /app

USER mlpipeline

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from ml_pipeline import DataLoader; print('healthy')" || exit 1

# Default command: run the pipeline with sample data
CMD ["python", "ml_pipeline/run_pipeline.py", "--sample", "--output", "/app/reports"]

# Alternative entry points available via docker run:
# - Full pipeline: python ml_pipeline/run_pipeline.py --asset EURUSD --timeframe H1
# - Feature ETL: python unconventional_features.py --assets EURUSD --timeframes H1
# - Interactive: python -i
