FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python and deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip libpq-dev git && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt

# App
WORKDIR /app
COPY unconventional_features.py .
COPY analyze_patterns.py .
COPY features/ ./features/
RUN useradd -r appuser
USER appuser

ENTRYPOINT ["python3", "/app/unconventional_features.py"]