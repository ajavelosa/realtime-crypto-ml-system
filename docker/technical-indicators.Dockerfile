# Technical indicators service - includes ta-lib + dependencies from pyproject.toml
FROM base:latest

USER root

# Install system dependencies and ta-lib C library (only needed for this service)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    ca-certificates \
    wget \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Download and install ta-lib 0.6.4 with proper ARM64 support
ENV TALIB_DIR=/usr/local
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4/ && \
    ./configure --prefix=$TALIB_DIR && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.6.4-src.tar.gz ta-lib-0.6.4/ && \
    ldconfig

# Copy workspace files for proper dependency resolution
COPY pyproject.toml uv.lock ./
COPY services/ ./services/

# Install only technical-indicators package dependencies using uv workspace (including talib extra)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package technical-indicators --extra talib && \
    # Basic cleanup \
    find .venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find .venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
    echo "=== TECHNICAL-INDICATORS VENV SIZE ===" && du -sh .venv

# Set ownership for app directory
RUN chown -R appuser:appuser /app

USER appuser

WORKDIR /app/services/technical-indicators

CMD ["uv", "run", "--package", "technical-indicators", "python", "/app/services/technical-indicators/src/technical_indicators/main.py"]
