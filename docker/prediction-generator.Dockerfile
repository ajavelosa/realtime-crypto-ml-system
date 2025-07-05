# Training pipeline - predictor service with heavy ML dependencies
FROM base:latest

USER root

# Install additional build tools for heavy ML packages
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace files for proper dependency resolution
COPY pyproject.toml uv.lock ./
COPY services/ ./services/

# Install only predictor package dependencies using uv workspace (including training extras)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package predictor --extra training && \
    # Basic cleanup \
    find .venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find .venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
    echo "=== TRAINING-PIPELINE VENV SIZE ===" && du -sh .venv

# Set timeout for heavy ML package downloads
ENV UV_HTTP_TIMEOUT=300

# Set ownership for app directory
RUN chown -R appuser:appuser /app

USER appuser

WORKDIR /app/services/predictor

CMD ["uv", "run", "--package", "predictor", "python", "/app/services/predictor/src/predictor/predict.py"]
