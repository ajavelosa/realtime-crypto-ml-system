# Candles service - ultra-lightweight streaming service
FROM base:latest

USER root

# Copy workspace files for proper dependency resolution
COPY pyproject.toml uv.lock ./
COPY services/ ./services/

# Install only candles package dependencies using uv workspace
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package candles --no-dev && \
    # Basic cleanup \
    find .venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find .venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
    echo "=== CANDLES VENV SIZE ===" && du -sh .venv

# Set ownership for app directory
RUN chown -R appuser:appuser /app

USER appuser

WORKDIR /app/services/candles

CMD ["uv", "run", "--package", "candles", "python", "/app/services/candles/src/candles/main.py"]
