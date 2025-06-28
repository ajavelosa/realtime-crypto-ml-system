# Trades service - ultra-lightweight streaming service
FROM base:latest

USER root

# Copy workspace files for proper dependency resolution
COPY pyproject.toml uv.lock ./
COPY services/ ./services/

# Install only trades package dependencies using uv workspace with debug output
RUN --mount=type=cache,target=/root/.cache/uv \
    echo "=== BEFORE UV SYNC ===" && \
    ls -la /app/.venv/ 2>/dev/null || echo "No .venv yet" && \
    uv sync --frozen --package trades --no-dev && \
    echo "=== AFTER UV SYNC ===" && \
    ls -la /app/.venv/ && \
    ls -la /app/.venv/lib/python3.12/site-packages/ | head -10 && \
    echo "=== TESTING IMPORTS ===" && \
    /app/.venv/bin/python -c "import quixstreams, loguru; print('✅ Dependencies installed successfully!')" && \
    # Basic cleanup \
    find .venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find .venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
    echo "=== TRADES VENV SIZE ===" && du -sh .venv

# Set ownership for app directory
RUN chown -R appuser:appuser /app

USER appuser

WORKDIR /app/services/trades

CMD ["uv", "run", "--package", "trades", "python", "/app/services/trades/src/trades/main.py"]
