# Use optimized uv base image with Python 3.12
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Install build dependencies for ta-lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install ta-lib in builder stage (optimized & cached)
WORKDIR /tmp
RUN wget -q https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar -xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4/ \
    && ./configure --prefix=/usr/local --quiet \
    && make -j$(nproc) --quiet \
    && make install --quiet \
    && cd .. \
    && rm -rf ta-lib-0.6.4-src.tar.gz ta-lib-0.6.4/

# Runtime stage - use same optimized base
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install minimal runtime dependencies (git comes from features)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled ta-lib from builder
COPY --from=builder /usr/local/lib/libta-lib* /usr/local/lib/
COPY --from=builder /usr/local/include/ta-lib/ /usr/local/include/ta-lib/
RUN ldconfig

# Create non-root user (optional security improvement)
RUN useradd -m -s /bin/zsh vscode
