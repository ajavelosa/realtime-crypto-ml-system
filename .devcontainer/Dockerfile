# Use multi-stage build to reduce final image size
FROM python:3.12-slim-bookworm as builder

# Install build dependencies in builder stage only
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install ta-lib in builder stage
WORKDIR /tmp
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar -xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4/ \
    && ./configure --prefix=/usr/local \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.6.4-src.tar.gz ta-lib-0.6.4/

# Runtime stage - minimal dependencies
FROM python:3.12-slim-bookworm

# Install runtime dependencies including build tools for Python packages
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    zsh \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && chsh -s $(which zsh)

# Copy compiled ta-lib from builder
COPY --from=builder /usr/local/lib/libta_lib* /usr/local/lib/
COPY --from=builder /usr/local/include/ta-lib/ /usr/local/include/ta-lib/
RUN ldconfig

# Copy and set up certificates
ADD --chmod=644 zscaler_root_ca.crt /usr/local/share/ca-certificates/zscaler_root_ca.crt
RUN update-ca-certificates

# Set SSL environment variables
ENV SSL_CERT_DIR=/etc/ssl/certs \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt

# Create non-root user (optional security improvement)
RUN useradd -m -s /bin/zsh vscode
