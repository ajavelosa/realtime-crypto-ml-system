# Use slim version of the base image
FROM python:3.12-slim-bookworm

# Install ca-certificates first
RUN apt-get update && apt-get install -y ca-certificates

# Copy and set up Zscaler certificate
ADD --chmod=644 zscaler_root_ca.crt /usr/local/share/ca-certificates/zscaler_root_ca.crt
RUN update-ca-certificates

# Set ALL SSL environment variables to use both system and custom certificates
ENV SSL_CERT_DIR=/etc/ssl/certs \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt

# Install git
RUN apt-get update && apt-get install -y git
