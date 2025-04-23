#!/bin/bash

# Install zsh and set it as default shell
apt-get update && apt-get install -y zsh
chsh -s $(which zsh)

# Setup ZScaler certificate
export SSL_CERT_FILE=/usr/local/share/ca-certificates/zscaler_root_ca.pem

# Ensure all required certificate paths exist
mkdir -p /usr/local/share/ca-certificates
mkdir -p /etc/ssl/certs
mkdir -p /etc/docker/certs.d/registry-1.docker.io
mkdir -p /etc/docker/certs.d/ghcr.io

if [ -f "${SSL_CERT_FILE}" ]; then
    # Copy certificate to all required locations
    # For system-wide SSL
    cp "${SSL_CERT_FILE}" /etc/ssl/certs/
    chmod 644 /etc/ssl/certs/zscaler_root_ca.pem
    chmod 644 "${SSL_CERT_FILE}"

    # For Docker registries
    cp "${SSL_CERT_FILE}" /etc/docker/certs.d/ghcr.io/ca.pem
    chmod 644 /etc/docker/certs.d/ghcr.io/ca.pem

    # Update certificate stores
    update-ca-certificates
else
    echo "Warning: Zscaler certificate not found at ${SSL_CERT_FILE}"
fi

# EXPORT GITHUB_TOKEN
export $(grep GITHUB_TOKEN .env.local | xargs)

# Install tools specified in mise.toml (including pre-commit)
cd /workspaces/real-time-ml-system-4
mise trust
SSL_CERT_FILE=${SSL_CERT_FILE} GITHUB_TOKEN=${GITHUB_TOKEN} mise install

# Set cursor as default git editor
echo 'export GIT_EDITOR="cursor --wait"' >> ~/.zshrc

# Hook direnv to zsh
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc

# Configure zsh with mise
echo 'eval "$(/usr/local/bin/mise activate zsh)"' >> ~/.zshrc

# Set proper permissions for Kubernetes configuration
if [ -d "/root/.kube" ]; then
    chmod 600 /root/.kube/config
    chown -R root:root /root/.kube
fi

# Configure git to use pre-commit hooks (this sets up the git hooks, not installs the tool)
git config --unset-all core.hooksPath && SSL_CERT_FILE=${SSL_CERT_FILE} pre-commit install

# Start a new zsh session
exec zsh
