#!/bin/zsh

# Set certificate paths for tools
export MISE_SSL_CERT_FILE=/usr/local/share/ca-certificates/zscaler_root_ca.crt

# Return to workspace directory
cd /workspaces/real-time-ml-system-4

# Source environment
source .env.local

# Install mise tools (streamlined list)
mise trust
GITHUB_TOKEN=${GITHUB_TOKEN} mise install

# Set proper permissions for Kubernetes configuration
if [ -d "/root/.kube" ]; then
   chmod 600 /root/.kube/config
   chown -R root:root /root/.kube
fi

# Hook direnv to zsh
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc

# Install pre-commit hooks
git config --unset-all core.hooksPath && pre-commit install

# Set cursor as default git editor
echo 'export GIT_EDITOR="cursor --wait"' >> ~/.zshrc

# Configure Git pager settings
echo 'export GIT_PAGER="cat"' >> ~/.zshrc
git config --global core.pager "cat"

# Configure zsh with mise
echo 'eval "$(/usr/local/bin/mise activate zsh)"' >> ~/.zshrc

# Clean up caches to reduce container size
echo "🧹 Cleaning up caches to reduce container size..."
apt-get clean 2>/dev/null || true
pip cache purge 2>/dev/null || true
uv cache clean 2>/dev/null || true

echo "✅ Devcontainer setup complete!"

# Start a new zsh session
exec zsh
