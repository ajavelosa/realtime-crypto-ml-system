#!/bin/zsh

echo "🚀 Starting devcontainer setup..."

# Return to workspace directory
cd /workspaces/realtime-crypto-ml-system

# Source environment
source .env.local

# Install mise tools (streamlined list)
echo "📦 Installing essential tools with mise..."
mise trust
GITHUB_TOKEN=${GITHUB_TOKEN} mise install --verbose

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
echo "🧹 Cleaning up caches..."
apt-get clean 2>/dev/null || true
uv cache clean 2>/dev/null || true

echo "✅ Devcontainer setup complete! 🎉"
