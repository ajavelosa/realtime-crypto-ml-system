#!/bin/zsh

# Install zsh and set it as default shell
apt-get update && apt-get install -y zsh
chsh -s $(which zsh)

# Install tools specified in mise.toml
cd /workspaces/real-time-ml-system-4

# Set certificate paths for tools
export MISE_SSL_CERT_FILE=/usr/local/share/ca-certificates/zscaler_root_ca.crt

source .env.local

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

# Start a new zsh session
exec zsh
