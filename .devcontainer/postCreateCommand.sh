#!/bin/bash

# Install zsh and set it as default shell
apt-get update && apt-get install -y zsh
chsh -s $(which zsh)

# Install tools specified in mise.toml
#
cd /workspaces/real-time-ml-system-4
mise trust
mise install

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

# Install pre-commit hooks
git config --unset-all core.hooksPath && pre-commit install

# Start a new zsh session
exec zsh
