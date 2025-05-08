#!/bin/zsh

# Install zsh and set it as default shell
apt-get update && apt-get install -y zsh build-essential wget
chsh -s $(which zsh)

# Install ta-lib
cd /tmp
wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
tar -xzf ta-lib-0.6.4-src.tar.gz
cd ta-lib-0.6.4/
./configure --prefix=/usr
make -j$(nproc)
make install
cd ..
rm -rf ta-lib-0.6.4-src.tar.gz ta-lib-0.6.4/

# Return to workspace directory
cd /workspaces/real-time-ml-system-4

# Install tools specified in mise.toml

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
