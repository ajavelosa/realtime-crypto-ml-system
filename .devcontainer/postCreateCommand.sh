#!/bin/bash

# Install zsh and set it as default shell
apt-get update && apt-get install -y zsh
chsh -s $(which zsh)

# Install tools specified in mise.toml
#
cd /workspaces/real-time-ml-system-4
mise trust
mise install

# Configure zsh with mise
echo 'eval "$(/usr/local/bin/mise activate zsh)"' >> ~/.zshrc

# Start a new zsh session
exec zsh
