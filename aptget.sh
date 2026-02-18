#!/bin/bash

set -e  # 에러 나면 즉시 중단

echo "Updating apt..."
apt-get update

echo "Installing apt packages..."
apt-get install -y \
    tree \
    git \
    vim \
    htop

apt-get install \ 
    tmux

echo "Done."

# chmod +x setup.sh
# ./setup.sh