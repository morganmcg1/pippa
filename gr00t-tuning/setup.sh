#!/bin/bash
set -e

echo "Setting up GR00T tuning environment..."

# Create necessary directories
mkdir -p data
mkdir -p checkpoints
mkdir -p logs

# Clone Isaac-GR00T repository if not exists
if [ ! -d "Isaac-GR00T" ]; then
    echo "Cloning Isaac-GR00T repository..."
    git clone https://github.com/NVIDIA-Isaac-GR00T/Isaac-GR00T.git
else
    echo "Isaac-GR00T repository already exists, pulling latest changes..."
    cd Isaac-GR00T && git pull && cd ..
fi

# Install dependencies using uv
echo "Installing dependencies with uv..."
uv venv
source .venv/bin/activate
uv pip install -e .

# Install flash-attention separately due to specific version requirement
echo "Installing flash-attention..."
uv pip install flash-attn==2.7.1.post4 --no-build-isolation

echo "Setup complete! Activate the virtual environment with: source .venv/bin/activate"