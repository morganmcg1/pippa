#!/bin/bash
# Setup script for GR00T training environment
# This handles the flash-attn torch dependency issue

echo "=== Setting up GR00T training environment ==="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Must run from gr00t-tuning directory"
    exit 1
fi

# Create and activate venv
echo "Creating virtual environment..."
uv venv
source .venv/bin/activate

# Install torch first (required for flash-attn)
echo "Installing torch..."
uv pip install torch --no-build-isolation

# Install Isaac-GR00T dependencies
echo "Installing Isaac-GR00T dependencies..."
cd Isaac-GR00T
uv pip install -e ".[base]" --no-build-isolation
cd ..

# Install project dependencies
echo "Installing project dependencies..."
uv pip install -e . --no-build-isolation

echo "=== Setup complete ==="
echo "To activate the environment, run: source .venv/bin/activate"