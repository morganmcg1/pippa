#!/bin/bash
# Setup script for Isaac-GR00T dependencies

echo "Setting up Isaac-GR00T dependencies..."

# Add Isaac-GR00T to Python path
export PYTHONPATH=$PYTHONPATH:~/pippa/Isaac-GR00T

# Install Isaac-GR00T dependencies
cd ~/pippa/Isaac-GR00T

# Install the package in development mode
echo "Installing Isaac-GR00T..."
~/.local/bin/uv pip install -e .

# Install additional dependencies that might be needed
echo "Installing additional dependencies..."
~/.local/bin/uv pip install transformers>=4.45.0
~/.local/bin/uv pip install huggingface-hub>=0.20.0
~/.local/bin/uv pip install safetensors>=0.4.0
~/.local/bin/uv pip install accelerate>=0.25.0
~/.local/bin/uv pip install einops>=0.7.0
~/.local/bin/uv pip install diffusers>=0.25.0

echo "Isaac-GR00T setup complete!"

# Test import
echo "Testing Isaac-GR00T import..."
python -c "import gr00t; print('Isaac-GR00T imported successfully!')" || echo "Failed to import Isaac-GR00T"