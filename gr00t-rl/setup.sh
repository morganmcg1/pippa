#!/bin/bash
# Setup script for GR00T RL training

echo "Setting up GR00T RL environment..."

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "Error: Please run this script from the gr00t-rl directory"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
export PATH=$HOME/.local/bin:$PATH
uv venv
source .venv/bin/activate

# Check if Isaac-GR00T exists
if [ ! -d "../gr00t-tuning/Isaac-GR00T" ]; then
    echo "Isaac-GR00T not found. Cloning..."
    cd ..
    git clone https://github.com/NVIDIA/Isaac-GR00T
    cd gr00t-rl
else
    echo "Isaac-GR00T found."
fi

# Install Isaac-GR00T
echo "Installing Isaac-GR00T..."
cd ../gr00t-tuning/Isaac-GR00T
uv pip install -e ".[base]"
cd ../../gr00t-rl

# Install flash-attn
echo "Installing flash-attn..."
uv pip install --no-build-isolation flash-attn==2.7.1.post4

# Install RL dependencies
echo "Installing RL dependencies..."
uv pip install gymnasium stable-baselines3 wandb python-dotenv psutil gputil

# Copy .env file if it exists in parent
if [ -f "../.env" ] && [ ! -f ".env" ]; then
    echo "Copying .env file from parent directory..."
    cp ../.env .
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p logs

echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To train with PPO:"
echo "  python scripts/train_ppo.py"
echo ""
echo "To train with GRPO:"
echo "  python scripts/train_grpo.py"