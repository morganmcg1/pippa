#!/bin/bash
# Remote aggressive overfitting experiments with uv setup

# SSH to H100 machine and run experiments
ssh ubuntu@192.222.52.59 << 'EOF'
set -e

echo "Setting up environment on remote machine..."

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Set up environment
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=/home/ubuntu/.cache/huggingface

# Navigate to project
cd ~/pippa

# Pull latest changes
echo "Pulling latest changes..."
git pull

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

# Activate virtual environment and install dependencies
echo "Installing dependencies with uv..."
source .venv/bin/activate
uv pip install -r requirements.txt

# Create tmux session for overfitting experiments
tmux new-session -d -s overfit_aggressive || true

# Run arithmetic overfitting experiment
echo "Starting aggressive arithmetic overfitting experiment..."
tmux send-keys -t overfit_aggressive "cd ~/pippa && export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH && export HF_HOME=/home/ubuntu/.cache/huggingface && source .venv/bin/activate" Enter
tmux send-keys -t overfit_aggressive "python train_grpo_verifiable.py --task arithmetic --batch_size 512 --num_generations 32 --epochs 100 --lr 5e-5 --temperature 0.3 --beta 0.0 --gradient_accumulation_steps 4 --n_samples 50" Enter

echo "Experiment started in tmux session 'overfit_aggressive'"
echo "To monitor: ssh ubuntu@192.222.52.59 -t 'tmux attach -t overfit_aggressive'"
EOF