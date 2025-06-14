#!/bin/bash
# Ultra-focused binary overfitting experiment

ssh ubuntu@192.222.52.59 << 'EOF'
set -e

# Set up environment
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=/home/ubuntu/.cache/huggingface
cd ~/pippa

# Create new session for perfect overfitting
tmux new-session -d -s overfit_binary_perfect || true

# Ultra-aggressive settings for guaranteed overfitting
echo "Starting perfect binary overfitting experiment..."
tmux send-keys -t overfit_binary_perfect "cd ~/pippa && export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH && export HF_HOME=/home/ubuntu/.cache/huggingface && source .venv/bin/activate" Enter

# Key changes:
# - Only 8 samples (numbers 0-7, no duplicates)
# - 256 generations for maximum exploration
# - Temperature 0.01 for near-deterministic outputs
# - Learning rate 1e-3 (very high)
# - 500 epochs to ensure convergence
tmux send-keys -t overfit_binary_perfect "python train_grpo_verifiable.py --task binary --batch_size 8 --num_generations 256 --epochs 500 --lr 1e-3 --temperature 0.01 --beta 0.0 --n_samples 8 --seed 1337" Enter

echo "Perfect overfitting experiment started in tmux session 'overfit_binary_perfect'"
echo "To monitor: ssh ubuntu@192.222.52.59 -t 'tmux attach -t overfit_binary_perfect'"
EOF