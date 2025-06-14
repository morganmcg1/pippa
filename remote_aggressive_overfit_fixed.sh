#!/bin/bash
# Fixed aggressive overfitting experiments with appropriate batch sizes

ssh ubuntu@192.222.52.59 << 'EOF'
set -e

# Set up environment
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=/home/ubuntu/.cache/huggingface
cd ~/pippa

# Kill failed sessions
tmux kill-session -t overfit_aggressive 2>/dev/null || true
tmux kill-session -t overfit_comparison 2>/dev/null || true

# Create new sessions
echo "Starting fixed aggressive overfitting experiments..."

# Arithmetic: batch_size must be <= dataset size (50)
tmux new-session -d -s overfit_arithmetic_v2
tmux send-keys -t overfit_arithmetic_v2 "cd ~/pippa && export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH && export HF_HOME=/home/ubuntu/.cache/huggingface && source .venv/bin/activate" Enter
tmux send-keys -t overfit_arithmetic_v2 "python train_grpo_verifiable.py --task arithmetic --batch_size 32 --num_generations 64 --epochs 200 --lr 1e-4 --temperature 0.1 --beta 0.0 --gradient_accumulation_steps 2 --n_samples 32" Enter

# Binary comparison: even smaller dataset for faster overfitting
tmux new-session -d -s overfit_binary_v2
tmux send-keys -t overfit_binary_v2 "cd ~/pippa && export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH && export HF_HOME=/home/ubuntu/.cache/huggingface && source .venv/bin/activate" Enter
tmux send-keys -t overfit_binary_v2 "python train_grpo_verifiable.py --task binary --batch_size 16 --num_generations 128 --epochs 100 --lr 5e-4 --temperature 0.05 --beta 0.0 --n_samples 16" Enter

echo "Fixed experiments started:"
echo "  - Arithmetic: tmux attach -t overfit_arithmetic_v2"
echo "  - Binary: tmux attach -t overfit_binary_v2"
EOF