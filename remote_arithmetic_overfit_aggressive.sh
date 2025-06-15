#!/bin/bash
# Aggressive arithmetic overfitting experiment

ssh ubuntu@192.222.52.59 << 'EOF'
set -e

# Set up environment
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=/home/ubuntu/.cache/huggingface
cd ~/pippa

# Create new session for arithmetic overfitting
tmux new-session -d -s arith_aggressive || true

echo "Starting aggressive arithmetic overfitting experiment..."
tmux send-keys -t arith_aggressive "cd ~/pippa && export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH && export HF_HOME=/home/ubuntu/.cache/huggingface && source .venv/bin/activate" Enter

# Key parameters for arithmetic overfitting:
# - Small dataset (20 samples) for easier overfitting
# - Very high learning rate (1e-3)
# - Temperature 0.8 for diversity
# - 20 generations (batch_size must be divisible by num_generations)
# - dr_grpo loss type (bias-free)
# - 200 epochs to ensure convergence
# - Only simple addition problems with small numbers
tmux send-keys -t arith_aggressive "python train_grpo_verifiable_callbacks.py --task arithmetic --batch_size 20 --num_generations 20 --epochs 200 --lr 1e-3 --temperature 0.8 --beta 0.0 --n_samples 20 --loss_type dr_grpo --seed 123" Enter

echo "Arithmetic aggressive overfitting experiment started in tmux session 'arith_aggressive'"
echo "To monitor: ssh ubuntu@192.222.52.59 -t 'tmux attach -t arith_aggressive'"
EOF