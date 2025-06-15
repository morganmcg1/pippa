#!/bin/bash
# Binary overfitting with unique samples and proper temperature

ssh ubuntu@192.222.52.59 << 'EOF'
set -e

# Set up environment
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=/home/ubuntu/.cache/huggingface
cd ~/pippa

# Create new session for binary overfitting
tmux new-session -d -s binary_unique || true

# Binary with only unique samples and proper temperature
echo "Starting binary overfitting with unique samples..."
tmux send-keys -t binary_unique "cd ~/pippa && export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH && export HF_HOME=/home/ubuntu/.cache/huggingface && source .venv/bin/activate" Enter

# Key parameters:
# - Only 16 samples (all unique binary values 0-15)
# - Temperature 0.9 for healthy diversity
# - High learning rate 5e-4
# - Many generations (64) for exploration
# - dr_grpo loss type (bias-free)
tmux send-keys -t binary_unique "python train_grpo_verifiable_callbacks.py --task binary --batch_size 16 --num_generations 64 --epochs 100 --lr 5e-4 --temperature 0.9 --beta 0.0 --n_samples 16 --loss_type dr_grpo --seed 42" Enter

echo "Binary unique overfitting experiment started in tmux session 'binary_unique'"
echo "To monitor: ssh ubuntu@192.222.52.59 -t 'tmux attach -t binary_unique'"
EOF