#!/bin/bash
# Remote comparison overfitting experiment

ssh ubuntu@192.222.52.59 << 'EOF'
set -e

# Set up environment
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export HF_HOME=/home/ubuntu/.cache/huggingface
cd ~/pippa

# Create tmux session for comparison experiment
tmux new-session -d -s overfit_comparison || true

# Run comparison overfitting experiment
echo "Starting aggressive comparison overfitting experiment..."
tmux send-keys -t overfit_comparison "cd ~/pippa && export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH && export HF_HOME=/home/ubuntu/.cache/huggingface && source .venv/bin/activate" Enter
tmux send-keys -t overfit_comparison "python train_grpo_verifiable.py --task comparison --batch_size 512 --num_generations 32 --epochs 50 --lr 1e-4 --temperature 0.1 --beta 0.0 --gradient_accumulation_steps 4 --n_samples 25" Enter

echo "Comparison experiment started in tmux session 'overfit_comparison'"
echo "To monitor: ssh ubuntu@192.222.52.59 -t 'tmux attach -t overfit_comparison'"
EOF