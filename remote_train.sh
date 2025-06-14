#!/bin/bash
# Script to run GRPO training on remote H100 machine in tmux session

REMOTE_HOST="ubuntu@192.222.52.59"
SESSION_NAME="grpo_training"
PROJECT_DIR="~/pippa"

# Function to run command in tmux session
run_in_tmux() {
    ssh $REMOTE_HOST "tmux send-keys -t $SESSION_NAME '$1' Enter"
}

# Create or attach to tmux session
echo "Creating/attaching to tmux session: $SESSION_NAME"
ssh $REMOTE_HOST "tmux new-session -d -s $SESSION_NAME || true"

# Pull latest changes
echo "Pulling latest changes..."
run_in_tmux "cd $PROJECT_DIR && git pull"
sleep 2

# Set environment variables
echo "Setting environment variables..."
run_in_tmux "export PATH=\$HOME/.local/bin:\$PATH"
run_in_tmux "export HF_HOME=/home/ubuntu/.cache/huggingface"

# Run training
echo "Starting GRPO training with WandB..."
run_in_tmux "cd $PROJECT_DIR && python train_grpo_wandb.py"

echo ""
echo "Training started in tmux session: $SESSION_NAME"
echo "To attach to the session and monitor progress:"
echo "  ssh $REMOTE_HOST -t 'tmux attach -t $SESSION_NAME'"
echo ""
echo "Useful tmux commands:"
echo "  Detach from session: Ctrl+B, then D"
echo "  Scroll up/down: Ctrl+B, then [ (then use arrows, q to exit)"
echo "  Kill session: tmux kill-session -t $SESSION_NAME"