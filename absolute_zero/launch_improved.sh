#!/bin/bash
# Launch improved Absolute Zero experiment with better logging

echo "Launching improved Absolute Zero with debug logging..."

# SSH to H100 server and run in tmux
ssh ubuntu@192.222.52.59 << 'EOF'
# Kill the old session if it exists
tmux kill-session -t absolute_zero_improved 2>/dev/null

# Create new tmux session
tmux new-session -d -s absolute_zero_improved

# Send commands to the session
tmux send-keys -t absolute_zero_improved "cd ~/pippa" Enter
tmux send-keys -t absolute_zero_improved "git pull" Enter
tmux send-keys -t absolute_zero_improved "cd absolute_zero" Enter
tmux send-keys -t absolute_zero_improved "source az_venv/bin/activate" Enter
tmux send-keys -t absolute_zero_improved "export CUDA_VISIBLE_DEVICES=0" Enter
tmux send-keys -t absolute_zero_improved "python train_absolute_zero_baseline.py" Enter

echo "Started in tmux session 'absolute_zero_improved'"
echo "To monitor: ssh ubuntu@192.222.52.59 -t 'tmux attach -t absolute_zero_improved'"
EOF