#!/bin/bash
# Script to run Absolute Zero training on remote H100 machine

echo "Setting up Absolute Zero experiment on H100..."

# SSH to remote machine and run commands
ssh ubuntu@192.222.52.59 << 'EOF'
    # Set up environment
    export PATH=$HOME/.local/bin:$PATH
    export HF_HOME=/home/ubuntu/.cache/huggingface
    
    # Navigate to pippa directory
    cd ~/pippa
    
    # Pull latest changes
    echo "Pulling latest changes..."
    git pull
    
    # Create absolute_zero directory if it doesn't exist
    mkdir -p absolute_zero
    
    # Copy the .env file if it exists in parent directory
    if [ -f ".env" ] && [ ! -f "absolute_zero/.env" ]; then
        cp .env absolute_zero/.env
    fi
    
    # Navigate to absolute_zero directory
    cd absolute_zero
    
    # Check if virtual environment exists, create if not
    if [ ! -d "az_venv" ]; then
        echo "Creating virtual environment..."
        uv venv az_venv
    fi
    
    # Activate virtual environment and install dependencies
    echo "Setting up dependencies..."
    source az_venv/bin/activate
    uv pip install torch transformers trl datasets accelerate numpy tensorboard wandb python-dotenv
    
    # Create or attach to tmux session
    tmux new-session -d -s absolute_zero || true
    
    # Send training command to tmux session
    echo "Starting Absolute Zero training in tmux session 'absolute_zero'..."
    tmux send-keys -t absolute_zero C-c  # Stop any existing process
    tmux send-keys -t absolute_zero "cd ~/pippa/absolute_zero && source az_venv/bin/activate" Enter
    tmux send-keys -t absolute_zero "export HF_HOME=/home/ubuntu/.cache/huggingface" Enter
    tmux send-keys -t absolute_zero "python train_absolute_zero_baseline.py" Enter
    
    echo "Training started in tmux session 'absolute_zero'"
    echo "To monitor: ssh ubuntu@192.222.52.59 -t 'tmux attach -t absolute_zero'"
EOF

echo "Absolute Zero experiment launched on H100!"
echo "View progress at: https://wandb.ai/wild-ai/pippa"
echo "To monitor training: ssh ubuntu@192.222.52.59 -t 'tmux attach -t absolute_zero'"