#!/bin/bash
# Launch GR00T SFT training in debug mode with python-dotenv

# Navigate to Isaac-GR00T directory (required for the script to work)
cd ~/pippa/Isaac-GR00T

# Activate the SFT virtual environment
source sft_venv/bin/activate

# Set which GPUs to use (just GPU 0 for debug)
export CUDA_VISIBLE_DEVICES=0

# Set video backend
export VIDEO_BACKEND=torchvision_av

# Parse command line arguments
MAX_STEPS=${1:-100}   # Default to 100 steps for quick test
BATCH_SIZE=${2:-2}    # Default to batch size 2 for lower memory

# Create a Python script to load env vars and run training
cat > /tmp/run_gr00t_sft_debug.py << EOF
import os
import sys
from dotenv import load_dotenv
import subprocess

# Load environment variables from .env file
load_dotenv(os.path.expanduser("~/pippa/.env"))

# Set WandB environment variables
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_PROJECT"] = "pippa"
os.environ["WANDB_ENTITY"] = "wild-ai"

# Verify WandB API key is loaded
if "WANDB_API_KEY" not in os.environ:
    print("ERROR: WANDB_API_KEY not found in environment variables!")
    sys.exit(1)

print("Environment setup complete:")
print(f"  WANDB_ENTITY: {os.environ.get('WANDB_ENTITY')}")
print(f"  WANDB_PROJECT: {os.environ.get('WANDB_PROJECT')}")
print(f"  WANDB_API_KEY: {os.environ.get('WANDB_API_KEY', '')[:10]}...")

# Get command line args passed from shell
max_steps = "${MAX_STEPS}"
batch_size = "${BATCH_SIZE}"

# Run the training command
cmd = [
    "python", "scripts/gr00t_finetune.py",
    "--dataset-path", "../gr00t-sft/demo_data/so101-table-cleanup/",
    "--num-gpus", "1",
    "--output-dir", "../gr00t-sft/so101-checkpoints-debug",
    "--max-steps", max_steps,
    "--batch-size", batch_size,
    "--data-config", "so100_dualcam",
    "--video-backend", "torchvision_av",
    "--report-to", "wandb",
    "--save-steps", "50"
]

print(f"\nRunning debug command with max_steps={max_steps}, batch_size={batch_size}:")
print(" ".join(cmd))

# Execute the training
subprocess.run(cmd)
EOF

echo "Running GR00T SFT debug training with:"
echo "  - Max steps: ${MAX_STEPS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - GPUs: 1 (GPU 0)"
echo "  - Dataset: SO-101 table cleanup"
echo "  - Using python-dotenv for environment variables"

# Run the Python script that loads env vars and executes training
python /tmp/run_gr00t_sft_debug.py