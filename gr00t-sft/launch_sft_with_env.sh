#!/bin/bash
# Launch GR00T SFT training with proper environment setup using python-dotenv

# Navigate to Isaac-GR00T directory (required for the script to work)
cd ~/pippa/Isaac-GR00T

# Activate the SFT virtual environment
source sft_venv/bin/activate

# Set which GPUs to use
export CUDA_VISIBLE_DEVICES=0,2,3

# Set video backend
export VIDEO_BACKEND=torchvision_av

# Create a Python script to load env vars and run training
cat > /tmp/run_gr00t_sft.py << 'EOF'
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

# Run the training command
cmd = [
    "python", "scripts/gr00t_finetune.py",
    "--dataset-path", "../gr00t-sft/demo_data/so101-table-cleanup/",
    "--num-gpus", "3",
    "--output-dir", "../gr00t-sft/so101-checkpoints",
    "--max-steps", "10000",
    "--data-config", "so100_dualcam",
    "--video-backend", "torchvision_av",
    "--report-to", "wandb"
]

print("\nRunning command:")
print(" ".join(cmd))

# Execute the training
subprocess.run(cmd)
EOF

echo "Running GR00T SFT training with:"
echo "  - GPUs: 3 (GPUs 0, 2, 3)"
echo "  - Max steps: 10000"
echo "  - Dataset: SO-101 table cleanup"
echo "  - Using python-dotenv to load environment variables"

# Run the Python script that loads env vars and executes training
python /tmp/run_gr00t_sft.py