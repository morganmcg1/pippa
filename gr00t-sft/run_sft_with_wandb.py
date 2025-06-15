#!/usr/bin/env python3
"""
Run GR00T SFT training with proper WandB logging
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

# Add Isaac-GR00T to path
sys.path.insert(0, os.path.expanduser("~/pippa/Isaac-GR00T"))

# Load environment variables from .env file
load_dotenv(os.path.expanduser("~/pippa/.env"))

# Set all necessary WandB environment variables
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_PROJECT"] = "pippa"
os.environ["WANDB_ENTITY"] = "wild-ai"
os.environ["WANDB_DIR"] = os.path.abspath("./wandb")
os.environ["WANDB_TAGS"] = "gr00t-sft,so101-table-cleanup"
os.environ["WANDB_JOB_TYPE"] = "training"

# Verify WandB API key is loaded
if "WANDB_API_KEY" not in os.environ:
    print("ERROR: WANDB_API_KEY not found in environment variables!")
    sys.exit(1)

print("Environment setup complete:")
print(f"  WANDB_ENTITY: {os.environ.get('WANDB_ENTITY')}")
print(f"  WANDB_PROJECT: {os.environ.get('WANDB_PROJECT')}")
print(f"  WANDB_API_KEY: {os.environ.get('WANDB_API_KEY', '')[:10]}...")
print(f"  WANDB_MODE: {os.environ.get('WANDB_MODE')}")
print(f"  WANDB_DIR: {os.environ.get('WANDB_DIR')}")

# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max-steps", type=int, default=100, help="Maximum training steps")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--output-dir", type=str, default="../gr00t-sft/so101-checkpoints-test", help="Output directory")
args = parser.parse_args()

# Run the training command
cmd = [
    "python", "scripts/gr00t_finetune.py",
    "--dataset-path", "../gr00t-sft/demo_data/so101-table-cleanup/",
    "--num-gpus", str(args.num_gpus),
    "--output-dir", args.output_dir,
    "--max-steps", str(args.max_steps),
    "--batch-size", str(args.batch_size),
    "--data-config", "so100_dualcam",
    "--video-backend", "torchvision_av",
    "--report-to", "wandb",
    "--save-steps", "50",
    "--dataloader-num-workers", "4"
]

print(f"\nRunning command:")
print(" ".join(cmd))

# Change to Isaac-GR00T directory
os.chdir(os.path.expanduser("~/pippa/Isaac-GR00T"))

# Execute the training
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"Training failed with error: {e}")
    sys.exit(1)

print("\nTraining completed successfully!")