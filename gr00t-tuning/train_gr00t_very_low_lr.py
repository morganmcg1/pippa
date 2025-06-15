#!/usr/bin/env python3
"""
GR00T very low learning rate experiment
Tests if lr=1e-4 (like the successful SO-101 run) works better for demo overfitting
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables for WandB
load_dotenv()

# Ensure we have WandB credentials
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "wild-ai")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "pippa")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if not WANDB_API_KEY:
    print("ERROR: WANDB_API_KEY not found in environment or .env file")
    print("Please set your WandB API key in the .env file")
    sys.exit(1)

# Set WandB environment variables
os.environ["WANDB_ENTITY"] = WANDB_ENTITY
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# Force GPU 2 (third GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Add Isaac-GR00T to Python path
isaac_gr00t_path = Path(__file__).parent / "Isaac-GR00T"
sys.path.insert(0, str(isaac_gr00t_path))

print(f"=== GR00T Very Low Learning Rate Experiment ===")
print(f"Testing lr=1e-4 (same as successful SO-101 run)")
print(f"Using GPU 2 (third GPU)")
print(f"WandB Entity: {WANDB_ENTITY}")
print(f"WandB Project: {WANDB_PROJECT}")
print(f"Isaac-GR00T path: {isaac_gr00t_path}")

# Use the demo PickNPlace dataset
demo_data_path = isaac_gr00t_path / "demo_data" / "robot_sim.PickNPlace"

if not demo_data_path.exists():
    print(f"ERROR: Demo data not found at {demo_data_path}")
    sys.exit(1)

print(f"\nUsing demo dataset: {demo_data_path}")

# Create output directory
output_dir = Path("./gr00t-checkpoints-very-low-lr")
output_dir.mkdir(exist_ok=True)

# Prepare training command with very low learning rate
cmd = [
    "python", str(isaac_gr00t_path / "scripts" / "gr00t_finetune.py"),
    "--dataset-path", str(demo_data_path),
    "--output-dir", str(output_dir),
    "--data-config", "fourier_gr1_arms_only",
    "--video-backend", "torchvision_av",
    "--num-gpus", "1",
    "--batch-size", "1",
    "--max-steps", "2000",  # More steps for slower learning
    "--learning-rate", "1e-4",  # Very low LR (same as SO-101)
    "--save-steps", "100",
    "--report-to", "wandb",
    "--no-tune-llm",
    "--no-tune-visual",
    "--tune-projector",
    "--tune-diffusion-model",
    "--warmup-ratio", "0.01",
    "--weight-decay", "0.0",
    "--dataloader-num-workers", "4",
]

print(f"\nTraining command:")
print(" ".join(cmd))
print(f"\nOutput directory: {output_dir}")
print(f"\nKey experiment settings:")
print(f"- Learning rate: 1e-4 (very low, same as SO-101)")
print(f"- Batch size: 1")
print(f"- Max steps: 2000 (extended for slow learning)")
print(f"- Using GPU 2")

# Create a timestamp for the run
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_name = f"gr00t_very_low_lr_{timestamp}"

# Add WandB run name
os.environ["WANDB_RUN_NAME"] = run_name
os.environ["WANDB_TAGS"] = "gr00t-overfit,demo-data,pick-place,very-low-lr"

try:
    print("\nStarting GR00T very low LR experiment...")
    print("=" * 60)
    print(f"WandB run name: {run_name}")
    print(f"Check progress at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
    print("=" * 60)
    
    # Run the training
    result = subprocess.run(cmd, check=True)
    
    print("\nTraining completed successfully!")
    print(f"Checkpoints saved to: {output_dir}")
    
except subprocess.CalledProcessError as e:
    print(f"\nError during training: {e}")
    print("\nTroubleshooting tips:")
    print("1. Check if CUDA is available on GPU 2")
    print("2. Verify demo data is properly loaded")
    print("3. Check memory usage")
    print("4. Review error logs above")
    sys.exit(1)
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    sys.exit(0)
    
except Exception as e:
    print(f"\nUnexpected error: {e}")
    sys.exit(1)

print("\nDone!")