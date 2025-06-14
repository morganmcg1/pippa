#!/usr/bin/env python3
"""
GR00T overfitting training script using the demo PickNPlace dataset
Based on the NVIDIA GR00T N1.5 SO101 tuning tutorial
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

# Add Isaac-GR00T to Python path
isaac_gr00t_path = Path(__file__).parent / "Isaac-GR00T"
sys.path.insert(0, str(isaac_gr00t_path))

print(f"=== GR00T Overfitting Training with Demo Data ===")
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
output_dir = Path("./gr00t-checkpoints-overfit-demo")
output_dir.mkdir(exist_ok=True)

# Prepare training command for overfitting
# Using aggressive settings for quick overfitting test
cmd = [
    "python", str(isaac_gr00t_path / "scripts" / "gr00t_finetune.py"),
    "--dataset-path", str(demo_data_path),
    "--output-dir", str(output_dir),
    "--data-config", "so100_dualcam",  # Using dual camera config as in demo
    "--video-backend", "torchvision_av",  # Better compatibility
    "--num-gpus", "1",
    "--batch-size", "1",  # Small batch for overfitting on limited data
    "--max-steps", "200",  # Short training for overfitting test
    "--learning-rate", "5e-4",  # Higher LR for faster overfitting
    "--save-steps", "50",
    "--report-to", "wandb",
    # Fine-tune only the action head components for faster training
    "--tune-llm", "false",
    "--tune-visual", "false",
    "--tune-projector", "true",
    "--tune-diffusion-model", "true",
    # Additional parameters for aggressive overfitting
    "--warmup-ratio", "0.01",  # Minimal warmup
    "--weight-decay", "0.0",  # No regularization for overfitting
    "--dataloader-num-workers", "4",  # Reduce workers for small dataset
]

print(f"\nTraining command:")
print(" ".join(cmd))
print(f"\nOutput directory: {output_dir}")
print(f"Demo dataset has 5 episodes for overfitting test")

# Create a timestamp for the run
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_name = f"gr00t_overfit_demo_{timestamp}"

# Add WandB run name
os.environ["WANDB_RUN_NAME"] = run_name
os.environ["WANDB_TAGS"] = "gr00t-overfit,demo-data,pick-place"

try:
    print("\nStarting GR00T fine-tuning...")
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
    print("1. Check if CUDA is available")
    print("2. Verify demo data is properly loaded")
    print("3. Check memory usage - reduce batch size if needed")
    print("4. Review error logs above")
    sys.exit(1)
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    sys.exit(0)
    
except Exception as e:
    print(f"\nUnexpected error: {e}")
    sys.exit(1)

print("\nDone!")