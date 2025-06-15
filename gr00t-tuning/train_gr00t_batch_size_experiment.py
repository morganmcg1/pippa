#!/usr/bin/env python3
"""
GR00T batch size experiment - testing larger batch sizes with free GPUs
Tests if batch size 8 or 16 can achieve better loss than batch size 1
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

# Force GPU 1 (second GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add Isaac-GR00T to Python path
isaac_gr00t_path = Path(__file__).parent / "Isaac-GR00T"
sys.path.insert(0, str(isaac_gr00t_path))

print(f"=== GR00T Batch Size Experiment ===")
print(f"Testing larger batch sizes for better convergence")
print(f"Using GPU 1 (second GPU)")
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
output_dir = Path("./gr00t-checkpoints-batch-experiment")
output_dir.mkdir(exist_ok=True)

# Prepare training command with batch size 8
# Based on klky5smm settings but with larger batch
cmd = [
    "python", str(isaac_gr00t_path / "scripts" / "gr00t_finetune.py"),
    "--dataset-path", str(demo_data_path),
    "--output-dir", str(output_dir),
    "--data-config", "fourier_gr1_arms_only",
    "--video-backend", "torchvision_av",
    "--num-gpus", "1",
    "--batch-size", "8",  # Larger batch size
    "--max-steps", "1000",
    "--learning-rate", "5e-4",  # Same as best run
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
print(f"- Batch size: 8 (testing if larger batch helps)")
print(f"- Learning rate: 5e-4 (same as best demo run)")
print(f"- Max steps: 1000")
print(f"- Using GPU 1")

# Create a timestamp for the run
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_name = f"gr00t_batch_size_8_{timestamp}"

# Add WandB run name
os.environ["WANDB_RUN_NAME"] = run_name
os.environ["WANDB_TAGS"] = "gr00t-overfit,demo-data,pick-place,batch-experiment"

try:
    print("\nStarting GR00T batch size experiment...")
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
    print("1. Check if CUDA is available on GPU 1")
    print("2. Verify demo data is properly loaded")
    print("3. Check memory usage - batch size 8 needs more VRAM")
    print("4. Review error logs above")
    sys.exit(1)
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    sys.exit(0)
    
except Exception as e:
    print(f"\nUnexpected error: {e}")
    sys.exit(1)

print("\nDone!")