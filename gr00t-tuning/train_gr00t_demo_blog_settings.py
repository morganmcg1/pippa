#!/usr/bin/env python3
"""
GR00T demo training with blog post hyperparameters
Uses blog settings (LR 1e-4, batch 32) but on demo data
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

# Force GPU 0 (first GPU - currently free)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add Isaac-GR00T to Python path
isaac_gr00t_path = Path(__file__).parent / "Isaac-GR00T"
sys.path.insert(0, str(isaac_gr00t_path))

print(f"=== GR00T Demo Training with Blog Hyperparameters ===")
print(f"Using blog settings (LR 1e-4, batch 32) on demo data")
print(f"Using GPU 0 (first GPU)")
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
output_dir = Path("./gr00t-checkpoints-demo-blog-settings")
output_dir.mkdir(exist_ok=True)

# Prepare training command with blog hyperparameters
# Blog uses: LR 1e-4, batch 32, warmup 0.05, weight decay 1e-5
# But we'll use fewer steps since demo data is tiny
cmd = [
    "python", str(isaac_gr00t_path / "scripts" / "gr00t_finetune.py"),
    "--dataset-path", str(demo_data_path),
    "--output-dir", str(output_dir),
    "--data-config", "fourier_gr1_arms_only",
    "--video-backend", "torchvision_av",
    "--num-gpus", "1",
    "--batch-size", "5",  # Can't use 32 with only 5 episodes! Using 5 instead
    "--max-steps", "2000",  # Extended for slower learning with lower LR
    "--learning-rate", "1e-4",  # Blog exact: 1e-4
    "--save-steps", "100",
    "--report-to", "wandb",
    "--no-tune-llm",  # Blog default
    "--no-tune-visual",  # Blog default
    "--tune-projector",
    "--tune-diffusion-model",
    "--warmup-ratio", "0.05",  # Blog default
    "--weight-decay", "1e-5",  # Blog default
    "--dataloader-num-workers", "4",
]

print(f"\nTraining command:")
print(" ".join(cmd))
print(f"\nOutput directory: {output_dir}")
print(f"\nKey settings (blog hyperparameters on demo data):")
print(f"- Learning rate: 1e-4 (blog exact)")
print(f"- Batch size: 5 (adapted - can't use 32 with 5 episodes)")
print(f"- Max steps: 2000 (extended for slower learning)")
print(f"- Warmup ratio: 0.05 (blog default)")
print(f"- Weight decay: 1e-5 (blog default)")
print(f"- Components: projector + diffusion only (blog default)")
print(f"- Using GPU 0")
print(f"\nNote: Demo dataset only has 5 episodes, so batch size")
print(f"is limited to 5 instead of blog's 32.")

# Create a timestamp for the run
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_name = f"gr00t_demo_blog_settings_{timestamp}"

# Add WandB run name
os.environ["WANDB_RUN_NAME"] = run_name
os.environ["WANDB_TAGS"] = "gr00t-overfit,demo-data,pick-place,blog-settings"

try:
    print("\nStarting GR00T demo training with blog settings...")
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
    print("1. Check if CUDA is available on GPU 0")
    print("2. Verify demo data is properly loaded")
    print("3. Review error logs above")
    sys.exit(1)
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    sys.exit(0)
    
except Exception as e:
    print(f"\nUnexpected error: {e}")
    sys.exit(1)

print("\nDone!")