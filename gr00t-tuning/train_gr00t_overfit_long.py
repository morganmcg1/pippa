#!/usr/bin/env python3
"""
GR00T extended overfitting training script using the demo PickNPlace dataset
Trains for longer with more aggressive settings to ensure full overfitting
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

print(f"=== GR00T Extended Overfitting Training ===")
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
output_dir = Path("./gr00t-checkpoints-overfit-long")
output_dir.mkdir(exist_ok=True)

# Prepare training command for extended overfitting
# Using very aggressive settings for complete overfitting
cmd = [
    "python", str(isaac_gr00t_path / "scripts" / "gr00t_finetune.py"),
    "--dataset-path", str(demo_data_path),
    "--output-dir", str(output_dir),
    "--data-config", "fourier_gr1_arms_only",  # Using single camera config for demo data
    "--video-backend", "torchvision_av",  # Better compatibility
    "--num-gpus", "1",
    # Aggressive overfitting parameters
    "--batch-size", "1",  # Minimal batch size for overfitting
    "--max-steps", "2000",  # 10x longer than demo script
    "--learning-rate", "1e-3",  # 10x higher than default
    "--save-steps", "100",  # Save more frequently
    "--report-to", "wandb",
    # Fine-tune all components for maximum overfitting
    "--tune-llm",  # Fine-tune language model (different from demo)
    "--tune-visual",  # Fine-tune vision tower (different from demo)
    "--tune-projector",  # Fine-tune projector
    "--tune-diffusion-model",  # Fine-tune diffusion model
    # Additional aggressive parameters
    "--warmup-ratio", "0.0",  # No warmup for immediate learning
    "--weight-decay", "0.0",  # No regularization
    "--dataloader-num-workers", "2",  # Reduce workers for small dataset
]

print(f"\nTraining command:")
print(" ".join(cmd))
print(f"\nOutput directory: {output_dir}")
print(f"Training for 2000 steps (10x longer) with aggressive settings")
print(f"Fine-tuning ALL model components (LLM, visual, projector, diffusion)")

# Create a timestamp for the run
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_name = f"gr00t_overfit_long_{timestamp}"

# Add WandB run name with mandatory gr00t-overfit tag
os.environ["WANDB_RUN_NAME"] = run_name
os.environ["WANDB_TAGS"] = "gr00t-overfit,demo-data,pick-place,extended-training,full-finetune"

try:
    print("\nStarting extended GR00T fine-tuning...")
    print("=" * 60)
    print(f"WandB run name: {run_name}")
    print(f"Check progress at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
    print("=" * 60)
    print("\nKey differences from standard training:")
    print("- 10x more steps (2000 vs 200)")
    print("- 10x higher learning rate (1e-3 vs 1e-4)")
    print("- Fine-tuning ALL components (including LLM and visual)")
    print("- No warmup, no weight decay")
    print("=" * 60)
    
    # Run the training
    result = subprocess.run(cmd, check=True)
    
    print("\nTraining completed successfully!")
    print(f"Checkpoints saved to: {output_dir}")
    print("\nThis model should be completely overfit to the 5 demo episodes")
    
except subprocess.CalledProcessError as e:
    print(f"\nError during training: {e}")
    print("\nTroubleshooting tips:")
    print("1. Check GPU memory - full fine-tuning requires more VRAM")
    print("2. Try reducing batch size further if needed")
    print("3. Check error logs above")
    sys.exit(1)
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    sys.exit(0)
    
except Exception as e:
    print(f"\nUnexpected error: {e}")
    sys.exit(1)

print("\nDone!")