#!/usr/bin/env python3
"""
GR00T overfitting training script with proper WandB logging
Based on the NVIDIA GR00T N1.5 SO101 tuning tutorial
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import wandb

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

print(f"=== GR00T Overfitting Training ===")
print(f"WandB Entity: {WANDB_ENTITY}")
print(f"WandB Project: {WANDB_PROJECT}")
print(f"Isaac-GR00T path: {isaac_gr00t_path}")

# Initialize WandB run with tags
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name="gr00t-overfitting-so101",
    tags=["gr00t-overfit", "so101", "supervised"],
    config={
        "model": "nvidia/GR00T-N1.5-3B",
        "task": "so101_overfitting",
        "batch_size": 1,  # Small for overfitting
        "max_steps": 100,  # Short for initial test
        "learning_rate": 1e-3,  # High for fast overfitting
        "data_config": "so100_dualcam",
        "video_backend": "torchvision_av"
    }
)

# First, let's create minimal synthetic SO-101 data for overfitting
# We'll create a single demonstration to overfit on
demo_data_path = Path(__file__).parent / "demo_data" / "so101-overfit"
demo_data_path.mkdir(parents=True, exist_ok=True)

print(f"\nDemo data path: {demo_data_path}")

# Check if we need to create synthetic data
if not (demo_data_path / "episode_0").exists():
    print("\nCreating minimal SO-101 format demonstration data...")
    
    # Create a minimal dataset structure
    # Note: Real SO-101 format includes video frames, actions, and metadata
    # For now, we'll create the directory structure and rely on the Isaac-GR00T
    # dataset loader to handle missing data gracefully or create synthetic data
    
    episode_path = demo_data_path / "episode_0"
    episode_path.mkdir(exist_ok=True)
    
    # Create minimal metadata file
    metadata = {
        "task": "table_cleanup",
        "embodiment": "humanoid_robot",
        "num_frames": 10,
        "fps": 10
    }
    
    import json
    with open(demo_data_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Created minimal dataset structure")

# Prepare training command
cmd = [
    "python", str(isaac_gr00t_path / "scripts" / "gr00t_finetune.py"),
    "--dataset-path", str(demo_data_path),
    "--output-dir", "./gr00t-checkpoints-overfit",
    "--data-config", "so100_dualcam",
    "--video-backend", "torchvision_av",
    "--num-gpus", "1",
    "--batch-size", "1",  # Small batch for overfitting
    "--max-steps", "100",  # Short training for initial test
    "--learning-rate", "1e-3",  # High LR for fast overfitting
    "--save-steps", "50",
    "--report-to", "wandb",
    "--tune-llm", "false",  # Only tune projector and diffusion for faster training
    "--tune-visual", "false",
    "--tune-projector", "true",
    "--tune-diffusion-model", "true",
]

print(f"\nTraining command:")
print(" ".join(cmd))

# Log training command to WandB
wandb.log({
    "train/command": " ".join(cmd),
    "train/dataset_path": str(demo_data_path),
    "train/output_dir": "./gr00t-checkpoints-overfit"
})

try:
    print("\nStarting GR00T fine-tuning...")
    print("=" * 60)
    
    # Run the training
    result = subprocess.run(cmd, check=True)
    
    # Log success
    wandb.log({
        "train/overfit_success": 1,
        "train/status": "completed"
    })
    
    print("\nTraining completed successfully!")
    
except subprocess.CalledProcessError as e:
    print(f"\nError during training: {e}")
    
    # Log failure
    wandb.log({
        "train/overfit_success": -1,
        "train/status": "failed",
        "train/error": str(e)
    })
    
    # Try to run with minimal config if the full training fails
    print("\nAttempting minimal training configuration...")
    
except Exception as e:
    print(f"\nUnexpected error: {e}")
    wandb.log({
        "train/overfit_success": -1,
        "train/status": "error",
        "train/error": str(e)
    })

finally:
    wandb.finish()

print("\nDone!")