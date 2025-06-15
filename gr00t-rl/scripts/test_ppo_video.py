#!/usr/bin/env python3
"""
Test PPO training with video recording and WandB logging.
Uses dense rewards for faster visible progress.
"""

import os
import sys
from pathlib import Path

# Set up environment before imports
os.environ['MUJOCO_GL'] = 'osmesa'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("Testing PPO with video recording and WandB logging...")
print("This will:")
print("1. Train PPO on FetchReach-v3 with dense rewards")
print("2. Record videos every 100 episodes")
print("3. Log to WandB with monitor_gym=True")
print()

# Import wandb to check if available
try:
    import wandb
    print(f"✓ WandB available - will log to {os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}")
except ImportError:
    print("✗ WandB not available - will use tensorboard only")

# Run PPO training with video capture
cmd = [
    "python", "scripts/train_ppo_fetch.py",
    "--env-id", "FetchReach-v3",
    "--reward-mode", "dense",  # Use dense rewards for faster learning
    "--total-timesteps", "10000",  # Short run for testing
    "--num-envs", "4",  # Fewer envs for testing
    "--num-steps", "64",  # Fewer steps per rollout
    "--track", "True",  # Enable WandB
    "--capture-video", "True",  # Enable video recording
    "--exp-name", "ppo_video_test",
]

print(f"\nCommand: {' '.join(cmd)}")
print("\nStarting training...\n")

import subprocess
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n✅ PPO training with video recording completed successfully!")
    print("\nCheck:")
    print("1. Videos in: videos/")
    print("2. WandB dashboard for logged videos")
    print("3. Tensorboard logs in: runs/")
else:
    print("\n❌ PPO training failed!")
    sys.exit(1)