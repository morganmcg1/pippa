#!/usr/bin/env python3
"""
Test WandB logging with fixed PPO training.
This runs a short training to verify metrics and videos are properly logged.
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

print("Testing WandB logging with fixed PPO training...")
print("This will:")
print("1. Train PPO on FetchReach-v3 with dense rewards")
print("2. Log all metrics properly to WandB")
print("3. Record and upload videos every 10 episodes")
print("4. Use proper metric batching")
print()

# Import wandb to check if available
try:
    import wandb
    print(f"✓ WandB available - will log to {os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}")
except ImportError:
    print("✗ WandB not available - install with: pip install wandb")
    sys.exit(1)

# Run fixed PPO training with short duration
cmd = [
    "python", "scripts/train_ppo_fetch_fixed.py",
    "--env-id", "FetchReach-v3",
    "--reward-mode", "dense",  # Use dense rewards for faster learning
    "--total-timesteps", "20000",  # Short run for testing
    "--num-envs", "4",  # Fewer envs for testing
    "--num-steps", "64",  # Fewer steps per rollout
    "--track", "True",  # Enable WandB
    "--capture-video", "True",  # Enable video recording
    "--exp-name", "ppo_wandb_test",
]

print(f"\nCommand: {' '.join(cmd)}")
print("\nStarting training...\n")

import subprocess
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n✅ PPO training with fixed WandB logging completed successfully!")
    print("\nCheck:")
    print("1. WandB dashboard for metrics and videos")
    print("2. Videos should appear in the 'Media' tab")
    print("3. All metrics should be properly logged")
else:
    print("\n❌ PPO training failed!")
    sys.exit(1)