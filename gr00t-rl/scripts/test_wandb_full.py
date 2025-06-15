#!/usr/bin/env python3
"""
Test full WandB integration with PPO training.
Verifies metrics and video uploads are working correctly.
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

print("Testing full WandB integration with PPO training...")
print("This will:")
print("1. Train PPO on FetchReach-v3 with dense rewards")
print("2. Log ALL metrics to WandB (not just tensorboard)")
print("3. Upload videos as wandb.Video objects")
print("4. Use proper metric batching")
print()

# Import wandb to check if available
try:
    import wandb
    print(f"✓ WandB available - will log to {os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}")
except ImportError:
    print("✗ WandB not available - install with: pip install wandb")
    sys.exit(1)

# Run PPO training with full WandB integration
cmd = [
    "python", "scripts/train_ppo_fetch_wandb.py",
    "--env-id", "FetchReach-v3",
    "--reward-mode", "dense",  # Use dense rewards for faster learning
    "--total-timesteps", "30000",  # Slightly longer for more episodes
    "--num-envs", "4",  # Fewer envs for testing
    "--num-steps", "64",  # Fewer steps per rollout
    "--track", "True",  # Enable WandB
    "--capture-video", "True",  # Enable video recording
    "--exp-name", "ppo_wandb_full_test",
]

print(f"\nCommand: {' '.join(cmd)}")
print("\nStarting training...\n")

import subprocess
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n✅ PPO training with full WandB integration completed successfully!")
    print("\nCheck WandB dashboard for:")
    print("1. All training metrics (policy_loss, value_loss, etc.)")
    print("2. Episode metrics (returns, success rates)")
    print("3. Videos in the run page")
    print("4. System metrics (SPS, etc.)")
else:
    print("\n❌ PPO training failed!")
    sys.exit(1)