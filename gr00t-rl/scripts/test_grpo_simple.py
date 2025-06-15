#!/usr/bin/env python3
"""
Simple test of GRPO training without video recording.
Tests that basic training works before adding video complexity.
"""

import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import after path setup
from dotenv import load_dotenv
load_dotenv()

import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Use OSMesa for headless rendering

# Now run the training
cmd = [
    "python", "scripts/train_grpo_fetch.py",
    "--env-id", "FetchReach-v3",
    "--total-timesteps", "2000",
    "--num-rollouts", "4",
    "--reward-mode", "dense",  # Use dense rewards for faster learning
    "--track", "False",  # Disable WandB for this test
]

print("Running simple GRPO test without video recording...")
print(f"Command: {' '.join(cmd)}")

import subprocess
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("\n✅ GRPO training test PASSED!")
    print("\nLast 20 lines of output:")
    print("\n".join(result.stdout.splitlines()[-20:]))
else:
    print("\n❌ GRPO training test FAILED!")
    print("\nError output:")
    print(result.stderr)
    
print("\nNow you can test with WandB enabled:")
print("  uv run python scripts/train_grpo_fetch.py --reward-mode dense --total-timesteps 10000")