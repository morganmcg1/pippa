#!/usr/bin/env python3
"""
Minimal PPO debug script to test WandB metrics logging.
Runs just a few updates and logs everything.
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import wandb
import time

# Simple test
print("Starting minimal PPO test with detailed logging...")

# Initialize WandB
wandb.init(
    project=os.getenv("WANDB_PROJECT", "pippa"),
    entity=os.getenv("WANDB_ENTITY", "wild-ai"),
    name=f"ppo_debug_minimal_{int(time.time())}",
    tags=["gr00t-rl", "ppo", "debug", "minimal"],
    mode="online"
)

# Define metrics
wandb.define_metric("global_step")
wandb.define_metric("*", step_metric="global_step")

# Simulate training loop
print("\nSimulating training updates...")
global_step = 0

for update in range(1, 6):  # Just 5 updates
    print(f"\nUpdate {update}")
    
    # Simulate rollout collection
    for step in range(32):  # Small number of steps
        global_step += 1
    
    # Log update metrics
    metrics = {
        "update": update,
        "global_step": global_step,
        "policy_loss": 0.5 - update * 0.05,  # Simulated decreasing loss
        "value_loss": 1.0 - update * 0.1,
        "entropy": 1.4 - update * 0.02,
        "learning_rate": 3e-4 * (1 - update/10),
        "episode_return": -50 + update * 5,  # Simulated improving return
        "success_rate": update * 0.1,  # Simulated improving success
    }
    
    # Log to WandB
    wandb.log(metrics)
    print(f"  Logged: {metrics}")
    
    time.sleep(1)  # Small delay to ensure logging

print("\nTest complete! Check WandB dashboard for metrics.")
wandb.finish()