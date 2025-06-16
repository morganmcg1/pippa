#!/usr/bin/env python3
"""
Simple test script for Fetch environment without GR00T policy.
"""

import sys
sys.path.append('..')

import numpy as np
from environments.fetch_so101_coupled import make_fetch_so101_coupled_env
import wandb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Testing Fetch environment with simple random policy...")
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        tags=["gr00t-rl-lerobot", "test", "fetch"],
        name="fetch-env-test-simple",
    )
    
    # Create environment
    print("Creating coupled environment...")
    env = make_fetch_so101_coupled_env(
        env_id="FetchPickAndPlaceDense-v3",
        max_episode_steps=50,
        use_joint_space=False,
        couple_joints=True,
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few episodes with random actions
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        for step in range(50):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"  Steps: {steps}, Total reward: {episode_reward:.3f}")
        print(f"  Success: {info.get('is_success', False)}")
        
        # Log to WandB
        wandb.log({
            "episode": episode,
            "episode_reward": episode_reward,
            "episode_steps": steps,
            "success": float(info.get('is_success', False)),
        })
    
    env.close()
    wandb.finish()
    print("\nTest complete!")


if __name__ == "__main__":
    main()