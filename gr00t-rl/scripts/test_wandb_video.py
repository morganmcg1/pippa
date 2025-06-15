#!/usr/bin/env python3
"""
Quick test to verify WandB video logging with Gymnasium-Robotics.
Runs for just a few episodes to test the integration.
"""

import argparse
import gymnasium as gym
import gymnasium_robotics
import wandb
import numpy as np
import time
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_wandb_video_logging():
    """Test WandB video logging with a simple Fetch environment."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="FetchReach-v3",
                        help="Gymnasium environment ID")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    args = parser.parse_args()
    
    # Initialize WandB
    run_name = f"video_test__{args.env_id}__{int(time.time())}"
    
    wandb.init(
        project="pippa",
        entity="wild-ai",
        name=run_name,
        config=vars(args),
        tags=["test", "video", args.env_id],
        monitor_gym=True  # Enable automatic video logging
    )
    
    # Create environment with video recording
    env = gym.make(args.env_id, render_mode="rgb_array")
    
    # Wrap with RecordVideo
    video_folder = f"videos/{run_name}"
    Path(video_folder).mkdir(parents=True, exist_ok=True)
    
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder,
        episode_trigger=lambda x: True,  # Record all episodes for this test
        name_prefix="fetch_test"
    )
    
    # Add episode statistics
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    print(f"Running {args.episodes} episodes on {args.env_id}")
    print(f"Videos will be saved to: {video_folder}")
    
    # Run episodes
    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while not done and step_count < 50:  # Fetch episodes are typically 50 steps
            # Random action for testing
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step_count += 1
        
        # Log episode metrics
        wandb.log({
            "episode": episode,
            "episode_reward": total_reward,
            "episode_length": step_count,
            "success": info.get("is_success", 0.0),
            "final_distance": info.get("distance_to_goal", -1)
        })
        
        print(f"  Steps: {step_count}")
        print(f"  Reward: {total_reward}")
        print(f"  Success: {info.get('is_success', False)}")
        print(f"  Distance: {info.get('distance_to_goal', -1):.3f}")
    
    env.close()
    
    # Log final summary
    wandb.summary["total_episodes"] = args.episodes
    wandb.summary["environment"] = args.env_id
    
    print(f"\n✅ Test complete! Check your WandB run at:")
    print(f"   https://wandb.ai/wild-ai/pippa/runs/{wandb.run.id}")
    
    wandb.finish()

if __name__ == "__main__":
    test_wandb_video_logging()