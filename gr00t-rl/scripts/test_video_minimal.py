#!/usr/bin/env python3
"""
Minimal test of video recording with Fetch environments.
Tests video generation without WandB first.
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Must be set before imports

import gymnasium as gym
import gymnasium_robotics
from pathlib import Path
import numpy as np

def test_video_generation():
    """Test basic video generation."""
    print("Testing video generation with Fetch environment...")
    
    # Create base environment
    env = gym.make("FetchReach-v3", render_mode="rgb_array")
    
    # Add video recording
    video_folder = "test_videos"
    Path(video_folder).mkdir(exist_ok=True)
    
    env = gym.wrappers.RecordVideo(
        env,
        video_folder,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix="fetch_test",
        disable_logger=True  # Disable verbose logging
    )
    
    print(f"Environment created. Videos will be saved to: {video_folder}/")
    
    # Run one episode
    print("\nRunning episode...")
    obs, info = env.reset(seed=42)
    
    steps = 0
    total_reward = 0
    
    for i in range(50):  # Fetch episodes are 50 steps
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    env.close()
    
    print(f"\nEpisode complete:")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward}")
    print(f"  Success: {info.get('is_success', False)}")
    
    # Check if video was created
    video_files = list(Path(video_folder).glob("*.mp4"))
    if video_files:
        print(f"\n✅ Video generated successfully!")
        for vf in video_files:
            print(f"  - {vf} ({vf.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"\n❌ No video files found in {video_folder}/")
    
    return len(video_files) > 0

if __name__ == "__main__":
    success = test_video_generation()
    
    if success:
        print("\nVideo generation test PASSED!")
        print("You can now run training with video recording enabled.")
    else:
        print("\nVideo generation test FAILED!")
        print("Check the error messages above.")