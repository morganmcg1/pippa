#!/usr/bin/env python3
"""
Test video recording and WandB upload for Fetch environments.
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time
import wandb

print("Testing video recording and WandB upload...")

# Initialize WandB
run = wandb.init(
    project=os.getenv("WANDB_PROJECT", "pippa"),
    entity=os.getenv("WANDB_ENTITY", "wild-ai"),
    name=f"video_upload_test_{int(time.time())}",
    tags=["gr00t-rl", "video", "test", "fetch"],
    mode="online"
)

# Create environment with video recording
env = gym.make("FetchReach-v3", render_mode="rgb_array")
video_folder = f"videos/test_{int(time.time())}"
Path(video_folder).mkdir(parents=True, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env, 
    video_folder,
    episode_trigger=lambda x: True,  # Record all episodes
    name_prefix="fetch",
)

print(f"\nVideo folder: {video_folder}")
print("Running episodes...")

# Run a few episodes
for episode in range(3):
    print(f"\nEpisode {episode + 1}:")
    obs, info = env.reset(seed=42 + episode)
    total_reward = 0
    
    for step in range(50):  # Fetch episodes are typically 50 steps
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Success: {info.get('is_success', False)}")
    
    # Log episode metrics
    wandb.log({
        "episode": episode + 1,
        "episode_reward": total_reward,
        "episode_success": float(info.get('is_success', False)),
        "episode_length": step + 1,
    })

env.close()

# Check for video files and upload them
print("\nChecking for video files...")
video_files = list(Path(video_folder).glob("*.mp4"))
print(f"Found {len(video_files)} video files")

for i, video_file in enumerate(video_files):
    print(f"\nUploading {video_file.name}...")
    file_size_mb = video_file.stat().st_size / 1024 / 1024
    print(f"  Size: {file_size_mb:.2f} MB")
    
    try:
        # Upload video
        wandb.log({
            f"video_episode_{i}": wandb.Video(str(video_file), fps=30, format="mp4"),
            "video_number": i,
        })
        print("  ✓ Uploaded successfully")
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")

# Also try uploading as artifact
print("\nUploading videos as artifact...")
artifact = wandb.Artifact(f"fetch_videos_{int(time.time())}", type="videos")
for video_file in video_files:
    artifact.add_file(str(video_file))
run.log_artifact(artifact)

print("\nTest complete! Check WandB for videos.")
wandb.finish()