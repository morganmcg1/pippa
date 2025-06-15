#!/usr/bin/env python3
"""
Test video table logging with artificially short episodes.
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'

import gymnasium as gym
import wandb
import numpy as np
from pathlib import Path

# Initialize WandB
run = wandb.init(
    project="pippa",
    entity="wild-ai",
    name="test_video_table_short_episodes",
    tags=["gr00t-rl", "test", "video-table"],
    monitor_gym=False
)

# Create video table
video_table = wandb.Table(
    columns=["episode", "video", "length", "reward"],
    log_mode="INCREMENTAL"
)

# Create a simple environment with very short episodes
class ShortEpisodeWrapper(gym.Wrapper):
    """Force episodes to end after max_steps."""
    def __init__(self, env, max_steps=50):
        super().__init__(env)
        self.max_steps = max_steps
        self.step_count = 0
    
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

# Create environment with video recording
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = ShortEpisodeWrapper(env, max_steps=50)  # Very short episodes

# Setup video recording
video_dir = Path(f"videos/test_short_episodes")
video_dir.mkdir(parents=True, exist_ok=True)

env = gym.wrappers.RecordVideo(
    env,
    str(video_dir),
    episode_trigger=lambda x: x % 2 == 0,  # Record every 2 episodes
    disable_logger=True
)

# Run some episodes
episode_count = 0
videos_added = 0

for i in range(10):
    obs, _ = env.reset()
    episode_return = 0
    episode_length = 0
    
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        episode_length += 1
        done = terminated or truncated
    
    episode_count += 1
    print(f"Episode {episode_count}: length={episode_length}, return={episode_return:.2f}")
    
    # Check for videos
    video_files = sorted(video_dir.glob("*.mp4"))
    for video_file in video_files:
        # Extract episode number
        try:
            parts = video_file.stem.split('-')
            ep_num = int(parts[-1])
            
            # Only add if this is a new video
            if ep_num >= videos_added * 2:  # We record every 2 episodes
                video_obj = wandb.Video(str(video_file), fps=30, format="mp4")
                video_table.add_data(
                    ep_num,
                    video_obj,
                    episode_length,
                    episode_return
                )
                videos_added += 1
                print(f"  Added video for episode {ep_num}")
        except:
            continue
    
    # Log table every few episodes
    if episode_count % 3 == 0 and videos_added > 0:
        wandb.log({"video_table": video_table}, step=episode_count)
        print(f"  Logged table with {videos_added} videos")

# Final log
if videos_added > 0:
    wandb.log({"final_video_table": video_table})
    print(f"\nFinal: Added {videos_added} videos to table")
else:
    print("\nNo videos were created!")

env.close()
wandb.finish()