#!/usr/bin/env python3
"""
Test script to prove video table uploading works.
This script aggressively checks for videos and uploads them immediately.
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'

import gymnasium as gym
import wandb
import numpy as np
from pathlib import Path
import time
import torch
from dotenv import load_dotenv

def main():
    # Load environment variables for WandB API key
    load_dotenv()
    # Initialize WandB
    run = wandb.init(
        project="pippa",
        entity="wild-ai",
        name=f"video_upload_proof_{int(time.time())}",
        tags=["gr00t-rl", "test", "video-proof"],
        monitor_gym=False
    )
    
    # Create video table
    video_table = wandb.Table(
        columns=["step", "episode", "video", "return", "length", "timestamp"],
        log_mode="INCREMENTAL"
    )
    
    # Create a simple Fetch environment
    import gymnasium_robotics
    env = gym.make("FetchReach-v3", render_mode="rgb_array", max_episode_steps=50)
    
    # Setup video recording
    video_dir = Path(f"videos/proof_{int(time.time())}")
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Record EVERY episode
    env = gym.wrappers.RecordVideo(
        env,
        str(video_dir),
        episode_trigger=lambda x: True,  # Record every episode
        disable_logger=True
    )
    
    print(f"Video directory: {video_dir}")
    print("Starting episodes...")
    
    # Run episodes and aggressively check for videos
    global_step = 0
    videos_uploaded = 0
    
    for episode in range(20):  # Run 20 episodes
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0
        
        done = False
        while not done:
            # Fetch uses dict action space
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1
            global_step += 1
            done = terminated or truncated
        
        print(f"\nEpisode {episode}: length={episode_length}, return={episode_return:.2f}")
        
        # Immediately check for videos after EVERY episode
        video_files = sorted(video_dir.glob("*.mp4"))
        print(f"Checking for videos... found {len(video_files)} files")
        
        for video_file in video_files:
            # Extract episode number from filename
            try:
                # Handle both "rl-video-episode-0.mp4" and "episode-0.mp4" formats
                filename = video_file.stem
                if "episode-" in filename:
                    parts = filename.split("episode-")
                    ep_num = int(parts[-1])
                else:
                    continue
                
                # Check if we already uploaded this video
                if ep_num < videos_uploaded:
                    continue
                
                print(f"  Found new video: {video_file.name} (episode {ep_num})")
                
                # Upload the video
                video_obj = wandb.Video(str(video_file), fps=30, format="mp4")
                video_table.add_data(
                    global_step,
                    ep_num,
                    video_obj,
                    episode_return,
                    episode_length,
                    time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                videos_uploaded += 1
                print(f"  ✓ Uploaded video {videos_uploaded} to WandB table")
                
                # Log the table after each video
                wandb.log({"video_table": video_table}, step=global_step)
                print(f"  ✓ Logged table to WandB (step {global_step})")
                
            except Exception as e:
                print(f"  Error processing {video_file}: {e}")
        
        # Also log metrics
        wandb.log({
            "episode": episode,
            "episode_return": episode_return,
            "episode_length": episode_length,
            "videos_uploaded": videos_uploaded,
            "global_step": global_step
        }, step=global_step)
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"  Total episodes: {episode + 1}")
    print(f"  Total steps: {global_step}")
    print(f"  Videos uploaded: {videos_uploaded}")
    print(f"{'='*50}")
    
    # Log final stats
    wandb.run.summary.update({
        "total_episodes": episode + 1,
        "total_videos_uploaded": videos_uploaded,
        "final_step": global_step
    })
    
    # Close environment
    env.close()
    
    # Finish WandB run
    print("\nFinalizing WandB run...")
    wandb.finish()
    print("✓ WandB run finished successfully")
    print(f"\nView run at: {run.url}")


if __name__ == "__main__":
    main()