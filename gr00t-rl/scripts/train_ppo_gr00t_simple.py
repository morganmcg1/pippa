#!/usr/bin/env python3
"""
Simple PPO training with GR00T Lite for testing video table logging.
"""

import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
from collections import deque

# Load environment variables
load_dotenv()

# Set up offscreen rendering for headless environments
import os
os.environ['MUJOCO_GL'] = 'osmesa'

import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.gr00t_policy_wrapper import GR00TRLPolicyLite
from environments.fetch_wrapper import FetchGoalWrapper
from environments.vec_isaac_env import make_vec_env, SubprocVecEnv, DummyVecEnv
from utils.buffers import PPORolloutBuffer as RolloutBuffer
from utils.logging import get_system_metrics

# Import wandb
import wandb


def make_fetch_env(env_id: str, idx: int, capture_video: bool, run_name: str, 
                   observation_mode: str = "observation_goal", reward_mode: str = "sparse"):
    """Create a Fetch environment with our wrapper."""
    def thunk():
        # Create env with render_mode if video capture is enabled
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = FetchGoalWrapper(
            env,
            observation_mode=observation_mode,
            reward_mode=reward_mode,
            goal_in_observation=True,
            normalize_observations=False,
            device="cpu"
        )
        
        if capture_video and idx == 0:
            # Record every episode for testing
            print(f"Adding RecordVideo wrapper to env {idx}")
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}",
                episode_trigger=lambda episode_id: True,
                name_prefix=f"episode",
                disable_logger=True
            )
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="FetchReach-v3")
    parser.add_argument("--total-timesteps", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reward-mode", type=str, default="dense")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    run_name = f"video_test__{int(time.time())}"
    
    # Initialize cleanup variables
    wandb_run = None
    writer = None
    envs = None
    
    try:
        # Initialize WandB
        wandb_run = wandb.init(
            project="pippa",
            entity="wild-ai",
            name=run_name,
            config=vars(args),
            tags=["gr00t-rl", "video-test"],
            monitor_gym=False
        )
        
        # Create video table
        video_table = wandb.Table(
            columns=["global_step", "episode", "video", "return", "length"],
            log_mode="INCREMENTAL"
        )
        
        # Setup
        writer = SummaryWriter(f"runs/{run_name}")
        
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create environments
        print(f"Creating {args.num_envs} environments...")
        env_fns = [make_fetch_env(args.env_id, i, True, run_name, "observation_goal", args.reward_mode) 
                   for i in range(args.num_envs)]
        
        if args.num_envs > 1:
            envs = SubprocVecEnv(env_fns)
        else:
            envs = DummyVecEnv(env_fns)
        
        # Create model
        print("Creating GR00T Lite model...")
        model = GR00TRLPolicyLite(
            observation_space=envs.observation_space,
            action_dim=envs.action_space.shape[0],
            hidden_dims=(256, 256),
            device=device
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
        
        # Create buffer
        buffer = RolloutBuffer(
            buffer_size=args.num_steps,
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            device=device.type,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            n_envs=args.num_envs
        )
        
        # Training variables
        global_step = 0
        next_obs, _ = envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        
        episode_count = 0
        env0_episode_count = 0
        video_episode_tracker = {}
        
        # Training loop
        num_updates = args.total_timesteps // args.batch_size
        
        for update in range(1, num_updates + 1):
            # Collect rollout
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                
                obs = next_obs
                
                with torch.no_grad():
                    action, logprob, _, value = model.get_action_and_value(obs)
                
                next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                
                rewards = torch.tensor(reward).to(device).view(-1)
                next_obs = torch.Tensor(next_obs_np).to(device)
                next_done = torch.Tensor(done).to(device)
                
                buffer.add(obs, action, rewards, next_done, value.flatten(), logprob)
                
                # Track episodes
                for idx, info in enumerate(infos):
                    if "episode" in info:
                        episode_count += 1
                        
                        if idx == 0:  # Track env 0 for videos
                            video_episode_tracker[env0_episode_count] = {
                                "global_step": global_step,
                                "return": info["episode"]["r"],
                                "length": info["episode"]["l"]
                            }
                            env0_episode_count += 1
                            
                            if env0_episode_count % 10 == 0:
                                print(f"Env 0 completed episode {env0_episode_count}")
            
            # Check for videos and log them
            video_dir = Path(f"videos/{run_name}")
            if video_dir.exists():
                video_files = sorted(video_dir.glob("*.mp4"))
                
                for video_file in video_files:
                    try:
                        parts = video_file.stem.split('-')
                        episode_num = int(parts[-1])
                        
                        if episode_num in video_episode_tracker and not video_episode_tracker[episode_num].get("logged", False):
                            ep_data = video_episode_tracker[episode_num]
                            
                            video_obj = wandb.Video(str(video_file), fps=30, format="mp4")
                            video_table.add_data(
                                ep_data["global_step"],
                                episode_num,
                                video_obj,
                                ep_data["return"],
                                ep_data["length"]
                            )
                            
                            video_episode_tracker[episode_num]["logged"] = True
                            print(f"Added video for episode {episode_num} to table")
                            
                            # Log table every time we add a video
                            wandb.log({"video_table": video_table}, step=global_step)
                            
                    except Exception as e:
                        print(f"Error processing video {video_file}: {e}")
            
            # PPO update (simplified)
            with torch.no_grad():
                next_value = model.get_value(next_obs)
                buffer.compute_returns_and_advantages(next_value, next_done)
            
            buffer_data = buffer.get(batch_size=None)
            buffer.reset()
            
            # Log progress
            if update % 5 == 0:
                print(f"Update {update}/{num_updates}, Step {global_step}, Episodes: {episode_count}")
                wandb.log({"global_step": global_step, "episodes": episode_count}, step=global_step)
        
        print(f"\nTraining complete! Total episodes: {episode_count}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        
        if envs is not None:
            envs.close()
        
        if writer is not None:
            writer.close()
        
        if wandb_run is not None:
            # Log final table data
            if 'video_table' in locals():
                wandb.log({"final_video_table": video_table})
            wandb.finish()
            print("WandB run finished")


if __name__ == "__main__":
    main()