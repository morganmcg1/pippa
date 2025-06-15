#!/usr/bin/env python3
"""
Simple test for PPO training to debug issues.
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Create a simple environment
env = gym.make("FetchReach-v3")
print(f"Environment created: {env}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Test reset and step
obs, info = env.reset()
print(f"\nInitial observation shape: {obs.shape}")
print(f"Initial observation sample: {obs[:5]}")

# Take a random action
action = env.action_space.sample()
print(f"\nAction shape: {action.shape}")
print(f"Action sample: {action}")

next_obs, reward, terminated, truncated, info = env.step(action)
print(f"\nAfter step:")
print(f"  Next obs shape: {next_obs.shape}")
print(f"  Reward: {reward}")
print(f"  Terminated: {terminated}")
print(f"  Truncated: {truncated}")
print(f"  Info keys: {list(info.keys())}")

# Test with wrapper
from environments.fetch_wrapper import FetchGoalWrapper

wrapped_env = FetchGoalWrapper(
    env,
    observation_mode="observation_goal",
    reward_mode="sparse",
    goal_in_observation=True,
    normalize_observations=False,
    device="cpu"
)

print(f"\nWrapped environment:")
print(f"Observation space: {wrapped_env.observation_space}")
print(f"Action space: {wrapped_env.action_space}")

wrapped_obs, wrapped_info = wrapped_env.reset()
print(f"\nWrapped observation shape: {wrapped_obs.shape}")

# Test buffer
from utils.buffers import PPORolloutBuffer

buffer = PPORolloutBuffer(
    buffer_size=128,
    observation_space=wrapped_env.observation_space,
    action_space=wrapped_env.action_space,
    device="cpu",
    gamma=0.99,
    gae_lambda=0.95,
    n_envs=1
)

print(f"\nBuffer created successfully")
print(f"Buffer size: {buffer.buffer_size}")
print(f"Number of envs: {buffer.n_envs}")

# Test adding data
for i in range(10):
    obs_tensor = torch.tensor(wrapped_obs).unsqueeze(0)  # Add batch dimension
    action_tensor = torch.tensor(action).unsqueeze(0)
    reward_tensor = torch.tensor([reward])
    done_tensor = torch.tensor([terminated or truncated])
    value_tensor = torch.tensor([0.0])
    log_prob_tensor = torch.tensor([0.0])
    
    buffer.add(
        obs_tensor,
        action_tensor,
        reward_tensor,
        done_tensor,
        value_tensor,
        log_prob_tensor
    )
    
    wrapped_obs, reward, terminated, truncated, info = wrapped_env.step(action)
    action = wrapped_env.action_space.sample()

print(f"\nAdded {buffer.pos} samples to buffer")

# Test getting data
try:
    data = buffer.get(batch_size=None)
    print(f"\nBuffer get returned: {type(data)}")
    if isinstance(data, dict):
        print(f"Data keys: {list(data.keys())}")
    buffer.reset()
    print("Buffer reset successful")
except Exception as e:
    print(f"Error getting data from buffer: {e}")

print("\nTest completed successfully!")