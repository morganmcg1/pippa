#!/usr/bin/env python3
"""
Debug PPO implementation issues.
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from environments.vec_isaac_env import make_vec_env, DummyVecEnv


def test_single_env():
    """Test with single environment."""
    print("\nTesting single Pendulum environment...")
    
    # Create single environment
    env = gym.make("Pendulum-v1")
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPOGr00tActorCriticV2(
        observation_space=env.observation_space,
        action_dim=act_dim,
        use_multimodal_encoder=False,
        hidden_dims=(64, 64),  # Smaller for testing
        device=device
    ).to(device)
    
    print(f"Model created successfully")
    
    # Test forward pass
    obs = env.reset()[0]  # New API returns (obs, info)
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    
    print(f"Observation shape: {obs_tensor.shape}")
    
    with torch.no_grad():
        actions, log_probs, entropy, values = model.get_action_and_value(obs_tensor)
    
    print(f"Actions shape: {actions.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Entropy: {entropy.item() if entropy.numel() == 1 else entropy.mean().item()}")
    
    print("✓ Single environment test passed!")
    

def test_vec_env():
    """Test with vectorized environment."""
    print("\nTesting vectorized Pendulum environment...")
    
    # Create vectorized environment
    num_envs = 4
    envs = make_vec_env("Pendulum-v1", n_envs=num_envs, vec_env_cls=DummyVecEnv)
    
    print(f"Created {num_envs} environments")
    print(f"Observation space: {envs.observation_space}")
    print(f"Action space: {envs.action_space}")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPOGr00tActorCriticV2(
        observation_space=envs.observation_space,
        action_dim=envs.action_space.shape[0],
        use_multimodal_encoder=False,
        hidden_dims=(64, 64),  # Smaller for testing
        device=device
    ).to(device)
    
    print(f"Model created successfully")
    
    # Test forward pass
    obs = envs.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # New gym API returns (obs, info)
    print(f"Reset observation shape: {obs.shape}")
    
    obs_tensor = torch.from_numpy(obs).float().to(device)
    
    with torch.no_grad():
        actions, log_probs, entropy, values = model.get_action_and_value(obs_tensor)
    
    print(f"Actions shape: {actions.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Entropy: {entropy.item() if entropy.numel() == 1 else entropy.mean().item()}")
    
    # Test step
    actions_np = actions.cpu().numpy()
    next_obs, rewards, dones, infos = envs.step(actions_np)
    
    print(f"Step observation shape: {next_obs.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Dones shape: {dones.shape}")
    
    print("✓ Vectorized environment test passed!")


def main():
    """Run debug tests."""
    print("PPO Debug Tests")
    print("=" * 50)
    
    try:
        test_single_env()
    except Exception as e:
        print(f"❌ Single env test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_vec_env()
    except Exception as e:
        print(f"❌ Vec env test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()