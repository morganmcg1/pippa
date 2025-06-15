#!/usr/bin/env python3
"""
Simple test of Fetch environments with our PPO model.
Minimal dependencies for quick testing.
"""

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics  # This import registers the Fetch environments
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from environments.fetch_wrapper import FetchGoalWrapper


def test_fetch_ppo():
    """Test PPO model with Fetch environment."""
    print("Testing PPO with Fetch Environment")
    print("=" * 50)
    
    # Create environment
    env_name = "FetchReach-v3"
    env = gym.make(env_name)
    env = FetchGoalWrapper(
        env,
        observation_mode="observation_goal",
        reward_mode="sparse",
        device="cpu"
    )
    
    print(f"Environment: {env_name}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    model = PPOGr00tActorCriticV2(
        observation_space=env.observation_space,
        action_dim=env.action_space.shape[0],
        hidden_dims=(64, 64),
        use_multimodal_encoder=False,
        device=device
    ).to(device)
    
    print(f"Model created successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test rollout
    print("\nTesting rollout...")
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    for i in range(50):  # Fetch episodes are 50 steps
        # Get action from model
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = model.get_action_and_value(obs_tensor)
        
        # Execute action
        action_np = action.cpu().numpy().squeeze()
        obs, reward, terminated, truncated, info = env.step(action_np)
        
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode completed:")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward}")
    print(f"  Success: {info.get('is_success', False)}")
    print(f"  Final distance: {info.get('distance_to_goal', -1):.3f}")
    
    env.close()
    
    # Test with multiple environments
    print("\n" + "=" * 50)
    print("Testing with multiple environments...")
    
    num_envs = 4
    envs = []
    for _ in range(num_envs):
        env = gym.make(env_name)
        env = FetchGoalWrapper(env, observation_mode="observation_goal", reward_mode="sparse")
        envs.append(env)
    
    # Reset all
    observations = []
    for env in envs:
        obs, _ = env.reset()
        observations.append(obs)
    
    obs_batch = torch.FloatTensor(np.array(observations)).to(device)
    
    # Get actions for all
    with torch.no_grad():
        actions, log_probs, entropy, values = model.get_action_and_value(obs_batch)
    
    print(f"✓ Batch inference successful!")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Mean entropy: {entropy.mean().item():.4f}")
    
    # Close all
    for env in envs:
        env.close()
    
    print("\n✅ All tests passed!")


def test_grpo_setup():
    """Test GRPO setup with Fetch."""
    print("\n" + "=" * 50)
    print("Testing GRPO Setup")
    print("=" * 50)
    
    env_name = "FetchReach-v3"
    num_rollouts = 4
    
    # Create environments
    envs = []
    for _ in range(num_rollouts):
        env = gym.make(env_name)
        env = FetchGoalWrapper(env, observation_mode="observation_goal", reward_mode="sparse")
        envs.append(env)
    
    # Reset all with same seed
    seed = 42
    observations = []
    for env in envs:
        obs, _ = env.reset(seed=seed)
        observations.append(obs)
    
    print(f"✓ Created {num_rollouts} environments")
    print(f"✓ All reset to same initial state")
    
    # Check initial states are identical
    for i in range(1, num_rollouts):
        if not np.allclose(observations[0], observations[i]):
            print("✗ Initial states are not identical!")
        else:
            print(f"✓ Environment {i} has identical initial state")
    
    # Close all
    for env in envs:
        env.close()
    
    print("\n✓ GRPO setup test complete!")


if __name__ == "__main__":
    test_fetch_ppo()
    test_grpo_setup()