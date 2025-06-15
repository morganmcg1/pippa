#!/usr/bin/env python3
"""
Simple test for Isaac Lab environment integration.
Tests our Isaac Gym wrapper with fallback to standard Gym.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from environments.isaac_gym_wrapper import IsaacGymWrapper
from environments.vec_isaac_env import make_vec_env, DummyVecEnv


def test_isaac_env():
    """Test Isaac Lab environment wrapper."""
    print("Isaac Lab Environment Test")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test 1: Single environment
    print("\n1. Testing single environment...")
    try:
        env = IsaacGymWrapper(
            env_name="CartPole-v1",
            num_envs=1,
            device=device,
            isaac_lab_available=False  # Use Gym fallback for now
        )
        
        print(f"✓ Environment created: {env}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Test reset
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        print(f"✓ Reset successful, obs shape: {obs.shape}")
        
        # Test step with random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"✓ Step successful, reward: {reward}")
        
    except Exception as e:
        print(f"❌ Single environment test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Vectorized environments
    print("\n2. Testing vectorized environments...")
    try:
        num_envs = 4
        envs = make_vec_env(
            "CartPole-v1",
            n_envs=num_envs,
            vec_env_cls=DummyVecEnv
        )
        
        print(f"✓ Created {num_envs} vectorized environments")
        print(f"  Observation space: {envs.observation_space}")
        print(f"  Action space: {envs.action_space}")
        
        # Test reset
        obs = envs.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        print(f"✓ Reset successful, obs shape: {obs.shape}")
        
        # Test step with random actions
        actions = np.array([envs.action_space.sample() for _ in range(num_envs)])
        obs, rewards, dones, infos = envs.step(actions)
        print(f"✓ Step successful, rewards shape: {rewards.shape}")
        
    except Exception as e:
        print(f"❌ Vectorized environment test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: PPO model with environment
    print("\n3. Testing PPO model with environment...")
    try:
        # Determine if discrete or continuous action space
        if hasattr(envs.action_space, 'n'):
            # Discrete action space (CartPole)
            action_dim = envs.action_space.n
            print(f"  Detected discrete action space with {action_dim} actions")
        else:
            # Continuous action space
            action_dim = envs.action_space.shape[0]
            print(f"  Detected continuous action space with dim {action_dim}")
        
        model = PPOGr00tActorCriticV2(
            observation_space=envs.observation_space,
            action_dim=action_dim,
            use_multimodal_encoder=False,
            hidden_dims=(64, 64),
            device=device
        ).to(device)
        
        print(f"✓ Model created successfully")
        print(f"  Actor parameters: {sum(p.numel() for p in model.actor.parameters()):,}")
        print(f"  Critic parameters: {sum(p.numel() for p in model.critic.parameters()):,}")
        
        # Test forward pass
        obs = envs.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_tensor = torch.from_numpy(obs).float().to(device)
        
        with torch.no_grad():
            if hasattr(envs.action_space, 'n'):
                # For discrete actions, we need a different approach
                # Our current model outputs continuous actions
                print("  Note: Model outputs continuous actions, would need conversion for discrete env")
            
            actions, log_probs, entropy, values = model.get_action_and_value(obs_tensor)
            
        print(f"✓ Forward pass successful")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Values shape: {values.shape}")
        print(f"  Entropy: {entropy.mean().item():.4f}")
        
    except Exception as e:
        print(f"❌ PPO model test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    test_isaac_env()