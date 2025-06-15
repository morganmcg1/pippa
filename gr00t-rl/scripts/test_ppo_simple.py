#!/usr/bin/env python3
"""
Simple test for PPO on continuous control tasks.
Tests basic functionality without Isaac Lab.
"""

import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from configs.ppo_config_v2 import PPOConfigV2


def test_pendulum():
    """Test PPO on Pendulum-v1 (continuous control)."""
    print("\n" + "="*50)
    print("Testing PPO V2 on Pendulum-v1")
    print("="*50)
    
    # Create single environment to get spaces
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create actor-critic model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PPOGr00tActorCriticV2(
        observation_space=env.observation_space,
        action_dim=act_dim,
        use_multimodal_encoder=False,
        device=device
    ).to(device)
    
    print(f"\nModel created successfully!")
    print(f"Actor parameters: {sum(p.numel() for p in model.actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in model.critic.parameters()):,}")
    
    # Test forward pass
    obs = torch.randn(4, obs_dim).to(device)  # Batch of 4
    
    # Test action generation
    with torch.no_grad():
        actions, log_probs, entropy, values = model.get_action_and_value(obs)
        
    print(f"\nForward pass successful!")
    print(f"Actions shape: {actions.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Entropy: {entropy.mean().item():.4f}")
    
    # Test evaluation
    actions_test = torch.randn(4, act_dim).to(device)
    with torch.no_grad():
        log_probs_test, entropy_test, values_test = model.evaluate_actions(obs, actions_test)
        
    print(f"\nEvaluation successful!")
    print(f"Log probs: {log_probs_test.mean().item():.4f}")
    
    env.close()
    return True


def test_multimodal():
    """Test multimodal observation support."""
    print("\n" + "="*50)
    print("Testing Multi-modal Observation Support")
    print("="*50)
    
    # Create dict observation space
    obs_space = gym.spaces.Dict({
        'proprioception': gym.spaces.Box(-1, 1, shape=(12,)),
        'vision': gym.spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8),
        'language': gym.spaces.Box(-1, 1, shape=(768,))  # Language embedding
    })
    
    act_space = gym.spaces.Box(-1, 1, shape=(6,))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with multimodal support
    model = PPOGr00tActorCriticV2(
        observation_space=obs_space,
        action_dim=act_space.shape[0],
        use_multimodal_encoder=True,
        device=device
    ).to(device)
    
    print("Multi-modal model created successfully!")
    
    # Test forward pass with dict observations
    batch_size = 2
    obs = {
        'proprioception': torch.randn(batch_size, 12).to(device),
        'vision': torch.randn(batch_size, 3, 84, 84).to(device),  # CHW format
        'language': torch.randn(batch_size, 768).to(device)
    }
    
    with torch.no_grad():
        actions, log_probs, entropy, values = model.get_action_and_value(obs)
        
    print(f"\nMulti-modal forward pass successful!")
    print(f"Actions shape: {actions.shape}")
    print(f"Values shape: {values.shape}")
    
    return True


def test_gpu_utilization():
    """Test GPU memory usage."""
    if torch.cuda.is_available():
        print("\n" + "="*50)
        print("Testing GPU Utilization")
        print("="*50)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Create large batch environment simulation
        batch_size = 256
        obs_dim = 45  # Typical robot proprioception
        act_dim = 12  # Typical robot joints
        
        device = torch.device("cuda")
        
        # Create model
        obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,))
        act_space = gym.spaces.Box(-1, 1, shape=(act_dim,))
        
        model = PPOGr00tActorCriticV2(
            observation_space=obs_space,
            action_dim=act_dim,
            use_multimodal_encoder=False,
            device=device
        ).to(device)
        
        # Simulate batch processing
        obs = torch.randn(batch_size, obs_dim).to(device)
        
        with torch.no_grad():
            actions, log_probs, entropy, values = model.get_action_and_value(obs)
            
        # Get memory after forward pass
        current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_used = current_memory - initial_memory
        
        print(f"Batch size: {batch_size}")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Memory per sample: {memory_used / batch_size * 1000:.2f} KB")
        
        # Cleanup
        del model, obs, actions, log_probs, entropy, values
        torch.cuda.empty_cache()
        
        return True
    else:
        print("\nGPU not available, skipping GPU tests")
        return True


def main():
    """Run all tests."""
    print("PPO V2 Simple Tests")
    print("==================")
    
    tests = [
        ("Pendulum Test", test_pendulum),
        ("Multi-modal Test", test_multimodal),
        ("GPU Test", test_gpu_utilization)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    all_passed = True
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    main()