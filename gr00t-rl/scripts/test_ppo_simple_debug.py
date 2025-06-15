#!/usr/bin/env python3
"""
Simple test to verify PPO works with basic setup.
"""

import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2


def test_basic():
    """Test basic PPO functionality."""
    print("Testing Basic PPO Setup")
    print("=" * 50)
    
    # Create simple environment
    env = gym.make("Pendulum-v1")
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"Environment: Pendulum-v1")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")
    
    # Create model with explicit dimensions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Debug: Print what we're passing to the model
    print(f"\nCreating model with:")
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_dim: {act_dim}")
    print(f"  use_multimodal_encoder: False")
    
    try:
        model = PPOGr00tActorCriticV2(
            observation_space=env.observation_space,
            action_dim=act_dim,
            hidden_dims=[64, 64],  # Smaller network for testing
            use_multimodal_encoder=False,
            device=device
        ).to(device)
        
        print(f"\n✓ Model created successfully!")
        
        # Print network architecture
        print(f"\nActor network (trunk):")
        for i, layer in enumerate(model.actor.trunk):
            if hasattr(layer, 'in_features'):
                print(f"  Layer {i}: Linear({layer.in_features} -> {layer.out_features})")
        print(f"  Mean layer: Linear({model.actor.mean_layer.in_features} -> {model.actor.mean_layer.out_features})")
        
        print(f"\nCritic network:")
        for i, layer in enumerate(model.critic.net):
            if hasattr(layer, 'in_features'):
                print(f"  Layer {i}: Linear({layer.in_features} -> {layer.out_features})")
        
        # Test forward pass
        obs = env.reset()[0]
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        
        print(f"\nTesting forward pass:")
        print(f"  Input shape: {obs_tensor.shape}")
        
        with torch.no_grad():
            # Test individual components
            features = model._process_observation(obs_tensor)
            print(f"  Processed features shape: {features.shape}")
            
            # Test critic
            value = model.critic(features)
            print(f"  Value output shape: {value.shape}")
            
            # Test full forward
            actions, log_probs, entropy, values = model.get_action_and_value(obs_tensor)
            print(f"  Action shape: {actions.shape}")
            print(f"  Log prob shape: {log_probs.shape}")
            print(f"  Entropy: {entropy.item():.4f}")
            print(f"  Value: {values.item():.4f}")
        
        print(f"\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_basic()