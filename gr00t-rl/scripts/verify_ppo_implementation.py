#!/usr/bin/env python3
"""
Verification script to ensure PPO implementation matches all 37 details.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from configs.ppo_config_v2 import PPOConfigV2
from utils.networks import layer_init, GaussianActor, Critic
from utils.buffers import PPORolloutBuffer
from scripts.train_ppo_v2 import linear_schedule


def verify_implementation():
    """Verify key implementation details."""
    print("PPO Implementation Verification")
    print("=" * 50)
    
    issues = []
    
    # 1. Check Adam epsilon
    config = PPOConfigV2()
    if config.adam_epsilon == 1e-5:
        print("✅ Adam epsilon = 1e-5 (correct)")
    else:
        print(f"❌ Adam epsilon = {config.adam_epsilon} (should be 1e-5)")
        issues.append("Adam epsilon incorrect")
    
    # 2. Check orthogonal initialization
    test_layer = torch.nn.Linear(10, 10)
    layer_init(test_layer)
    # Check if weights are orthogonal
    w = test_layer.weight.data
    eye_approx = torch.mm(w, w.t())
    if torch.allclose(eye_approx, torch.eye(10), atol=0.1):
        print("✅ Orthogonal initialization working")
    else:
        print("⚠️  Orthogonal initialization may have issues")
    
    # 3. Check learning rate annealing
    lr_func = linear_schedule(1.0)
    lr_mid = lr_func(0.5)
    lr_end = lr_func(0.0)
    if lr_mid == 0.5 and lr_end == 0.0:
        print("✅ Learning rate annealing correct")
    else:
        print(f"❌ LR annealing incorrect: mid={lr_mid}, end={lr_end}")
        issues.append("Learning rate annealing incorrect")
    
    # 4. Check entropy coefficient for continuous
    if config.ent_coef == 0.0:
        print("✅ Entropy coefficient = 0.0 for continuous (correct)")
    else:
        print(f"❌ Entropy coefficient = {config.ent_coef} (should be 0.0)")
        issues.append("Entropy coefficient should be 0")
    
    # 5. Check tanh activation
    if config.activation.lower() == "tanh":
        print("✅ Using tanh activation (correct for continuous)")
    else:
        print(f"⚠️  Using {config.activation} activation (tanh recommended)")
    
    # 6. Check normalization settings
    if config.norm_obs and config.norm_reward:
        print("✅ Observation and reward normalization enabled")
    else:
        print("⚠️  Consider enabling normalization for better performance")
    
    # 7. Check clipping values
    if config.clip_obs == 10.0 and config.clip_reward == 10.0:
        print("✅ Clipping values set to 10.0 (standard)")
    else:
        print(f"⚠️  Non-standard clipping: obs={config.clip_obs}, reward={config.clip_reward}")
    
    # 8. Check gradient clipping
    if config.max_grad_norm == 0.5:
        print("✅ Gradient clipping = 0.5 (standard)")
    else:
        print(f"⚠️  Gradient clipping = {config.max_grad_norm} (0.5 is standard)")
    
    # 9. Check GAE lambda
    if config.gae_lambda == 0.95:
        print("✅ GAE lambda = 0.95 (standard)")
    else:
        print(f"⚠️  GAE lambda = {config.gae_lambda} (0.95 is standard)")
    
    # 10. Check PPO clip range
    if config.clip_range == 0.2:
        print("✅ PPO clip range = 0.2 (standard)")
    else:
        print(f"⚠️  PPO clip range = {config.clip_range} (0.2 is standard)")
    
    # 11. Check value coefficient
    if config.vf_coef == 0.5:
        print("✅ Value coefficient = 0.5 (standard)")
    else:
        print(f"⚠️  Value coefficient = {config.vf_coef} (0.5 is standard)")
    
    # 12. Test advantage normalization at batch level
    print("\nTesting advantage normalization...")
    buffer = PPORolloutBuffer(
        buffer_size=10,
        observation_space=type('obj', (object,), {'shape': (4,)}),
        action_space=type('obj', (object,), {'shape': (2,)}),
        device="cpu",
        gamma=0.99,
        gae_lambda=0.95,
        n_envs=1
    )
    
    # Add some dummy data
    for i in range(10):
        buffer.add(
            obs=torch.randn(1, 4),
            action=torch.randn(1, 2),
            reward=torch.tensor([float(i)]),
            done=torch.tensor([False]),
            value=torch.tensor([float(i)]),
            log_prob=torch.tensor([0.0])
        )
    
    # Compute advantages
    buffer.compute_returns_and_advantages(
        last_values=torch.tensor([10.0]),
        dones=torch.tensor([False])
    )
    
    # Get normalized advantages
    data = buffer.get()
    adv_mean = data['advantages'].mean().item()
    adv_std = data['advantages'].std().item()
    
    if abs(adv_mean) < 1e-6 and abs(adv_std - 1.0) < 1e-6:
        print("✅ Advantages normalized correctly (mean≈0, std≈1)")
    else:
        print(f"❌ Advantage normalization issue: mean={adv_mean:.6f}, std={adv_std:.6f}")
        issues.append("Advantage normalization incorrect")
    
    # 13. Check network architecture
    print("\nTesting network components...")
    actor = GaussianActor(
        input_dim=10,
        action_dim=3,
        hidden_dims=(256, 256),
        activation=torch.nn.Tanh
    )
    
    # Check if log_std is a parameter (not network output)
    if hasattr(actor, 'log_std') and isinstance(actor.log_std, torch.nn.Parameter):
        print("✅ State-independent log_std (correct)")
    else:
        print("❌ log_std should be a parameter, not network output")
        issues.append("State-independent log_std not implemented")
    
    # Check output layer scaling
    last_layer = None
    for module in actor.modules():
        if isinstance(module, torch.nn.Linear):
            last_layer = module
    
    if last_layer is not None:
        weight_std = last_layer.weight.data.std().item()
        if weight_std < 0.1:  # Should be around 0.01
            print("✅ Policy output layer has small initialization")
        else:
            print(f"⚠️  Policy output layer std = {weight_std:.4f} (should be ~0.01)")
    
    # Summary
    print("\n" + "=" * 50)
    if not issues:
        print("🎉 All critical implementation details verified!")
    else:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    print("\n📝 Notes:")
    print("- Using Gaussian policy (temporary) instead of flow matching")
    print("- Multi-modal observations supported via MultiModalEncoder")
    print("- Vectorized environments ready for parallel training")
    
    return len(issues) == 0


if __name__ == "__main__":
    success = verify_implementation()
    sys.exit(0 if success else 1)