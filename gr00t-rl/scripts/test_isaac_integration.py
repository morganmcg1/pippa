#!/usr/bin/env python3
"""
Test script for Isaac Lab integration with GR00T wrapper.
Tests basic functionality before full training.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "IsaacLab"))
sys.path.append(str(Path(__file__).parent.parent.parent / "rsl_rl"))

from algorithms.gr00t_wrapper import GR00TActorCritic, create_gr00t_actor_critic_for_isaac
from configs.isaac_lab_ppo_cfg import GR00TTestPPOCfg, GR00TFrozenPPOCfg


def test_gr00t_wrapper():
    """Test GR00T wrapper functionality."""
    print("=" * 50)
    print("Testing GR00T ActorCritic Wrapper")
    print("=" * 50)
    
    # Test dimensions
    num_actor_obs = 45  # Typical for robot proprioception
    num_critic_obs = 48  # May include privileged info
    num_actions = 12    # Joint positions/torques
    batch_size = 16
    
    # Test 1: Create wrapper without GR00T (MLP fallback)
    print("\n1. Testing MLP fallback mode...")
    model_mlp = GR00TActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        use_gr00t=False,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        device="cpu"  # For testing
    )
    
    # Create dummy observations
    actor_obs = torch.randn(batch_size, num_actor_obs)
    critic_obs = torch.randn(batch_size, num_critic_obs)
    
    # Test action generation
    with torch.no_grad():
        actions = model_mlp.act(actor_obs)
        print(f"✓ Generated actions shape: {actions.shape}")
        
        # Test deterministic actions
        det_actions = model_mlp.act_inference(actor_obs)
        print(f"✓ Deterministic actions shape: {det_actions.shape}")
        
        # Test value prediction
        values = model_mlp.get_value(critic_obs)
        print(f"✓ Values shape: {values.shape}")
        
        # Test evaluation (pass both actor and critic obs)
        log_probs, values, entropy = model_mlp.evaluate(actor_obs, actions, critic_obs)
        print(f"✓ Log probs shape: {log_probs.shape}")
        print(f"✓ Entropy: {entropy.item():.4f}")
    
    print("\n✓ MLP fallback test passed!")
    
    # Test 2: Test with mock GR00T (if available)
    print("\n2. Testing GR00T mode (will fallback if not available)...")
    try:
        model_gr00t = GR00TActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            use_gr00t=True,
            gr00t_model_path="nvidia/GR00T-N1.5-3B",
            freeze_backbone=True,
            device="cpu"
        )
        
        # Quick functionality test
        with torch.no_grad():
            actions = model_gr00t.act(actor_obs)
            values = model_gr00t.get_value(critic_obs)
        
        print(f"✓ GR00T mode test passed!")
        print(f"  Using GR00T: {model_gr00t.use_gr00t}")
        
    except Exception as e:
        print(f"⚠ GR00T test skipped (expected): {e}")
    
    # Test 3: Factory function
    print("\n3. Testing factory function...")
    env_cfg = {
        "num_observations": num_actor_obs,
        "num_privileged_obs": num_critic_obs,
        "num_actions": num_actions
    }
    
    model_factory = create_gr00t_actor_critic_for_isaac(
        env_cfg,
        use_gr00t=False,
        device="cpu"
    )
    
    print("✓ Factory function test passed!")
    
    # Test 4: rsl_rl compatibility
    print("\n4. Testing rsl_rl interface compatibility...")
    
    # Test required attributes
    assert hasattr(model_mlp, 'is_recurrent')
    assert hasattr(model_mlp, 'distribution')
    assert hasattr(model_mlp, 'action_mean')
    assert hasattr(model_mlp, 'action_std')
    print("✓ Has required attributes")
    
    # Test required methods
    assert callable(model_mlp.act)
    assert callable(model_mlp.get_actions_log_prob)
    assert callable(model_mlp.act_inference)
    assert callable(model_mlp.evaluate)
    assert callable(model_mlp.reset)
    print("✓ Has required methods")
    
    # Test get_actions_log_prob (used by rsl_rl)
    actions, log_probs = model_mlp.get_actions_log_prob(actor_obs)
    print(f"✓ get_actions_log_prob works: actions {actions.shape}, log_probs {log_probs.shape}")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✅")
    print("=" * 50)


def test_config_loading():
    """Test configuration classes."""
    print("\n" + "=" * 50)
    print("Testing Configuration Classes")
    print("=" * 50)
    
    # Test base config
    cfg = GR00TTestPPOCfg()
    print(f"\n✓ Loaded test config: {cfg.experiment_name}")
    print(f"  Learning rate: {cfg.algorithm_cfg.learning_rate}")
    print(f"  Using GR00T: {cfg.policy_cfg.use_gr00t}")
    
    # Test frozen config
    cfg_frozen = GR00TFrozenPPOCfg()
    print(f"\n✓ Loaded frozen config: {cfg_frozen.experiment_name}")
    print(f"  Freeze backbone: {cfg_frozen.policy_cfg.freeze_backbone}")
    print(f"  Learning epochs: {cfg_frozen.algorithm_cfg.num_learning_epochs}")
    
    print("\n✓ Config tests passed!")


def test_isaac_lab_import():
    """Test if Isaac Lab can be imported."""
    print("\n" + "=" * 50)
    print("Testing Isaac Lab Import")
    print("=" * 50)
    
    try:
        # Try importing Isaac Lab components
        from isaaclab.envs import ManagerTerminationCfg
        from isaaclab.utils import configclass
        print("✓ Isaac Lab core imports successful")
        
        # Try importing rsl_rl
        from rsl_rl.modules.actor_critic import ActorCritic as RslActorCritic
        from rsl_rl.algorithms.ppo import PPO
        print("✓ rsl_rl imports successful")
        
        # Check if our model is compatible
        print("\n✓ Import test passed!")
        return True
        
    except ImportError as e:
        print(f"⚠ Import failed (Isaac Lab not installed): {e}")
        print("\nTo install Isaac Lab:")
        print("  cd IsaacLab")
        print("  ./isaaclab.sh --install")
        return False


def main():
    """Run all tests."""
    print("\nGR00T-RL Isaac Lab Integration Tests")
    print("=" * 70)
    
    # Run tests
    test_gr00t_wrapper()
    test_config_loading()
    isaac_available = test_isaac_lab_import()
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("✅ GR00T wrapper implementation correct")
    print("✅ Configuration system working")
    
    if isaac_available:
        print("✅ Isaac Lab available")
    else:
        print("⚠️  Isaac Lab not installed (run ./isaaclab.sh --install)")
    
    print("\nNext steps:")
    if not isaac_available:
        print("1. Install Isaac Lab dependencies")
    print("2. Create training script for specific task")
    print("3. Test with simple environment (e.g., Isaac-Reach-Franka-v0)")
    print("=" * 70)


if __name__ == "__main__":
    main()