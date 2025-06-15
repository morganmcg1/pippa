#!/usr/bin/env python3
"""
Test script for improved PPO implementation.
Tests basic functionality with a simple environment.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from configs.ppo_config_v2 import PPOConfigV2
from scripts.train_ppo_v2 import PPOTrainerV2


def test_cartpole():
    """Test PPO on CartPole-v1 for quick validation."""
    config = PPOConfigV2(
        # Use simple environment for testing
        env_name="CartPole-v1",
        num_envs=4,
        # Small scale for quick test
        total_timesteps=20_000,
        n_steps=128,
        batch_size=512,  # Will be recalculated
        n_epochs=4,
        # Disable multimodal for CartPole
        use_multimodal_encoder=False,
        # Fast logging
        log_interval=1,
        save_interval=100,
        # Disable video for speed
        capture_video=False,
        # Local testing - no WandB
        track=False,
        # Test tag
        exp_name="ppo_v2_test_cartpole"
    )
    
    print("Testing PPO V2 implementation on CartPole-v1...")
    print(f"Config: {config.num_envs} envs, {config.total_timesteps} timesteps")
    
    try:
        # Create trainer
        trainer = PPOTrainerV2(config)
        
        # Run training
        trainer.train()
        
        print("\n✓ PPO V2 test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ PPO V2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_continuous_control():
    """Test PPO on continuous control environment."""
    try:
        import gymnasium as gym
        # Check if we have a continuous control env available
        env = gym.make("Pendulum-v1")
        env.close()
        
        config = PPOConfigV2(
            env_name="Pendulum-v1",
            num_envs=2,
            total_timesteps=10_000,
            n_steps=256,
            n_epochs=4,
            use_multimodal_encoder=False,
            log_interval=1,
            save_interval=100,
            capture_video=False,
            track=False,
            exp_name="ppo_v2_test_pendulum"
        )
        
        print("\nTesting PPO V2 on continuous control (Pendulum-v1)...")
        
        trainer = PPOTrainerV2(config)
        trainer.train()
        
        print("\n✓ Continuous control test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Continuous control test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    success = True
    
    # Test 1: Basic CartPole
    if not test_cartpole():
        success = False
        
    # Test 2: Continuous control
    if not test_continuous_control():
        success = False
        
    if success:
        print("\n🎉 All PPO V2 tests passed!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        
    sys.exit(0 if success else 1)