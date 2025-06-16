"""
Test script to verify Fetch environment wrapper and GR00T policy integration.
"""

import sys
sys.path.append('..')

import gymnasium as gym
import numpy as np
from environments.fetch_wrapper import make_fetch_so101_env
from policies.gr00t_policy import GR00TPolicy, GR00TConfig
import matplotlib.pyplot as plt
from PIL import Image


def test_environment():
    """Test the Fetch wrapper environment."""
    print("Testing Fetch SO-101 wrapper...")
    
    # Create environment
    env = make_fetch_so101_env(
        render_mode="rgb_array",
        max_episode_steps=50,
        instruction="Pick the red cube and place it in the target area"
    )
    
    # Test reset
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Image shapes: front={obs['observation']['images']['front'].shape}, "
          f"wrist={obs['observation']['images']['wrist'].shape}")
    print(f"State shape: {obs['observation']['state'].shape}")
    print(f"Instruction: {obs['instruction']}")
    
    # Test random actions
    print("\nTesting random actions...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        
    env.close()
    print("Environment test passed! ✓")
    return True


def test_policy():
    """Test the GR00T policy wrapper."""
    print("\nTesting GR00T policy...")
    
    # Create policy
    config = GR00TConfig()
    policy = GR00TPolicy(config)
    print(f"Policy device: {policy.device}")
    
    # Create dummy observation
    batch = {
        "observation": {
            "images": {
                "front": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "wrist": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            },
            "state": np.random.randn(6).astype(np.float32),
        }
    }
    
    # Convert to torch tensors
    import torch
    batch_torch = {
        "observation": {
            "images": {
                "front": torch.from_numpy(batch["observation"]["images"]["front"]),
                "wrist": torch.from_numpy(batch["observation"]["images"]["wrist"]),
            },
            "state": torch.from_numpy(batch["observation"]["state"]),
        }
    }
    
    # Test action selection
    action = policy.select_action(batch_torch)
    print(f"Action shape: {action.shape}")
    print(f"Action values: {action.cpu().numpy()}")
    
    # Test forward pass (with target actions)
    batch_torch["action"] = torch.randn(1, 1, 6)  # Batch size 1, horizon 1, 6 joints
    outputs = policy.forward(batch_torch)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Metrics: {outputs['metrics']}")
    
    print("Policy test passed! ✓")
    return True


def test_integration():
    """Test full integration of environment and policy."""
    print("\nTesting full integration...")
    
    # Create environment and policy
    env = make_fetch_so101_env(max_episode_steps=20)
    config = GR00TConfig()
    policy = GR00TPolicy(config)
    
    # Run episode
    obs, info = env.reset()
    total_reward = 0
    images = []
    
    print("Running episode with GR00T policy...")
    for step in range(20):
        # Convert observation to torch format
        import torch
        batch = {
            "observation": {
                "images": {
                    "front": torch.from_numpy(obs["observation"]["images"]["front"]).float(),
                    "wrist": torch.from_numpy(obs["observation"]["images"]["wrist"]).float(),
                },
                "state": torch.from_numpy(obs["observation"]["state"]).float(),
            }
        }
        
        # Get action from policy
        with torch.no_grad():
            action = policy.select_action(batch)
            action_np = action.cpu().numpy().squeeze()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward
        
        # Save image for visualization
        if step % 5 == 0:
            images.append(obs["observation"]["images"]["front"])
        
        if terminated or truncated:
            break
    
    print(f"Episode finished: total_reward={total_reward:.3f}, steps={step+1}")
    
    # Visualize trajectory
    if images:
        fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
        for i, img in enumerate(images):
            axes[i].imshow(img)
            axes[i].set_title(f"Step {i*5}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig('fetch_trajectory.png')
        print("Saved trajectory visualization to fetch_trajectory.png")
    
    env.close()
    print("Integration test passed! ✓")
    return True


def main():
    """Run all tests."""
    print("GR00T-LeRobot Integration Tests")
    print("=" * 50)
    
    # Check dependencies
    try:
        import gymnasium_robotics
        print("✓ Gymnasium-Robotics installed")
    except ImportError:
        print("✗ Gymnasium-Robotics not installed. Run: pip install gymnasium-robotics")
        return
    
    # Run tests
    tests = [
        ("Environment", test_environment),
        ("Policy", test_policy),
        ("Integration", test_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\nAll tests passed! 🎉")
    else:
        print("\nSome tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()