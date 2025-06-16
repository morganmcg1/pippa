"""
Test script for actual GR00T model integration with Fetch environment.

This tests the full pipeline with the real GR00T model loaded from WandB.
"""

import sys
sys.path.append('..')

import os
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Set up Isaac-GR00T path
isaac_groot_path = os.path.expanduser("~/pippa/Isaac-GR00T")
if os.path.exists(isaac_groot_path):
    sys.path.insert(0, isaac_groot_path)

from environments.fetch_wrapper import make_fetch_so101_env
from policies.gr00t_policy import GR00TPolicy, GR00TConfig


def test_groot_loading():
    """Test loading the actual GR00T model from WandB."""
    print("=" * 50)
    print("Testing GR00T Model Loading")
    print("=" * 50)
    
    # Create policy with actual GR00T
    config = GR00TConfig(
        wandb_artifact_path="wild-ai/pippa/gr00t-sft-so100_dualcam-bs32:v0",
        data_config="so100_dualcam",
        embodiment_tag="new_embodiment",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Artifact: {config.wandb_artifact_path}")
    print(f"  Data config: {config.data_config}")
    print(f"  Embodiment: {config.embodiment_tag}")
    
    try:
        policy = GR00TPolicy(config)
        print("\n✓ GR00T policy created successfully!")
        
        # Verify actual model was loaded
        print("✓ Actual GR00T model loaded!")
        print(f"  Model type: {type(policy.groot_policy)}")
        print(f"  Model device: {policy.device}")
        
        return policy
    except Exception as e:
        print(f"\n✗ Failed to load GR00T model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_inference(policy):
    """Test GR00T inference on dummy data."""
    print("\n" + "=" * 50)
    print("Testing GR00T Inference")
    print("=" * 50)
    
    # Create dummy observation
    batch = {
        "observation": {
            "images": {
                "front": torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8),
                "wrist": torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8),
            },
            "state": torch.randn(6, dtype=torch.float32),
        },
        "instruction": "Pick the object and place it in the target location"
    }
    
    try:
        # Test action selection
        print("Testing action selection...")
        action = policy.select_action(batch)
        print(f"✓ Action shape: {action.shape}")
        print(f"✓ Action values: {action.squeeze().cpu().numpy()}")
        print(f"✓ Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
        
        # Test forward pass (if training)
        print("\nTesting forward pass...")
        batch["action"] = torch.randn(1, 1, 6)  # Add target actions
        outputs = policy.forward(batch)
        print(f"✓ Loss: {outputs['loss'].item():.4f}")
        print(f"✓ Metrics: {outputs['metrics']}")
        
        return True
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_integration(policy):
    """Test GR00T with Fetch environment."""
    print("\n" + "=" * 50)
    print("Testing Environment Integration")
    print("=" * 50)
    
    # Create environment
    env = make_fetch_so101_env(
        max_episode_steps=50,
        instruction="Pick the cube and place it in the target area"
    )
    
    print("Running episode with GR00T policy...")
    
    try:
        obs, info = env.reset()
        total_reward = 0
        images = []
        actions_history = []
        
        for step in range(20):
            # Convert observation to torch format
            batch = {
                "observation": {
                    "images": {
                        "front": torch.from_numpy(obs["observation"]["images"]["front"]).float(),
                        "wrist": torch.from_numpy(obs["observation"]["images"]["wrist"]).float(),
                    },
                    "state": torch.from_numpy(obs["observation"]["state"]).float(),
                },
                "instruction": obs["instruction"]
            }
            
            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(batch)
                action_np = action.cpu().numpy().squeeze()
            
            # Store action for analysis
            actions_history.append(action_np)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            
            # Save image for visualization
            if step % 5 == 0:
                images.append(obs["observation"]["images"]["front"])
            
            if terminated or truncated:
                break
        
        print(f"\n✓ Episode completed!")
        print(f"  Steps: {step + 1}")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Success: {info.get('is_success', False)}")
        
        # Analyze actions
        actions_array = np.array(actions_history)
        print(f"\nAction statistics:")
        print(f"  Mean: {actions_array.mean(axis=0)}")
        print(f"  Std:  {actions_array.std(axis=0)}")
        print(f"  Range: [{actions_array.min():.3f}, {actions_array.max():.3f}]")
        
        # Save trajectory visualization
        if images:
            save_trajectory_visualization(images, "groot_fetch_trajectory.png")
            print(f"\n✓ Saved trajectory visualization")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"\n✗ Environment integration failed: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False


def save_trajectory_visualization(images, filename):
    """Save trajectory images as a figure."""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    if len(images) == 1:
        axes = [axes]
    
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(f"Step {i*5}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Run all tests."""
    print("GR00T-LeRobot Integration Test Suite")
    print("=" * 50)
    print("Testing actual GR00T model with Fetch environment")
    print()
    
    # Check dependencies
    try:
        import gymnasium_robotics
        print("✓ Gymnasium-Robotics installed")
    except ImportError:
        print("✗ Gymnasium-Robotics not installed. Run: pip install gymnasium-robotics")
        return
    
    try:
        from gr00t.model.policy import Gr00tPolicy
        print("✓ Isaac-GR00T available")
    except ImportError:
        print("✗ Isaac-GR00T not found. Make sure it's in ~/pippa/Isaac-GR00T")
        print("  This is REQUIRED for the GR00T policy to function.")
        print("  Cannot proceed without Isaac-GR00T.")
        return
    
    # Run tests
    results = []
    
    # Test 1: Load GR00T model
    print("\n" + "-" * 50)
    policy = test_groot_loading()
    results.append(("Model Loading", policy is not None))
    
    if policy:
        # Test 2: Inference
        print("\n" + "-" * 50)
        inference_success = test_inference(policy)
        results.append(("Inference", inference_success))
        
        # Test 3: Environment integration
        print("\n" + "-" * 50)
        env_success = test_environment_integration(policy)
        results.append(("Environment Integration", env_success))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<30} {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\nAll tests passed! 🎉")
        print("\nThe GR00T model is successfully integrated with the Fetch environment.")
        print("You can now proceed with RL training using SAC.")
    else:
        print("\nSome tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("1. WandB authentication - run: wandb login")
        print("2. Missing dependencies - check pyproject.toml")
        print("3. CUDA memory - try smaller batch sizes")


if __name__ == "__main__":
    main()