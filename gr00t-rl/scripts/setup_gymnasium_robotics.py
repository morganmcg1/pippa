#!/usr/bin/env python3
"""
Setup script for Gymnasium-Robotics installation and verification.
This ensures MuJoCo and Gymnasium-Robotics are properly installed.
"""

import subprocess
import sys
import os


def check_mujoco():
    """Check if MuJoCo is installed and working."""
    print("Checking MuJoCo installation...")
    try:
        import mujoco
        print(f"✓ MuJoCo version: {mujoco.__version__}")
        return True
    except ImportError:
        print("✗ MuJoCo not found")
        return False


def check_gymnasium_robotics():
    """Check if Gymnasium-Robotics is installed."""
    print("\nChecking Gymnasium-Robotics installation...")
    try:
        import gymnasium_robotics
        print(f"✓ Gymnasium-Robotics version: {gymnasium_robotics.__version__}")
        
        # Try to list available environments
        import gymnasium as gym
        robotics_envs = [env for env in gym.envs.registry.keys() if 'Fetch' in env or 'Hand' in env]
        print(f"✓ Found {len(robotics_envs)} robotics environments")
        
        # Show some examples
        if robotics_envs:
            print("  Examples:")
            for env in robotics_envs[:5]:
                print(f"    - {env}")
            if len(robotics_envs) > 5:
                print(f"    ... and {len(robotics_envs) - 5} more")
        
        return True
    except ImportError:
        print("✗ Gymnasium-Robotics not found")
        return False


def test_fetch_environment():
    """Test creating and running a Fetch environment."""
    print("\nTesting Fetch environment creation...")
    try:
        import gymnasium as gym
        
        # Create FetchReach environment
        env = gym.make("FetchReach-v3")
        print("✓ Successfully created FetchReach-v3 environment")
        
        # Check observation space
        obs, info = env.reset()
        print(f"✓ Observation space:")
        print(f"    - observation shape: {obs['observation'].shape}")
        print(f"    - desired_goal shape: {obs['desired_goal'].shape}")
        print(f"    - achieved_goal shape: {obs['achieved_goal'].shape}")
        
        # Check action space
        print(f"✓ Action space: {env.action_space}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Successfully executed step")
        print(f"    - Reward: {reward}")
        print(f"    - Terminated: {terminated}")
        print(f"    - Info keys: {list(info.keys())}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error testing Fetch environment: {e}")
        return False


def install_dependencies():
    """Install MuJoCo and Gymnasium-Robotics."""
    print("\nInstalling dependencies...")
    
    # Install MuJoCo
    print("Installing MuJoCo...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mujoco"])
    
    # Install Gymnasium-Robotics
    print("\nInstalling Gymnasium-Robotics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium-robotics"])
    
    print("\n✓ Dependencies installed successfully")


def main():
    """Main setup function."""
    print("=" * 60)
    print("Gymnasium-Robotics Setup for GR00T RL")
    print("=" * 60)
    
    # Check if already installed
    mujoco_ok = check_mujoco()
    gym_robotics_ok = check_gymnasium_robotics()
    
    if not mujoco_ok or not gym_robotics_ok:
        print("\nMissing dependencies detected. Installing...")
        try:
            install_dependencies()
            
            # Re-check after installation
            mujoco_ok = check_mujoco()
            gym_robotics_ok = check_gymnasium_robotics()
            
        except Exception as e:
            print(f"\n✗ Error during installation: {e}")
            print("\nPlease install manually:")
            print("  pip install mujoco gymnasium-robotics")
            sys.exit(1)
    
    # Test environment
    if mujoco_ok and gym_robotics_ok:
        env_ok = test_fetch_environment()
        
        if env_ok:
            print("\n" + "=" * 60)
            print("✅ Setup Complete! Gymnasium-Robotics is ready for use.")
            print("=" * 60)
            
            print("\nNext steps:")
            print("1. Run scripts/test_fetch_environments.py to explore available tasks")
            print("2. Run scripts/train_ppo_fetch.py to start PPO training")
            print("3. Run scripts/train_grpo_fetch.py to test GRPO approach")
        else:
            print("\n⚠️  Setup complete but environment test failed.")
            print("This might be due to display/rendering issues on headless systems.")
    else:
        print("\n✗ Setup failed. Please check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()