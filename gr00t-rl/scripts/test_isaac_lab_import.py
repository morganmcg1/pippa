#!/usr/bin/env python3
"""
Test if Isaac Lab can be imported and used with our PPO implementation.
"""

import sys
import os

def test_isaac_lab_import():
    """Test importing Isaac Lab components."""
    print("Testing Isaac Lab Import")
    print("=" * 50)
    
    try:
        # Try to import Isaac Lab
        import isaaclab
        print("✓ Successfully imported isaaclab")
        
        # Try to import environment utilities
        from isaaclab.envs import ManagerBasedEnv
        print("✓ Successfully imported ManagerBasedEnv")
        
        # Try to import task registry
        import isaaclab_tasks
        print("✓ Successfully imported isaaclab_tasks")
        
        print("\n✅ Isaac Lab imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nIsaac Lab is not available in the current environment.")
        print("Make sure to:")
        print("1. Install Isaac Lab following the official instructions")
        print("2. Activate the Isaac Lab environment")
        print("3. Set the ISAAC_PATH environment variable if needed")
        return False


def test_isaac_lab_env_creation():
    """Test creating an Isaac Lab environment."""
    print("\nTesting Isaac Lab Environment Creation")
    print("=" * 50)
    
    try:
        from isaaclab.envs import ManagerBasedEnvCfg
        from isaaclab_tasks.manager_based.classic.cartpole import CartpoleEnvCfg
        
        # Create environment configuration
        env_cfg = CartpoleEnvCfg()
        env_cfg.scene.num_envs = 4
        
        print(f"✓ Created Cartpole environment config with {env_cfg.scene.num_envs} envs")
        
        # Note: Actually creating the environment requires Isaac Sim to be running
        # which might not be available in headless mode
        print("\n✅ Isaac Lab environment configuration successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Test imports
    import_success = test_isaac_lab_import()
    
    # Only test environment creation if imports work
    if import_success:
        test_isaac_lab_env_creation()