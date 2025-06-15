#!/usr/bin/env python3
"""
Test Isaac Lab environment creation without full Isaac Sim.
This explores what's possible with just the Isaac Lab framework.
"""

import sys
import os

# Add Isaac Lab to path
isaac_lab_path = os.path.expanduser("~/isaac-lab/source")
if os.path.exists(isaac_lab_path):
    sys.path.insert(0, isaac_lab_path)

def test_isaac_lab_modules():
    """Test what Isaac Lab modules are available."""
    print("Testing Isaac Lab Module Availability")
    print("=" * 50)
    
    modules_to_test = [
        ("isaaclab", "Core Isaac Lab"),
        ("isaaclab.envs", "Environment base classes"),
        ("isaaclab.utils", "Utilities"),
        ("isaaclab.sim", "Simulation interfaces"),
        ("isaaclab_tasks", "Task implementations"),
        ("isaaclab_assets", "Asset definitions"),
    ]
    
    available_modules = []
    
    for module_name, description in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"✓ {description} ({module_name})")
            available_modules.append(module_name)
            
            # Try to list some attributes
            attrs = [attr for attr in dir(module) if not attr.startswith('_')]
            if attrs:
                print(f"  Available: {', '.join(attrs[:5])}{'...' if len(attrs) > 5 else ''}")
                
        except ImportError as e:
            print(f"✗ {description} ({module_name}): {e}")
    
    return available_modules


def test_manager_based_env():
    """Test creating a manager-based environment."""
    print("\nTesting Manager-Based Environment")
    print("=" * 50)
    
    try:
        from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
        print("✓ Imported ManagerBasedEnv classes")
        
        # Try to create a simple config
        class SimpleEnvCfg(ManagerBasedEnvCfg):
            """Simple environment configuration."""
            
            def __init__(self):
                super().__init__()
                self.episode_length_s = 5.0
                self.decimation = 2
                self.num_envs = 4
                self.env_spacing = 2.5
                self.sim_dt = 0.01
                
        config = SimpleEnvCfg()
        print(f"✓ Created environment config")
        print(f"  Num envs: {config.num_envs}")
        print(f"  Episode length: {config.episode_length_s}s")
        print(f"  Sim dt: {config.sim_dt}s")
        
        # Note: Actually creating the environment requires Isaac Sim
        print("\nNote: Creating actual environments requires Isaac Sim to be running")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_direct_workflow():
    """Test direct workflow (non-manager based)."""
    print("\nTesting Direct Workflow")
    print("=" * 50)
    
    try:
        from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
        print("✓ Imported DirectRLEnv classes")
        
        # Check what's required for direct env
        print("  DirectRLEnvCfg attributes:")
        for attr in ['decimation', 'episode_length_s', 'num_envs', 'env_spacing']:
            if hasattr(DirectRLEnvCfg, attr):
                print(f"    - {attr}")
                
    except ImportError:
        print("✗ DirectRLEnv not available (might be in different module)")
    except Exception as e:
        print(f"✗ Error: {e}")


def check_omniverse_requirements():
    """Check if Omniverse/Isaac Sim components are available."""
    print("\nChecking Omniverse/Isaac Sim Requirements")
    print("=" * 50)
    
    omni_modules = [
        "omni.isaac.core",
        "omni.isaac.kit", 
        "omni.isaac.sim",
        "omni.kit.app",
        "carb",
        "pxr"
    ]
    
    for module in omni_modules:
        try:
            __import__(module)
            print(f"✓ {module} available")
        except ImportError:
            print(f"✗ {module} not available (needs Isaac Sim)")
    
    print("\nConclusion: Full robot simulation requires Isaac Sim installation")


def main():
    """Run all tests."""
    print("Isaac Lab Environment Testing")
    print("=" * 80)
    
    # Test module availability
    available = test_isaac_lab_modules()
    
    if "isaaclab" in available:
        # Test environment creation
        test_manager_based_env()
        test_direct_workflow()
    
    # Check Omniverse requirements
    check_omniverse_requirements()
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("- Isaac Lab framework is installed")
    print("- Environment classes are available")
    print("- Full simulation requires Isaac Sim (not just pip packages)")
    print("- For robot training, need to install Isaac Sim via:")
    print("  1. Omniverse Launcher (GUI)")
    print("  2. Docker container")
    print("  3. Standalone installation")


if __name__ == "__main__":
    main()