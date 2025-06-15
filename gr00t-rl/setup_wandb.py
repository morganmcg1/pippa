#!/usr/bin/env python3
"""
Setup script to ensure WandB is properly configured for gr00t-rl experiments.
"""

import os
from pathlib import Path
from dotenv import load_dotenv, set_key

def setup_wandb_env():
    """Ensure .env file exists with proper WandB configuration."""
    env_path = Path(__file__).parent.parent / '.env'
    
    # Load existing .env if it exists
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Found existing .env file at {env_path}")
    else:
        print(f"Creating new .env file at {env_path}")
        env_path.touch()
    
    # Check and set required environment variables
    required_vars = {
        'WANDB_ENTITY': 'wild-ai',
        'WANDB_PROJECT': 'pippa',
    }
    
    for key, default_value in required_vars.items():
        current_value = os.getenv(key)
        if not current_value:
            set_key(env_path, key, default_value)
            print(f"✓ Set {key}={default_value}")
        else:
            print(f"✓ {key} already set to: {current_value}")
    
    # Check for API key
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        print("\n⚠️  WANDB_API_KEY not found in .env file")
        print("Please add your WandB API key to the .env file:")
        print(f"  echo 'WANDB_API_KEY=your_key_here' >> {env_path}")
        print("\nYou can find your API key at: https://wandb.ai/authorize")
    else:
        print("✓ WANDB_API_KEY is configured")
    
    # Create .env.example if it doesn't exist
    example_path = env_path.parent / '.env.example'
    if not example_path.exists():
        with open(example_path, 'w') as f:
            f.write("# WandB Configuration\n")
            f.write("WANDB_ENTITY=wild-ai\n")
            f.write("WANDB_PROJECT=pippa\n")
            f.write("WANDB_API_KEY=your_key_here\n")
        print(f"\n✓ Created {example_path} for reference")

def test_wandb_import():
    """Test that wandb can be imported and initialized."""
    try:
        import wandb
        print("\n✓ WandB library imported successfully")
        print(f"  Version: {wandb.__version__}")
        
        # Test if we can get the API
        if os.getenv('WANDB_API_KEY'):
            try:
                api = wandb.Api()
                print("✓ WandB API connection successful")
            except Exception as e:
                print(f"⚠️  WandB API connection failed: {e}")
        
    except ImportError:
        print("\n❌ WandB not installed!")
        print("Install with: uv pip install wandb")

def main():
    """Run all setup checks."""
    print("WandB Setup for gr00t-rl")
    print("=" * 50)
    
    setup_wandb_env()
    test_wandb_import()
    
    print("\n" + "=" * 50)
    print("Setup complete! You can now run training with WandB logging:")
    print("  uv run python scripts/train_ppo_fetch.py")
    print("  uv run python scripts/train_grpo_fetch.py")
    print("\nTraining will automatically log to: https://wandb.ai/wild-ai/pippa")

if __name__ == "__main__":
    main()