#!/usr/bin/env python3
"""
Quick test of PPO and GRPO training on Fetch environments.
Runs for just a few iterations to verify everything works.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success!")
        if result.stdout:
            print("\nOutput:")
            print(result.stdout[-500:])  # Last 500 chars
    else:
        print("❌ Failed!")
        print("\nError:")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """Run quick training tests."""
    print("Quick Training Test Suite")
    print("=" * 60)
    
    # Test PPO
    ppo_cmd = (
        "python scripts/train_ppo_fetch.py "
        "--env-id FetchReach-v3 "
        "--total-timesteps 5000 "
        "--num-envs 2 "
        "--num-steps 50 "
        "--learning-rate 3e-4 "
        "--seed 42 "
        "--cuda True"
    )
    
    ppo_success = run_command(ppo_cmd, "PPO on FetchReach")
    
    # Test GRPO
    grpo_cmd = (
        "python scripts/train_grpo_fetch.py "
        "--env-id FetchReach-v3 "
        "--total-timesteps 5000 "
        "--num-rollouts 4 "
        "--learning-rate 3e-4 "
        "--seed 42 "
        "--cuda True"
    )
    
    grpo_success = run_command(grpo_cmd, "GRPO on FetchReach")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"PPO Test:  {'✅ PASSED' if ppo_success else '❌ FAILED'}")
    print(f"GRPO Test: {'✅ PASSED' if grpo_success else '❌ FAILED'}")
    
    if ppo_success and grpo_success:
        print("\n🎉 All tests passed! Training pipeline is working.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())