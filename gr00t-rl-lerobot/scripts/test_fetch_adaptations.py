#!/usr/bin/env python3
"""
Test script to compare different Fetch adaptation approaches for SO-101.

This compares:
1. Cartesian-only approach (original wrapper)
2. Joint-coupling approach (new wrapper)
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from environments.fetch_wrapper import make_fetch_so101_env
from environments.fetch_so101_coupled import make_fetch_so101_coupled_env


def test_cartesian_approach():
    """Test the Cartesian-only approach."""
    print("\n" + "="*60)
    print("Testing Cartesian-Only Approach (7-DoF hidden)")
    print("="*60)
    
    env = make_fetch_so101_env(max_episode_steps=50)
    
    print(f"Action space: {env.action_space}")
    print(f"Action shape: {env.action_space.shape}")
    print("Actions are: [dx, dy, dz, gripper] in Cartesian space")
    print("MuJoCo internally resolves to 7 joints")
    
    obs, info = env.reset()
    print(f"\nState shape: {obs['observation']['state'].shape}")
    print("Note: State is estimated from gripper position, not true joints")
    
    # Run a few steps
    rewards = []
    for i in range(10):
        # Random Cartesian actions
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if i == 0:
            print(f"\nFirst action: {action}")
            print(f"Reward: {reward}")
    
    env.close()
    
    return np.mean(rewards)


def test_coupled_joints_approach():
    """Test the joint-coupling approach."""
    print("\n" + "="*60)
    print("Testing Joint-Coupling Approach (6-DoF effective)")
    print("="*60)
    
    # Test with Cartesian actions first
    env = make_fetch_so101_coupled_env(
        max_episode_steps=50,
        use_joint_space=False,
        couple_joints=True
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Action shape: {env.action_space.shape}")
    print("Actions are: [dx, dy, dz, gripper] but joints 1&3 are coupled")
    print("Effectively simulates 6-DoF behavior")
    
    obs, info = env.reset()
    print(f"\nState shape: {obs['observation']['state'].shape}")
    print("State represents 6 coupled joints")
    
    # Run a few steps
    rewards = []
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if i == 0:
            print(f"\nFirst action: {action}")
            print(f"Reward: {reward}")
    
    env.close()
    
    return np.mean(rewards)


def test_joint_space_control():
    """Test joint-space control (if implemented)."""
    print("\n" + "="*60)
    print("Testing Joint-Space Control (6D actions)")
    print("="*60)
    
    try:
        env = make_fetch_so101_coupled_env(
            max_episode_steps=50,
            use_joint_space=True,
            couple_joints=True
        )
        
        print(f"Action space: {env.action_space}")
        print(f"Action shape: {env.action_space.shape}")
        print("Actions are: 6 joint velocities/positions")
        print("Mapped to 7-DoF Fetch via coupling")
        
        obs, info = env.reset()
        
        # Test with small joint movements
        action = np.array([0.1, -0.1, 0.05, 0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nJoint action: {action}")
        print(f"Resulting state: {obs['observation']['state']}")
        
        env.close()
        
    except Exception as e:
        print(f"Joint-space control not fully implemented: {e}")
        print("This would require modifying Fetch to accept joint commands")


def compare_approaches():
    """Compare the different adaptation approaches."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    approaches = [
        {
            "name": "Cartesian-Only (Current)",
            "pros": [
                "✓ No environment modifications needed",
                "✓ Can start training immediately",
                "✓ Stable and well-tested"
            ],
            "cons": [
                "✗ 7-DoF vs 6-DoF mismatch",
                "✗ Can't learn joint-specific behaviors",
                "✗ May not transfer well to real SO-101"
            ],
            "use_when": "Quick prototyping, visual-only learning"
        },
        {
            "name": "Joint Coupling",
            "pros": [
                "✓ Simulates 6-DoF behavior",
                "✓ Better matches SO-101 kinematics",
                "✓ Maintains Fetch infrastructure"
            ],
            "cons": [
                "✗ Still an approximation",
                "✗ Coupling may introduce artifacts",
                "✗ Requires careful tuning"
            ],
            "use_when": "Need better kinematic fidelity"
        },
        {
            "name": "Full 6-DoF Modification",
            "pros": [
                "✓ Exact 6-DoF representation",
                "✓ True SO-101 kinematics",
                "✓ Best sim-to-real transfer"
            ],
            "cons": [
                "✗ Complex implementation (2-4 weeks)",
                "✗ Risk of breaking Fetch",
                "✗ Maintenance burden"
            ],
            "use_when": "Production deployment, high fidelity needed"
        }
    ]
    
    for approach in approaches:
        print(f"\n{approach['name']}:")
        print("Pros:")
        for pro in approach['pros']:
            print(f"  {pro}")
        print("Cons:")
        for con in approach['cons']:
            print(f"  {con}")
        print(f"Use when: {approach['use_when']}")


def main():
    """Run all tests."""
    print("Fetch to SO-101 Adaptation Test Suite")
    print("Testing different approaches for 7-DoF → 6-DoF mapping")
    
    # Test approaches
    cartesian_reward = test_cartesian_approach()
    coupled_reward = test_coupled_joints_approach()
    test_joint_space_control()
    
    # Compare
    compare_approaches()
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("For GR00T RL experiments:")
    print("1. Start with Cartesian-only for immediate testing")
    print("2. Move to joint coupling for better fidelity")
    print("3. Consider full modification only if needed")
    print("\nThe visual observations matter more than perfect kinematics")
    print("for the GR00T model's learning process.")


if __name__ == "__main__":
    main()