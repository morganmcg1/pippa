#!/usr/bin/env python3
"""
Test and explore Gymnasium-Robotics Fetch environments.
This script helps understand the observation/action spaces and task structure.
"""

import numpy as np
import gymnasium as gym
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


def explore_fetch_environment(env_name: str, num_steps: int = 100):
    """Explore a Fetch environment and print detailed information."""
    print(f"\n{'=' * 60}")
    print(f"Exploring: {env_name}")
    print(f"{'=' * 60}")
    
    try:
        # Create environment
        env = gym.make(env_name)
        
        # Reset and get initial observation
        obs, info = env.reset()
        
        print("\n1. Observation Space Structure:")
        print(f"   Type: {type(obs)}")
        for key, value in obs.items():
            print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
            print(f"     Range: [{value.min():.3f}, {value.max():.3f}]")
        
        print("\n2. Action Space:")
        print(f"   Type: {env.action_space}")
        print(f"   Shape: {env.action_space.shape}")
        print(f"   Range: [{env.action_space.low}, {env.action_space.high}]")
        
        print("\n3. Task Details:")
        # Get task-specific information
        if hasattr(env, 'reward_type'):
            print(f"   Reward type: {env.reward_type}")
        
        # Run a few steps to understand dynamics
        print("\n4. Sample Trajectory (10 steps):")
        total_reward = 0
        for i in range(10):
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if i < 3:  # Print first few steps
                print(f"   Step {i+1}:")
                print(f"     Action: {action}")
                print(f"     Reward: {reward}")
                print(f"     Success: {info.get('is_success', False)}")
                print(f"     Distance to goal: {np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']):.3f}")
        
        print(f"\n   Total reward (10 steps): {total_reward}")
        
        # Analyze goal structure
        print("\n5. Goal Analysis:")
        print(f"   Initial goal: {obs['desired_goal']}")
        print(f"   Current achieved: {obs['achieved_goal']}")
        print(f"   Goal distance: {np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']):.3f}")
        
        # Check success criteria
        if 'is_success' in info:
            print(f"   Success threshold: < 0.05m (typically)")
        
        env.close()
        
    except Exception as e:
        print(f"✗ Error exploring {env_name}: {e}")
        import traceback
        traceback.print_exc()


def test_goal_conditioned_structure():
    """Test the goal-conditioned RL structure of Fetch environments."""
    print("\n" + "=" * 60)
    print("Testing Goal-Conditioned RL Structure")
    print("=" * 60)
    
    env = gym.make("FetchReach-v3")
    obs, info = env.reset()
    
    print("\n1. Goal-Conditioned Observation Structure:")
    print("   This is what makes these environments special for HER")
    
    # Show how to extract different components
    observation = obs['observation']
    achieved_goal = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    
    print(f"\n   observation (proprioceptive): {observation.shape}")
    print(f"   - Contains: gripper position, gripper velocity, gripper state")
    
    print(f"\n   achieved_goal (current state): {achieved_goal.shape}")
    print(f"   - Current end-effector position: {achieved_goal}")
    
    print(f"\n   desired_goal (target): {desired_goal.shape}")
    print(f"   - Target position: {desired_goal}")
    
    # Test compute_reward function (useful for HER)
    print("\n2. Reward Computation (for HER):")
    
    # Compute reward for current state
    current_reward = env.compute_reward(achieved_goal, desired_goal, info)
    print(f"   Current reward: {current_reward}")
    
    # Compute reward if we had achieved the goal
    hypothetical_reward = env.compute_reward(desired_goal, desired_goal, info)
    print(f"   Reward if goal achieved: {hypothetical_reward}")
    
    # This is key for HER - we can relabel any trajectory
    print("\n   This allows Hindsight Experience Replay to work!")
    print("   Failed trajectories can be relabeled with achieved goals")
    
    env.close()


def compare_fetch_tasks():
    """Compare different Fetch tasks to understand progression."""
    print("\n" + "=" * 60)
    print("Fetch Task Comparison")
    print("=" * 60)
    
    tasks = [
        ("FetchReach-v3", "Move end-effector to target position"),
        ("FetchPush-v3", "Push object to target position"),
        ("FetchSlide-v3", "Slide puck to target position"), 
        ("FetchPickAndPlace-v3", "Pick up object and place at target")
    ]
    
    print("\nTask Difficulty Progression:")
    for i, (env_name, description) in enumerate(tasks, 1):
        print(f"\n{i}. {env_name}:")
        print(f"   Description: {description}")
        
        try:
            env = gym.make(env_name)
            obs, _ = env.reset()
            
            # Analyze what makes each task unique
            obs_dim = obs['observation'].shape[0]
            print(f"   Observation dimension: {obs_dim}")
            
            if "Push" in env_name or "Slide" in env_name:
                print("   Additional complexity: Object dynamics")
            if "PickAndPlace" in env_name:
                print("   Additional complexity: Grasping + lifting")
                
            print(f"   Why it's harder: ", end="")
            if i == 1:
                print("Baseline - just position control")
            elif i == 2:
                print("Must reason about object-robot interaction")
            elif i == 3:
                print("Requires precise force control for sliding")
            elif i == 4:
                print("Combines grasping, lifting, and placing")
                
            env.close()
            
        except Exception as e:
            print(f"   ✗ Error: {e}")


def test_action_execution():
    """Test how actions affect the robot."""
    print("\n" + "=" * 60)
    print("Testing Action Execution")
    print("=" * 60)
    
    env = gym.make("FetchReach-v3")
    obs, _ = env.reset()
    
    print("\nAction Space Breakdown:")
    print("4D continuous action: [dx, dy, dz, gripper]")
    print("- dx, dy, dz: End-effector displacement in Cartesian space")
    print("- gripper: -1 (open) to 1 (close)")
    
    print("\nTesting specific actions:")
    
    # Test moving in each direction
    test_actions = [
        ([0.1, 0, 0, 0], "Move right (+x)"),
        ([-0.1, 0, 0, 0], "Move left (-x)"),
        ([0, 0.1, 0, 0], "Move forward (+y)"),
        ([0, -0.1, 0, 0], "Move backward (-y)"),
        ([0, 0, 0.1, 0], "Move up (+z)"),
        ([0, 0, -0.1, 0], "Move down (-z)"),
        ([0, 0, 0, 1], "Close gripper"),
        ([0, 0, 0, -1], "Open gripper"),
    ]
    
    for action, description in test_actions[:4]:  # Test first 4 actions
        obs, reward, _, _, _ = env.step(action)
        print(f"\n{description}:")
        print(f"  New position: {obs['achieved_goal']}")
        print(f"  Distance to goal: {np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']):.3f}")
    
    env.close()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Gymnasium-Robotics Fetch Environment Explorer")
    print("=" * 60)
    
    # Test if environments are available
    try:
        import gymnasium_robotics
        print(f"✓ Gymnasium-Robotics version: {gymnasium_robotics.__version__}")
    except ImportError:
        print("✗ Gymnasium-Robotics not installed!")
        print("  Run: python scripts/setup_gymnasium_robotics.py")
        return
    
    # Explore each Fetch environment
    fetch_envs = [
        "FetchReach-v3",
        "FetchPush-v3",
        "FetchSlide-v3",
        "FetchPickAndPlace-v3"
    ]
    
    # Basic exploration
    for env_name in fetch_envs:
        explore_fetch_environment(env_name)
    
    # Detailed tests
    test_goal_conditioned_structure()
    compare_fetch_tasks()
    test_action_execution()
    
    print("\n" + "=" * 60)
    print("Exploration Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("1. All Fetch envs use goal-conditioned observations")
    print("2. Actions are 4D: 3D position + gripper")
    print("3. Success = distance to goal < 0.05m")
    print("4. Perfect for testing both PPO and GRPO")
    print("5. HER can significantly improve sample efficiency")


if __name__ == "__main__":
    main()