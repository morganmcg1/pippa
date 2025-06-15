#!/usr/bin/env python3
"""
Explore Fetch environment behavior to understand rewards and success conditions.
"""

import numpy as np
import gymnasium as gym
import gymnasium_robotics

def explore_fetch_reach():
    """Explore FetchReach environment."""
    print("Exploring FetchReach-v3 Environment")
    print("=" * 60)
    
    env = gym.make("FetchReach-v3")
    
    # Reset and examine initial state
    obs, info = env.reset(seed=42)
    
    print("\nInitial observation keys:", obs.keys())
    print(f"Observation shape: {obs['observation'].shape}")
    print(f"Achieved goal shape: {obs['achieved_goal'].shape}")
    print(f"Desired goal shape: {obs['desired_goal'].shape}")
    
    print(f"\nInitial achieved goal: {obs['achieved_goal']}")
    print(f"Initial desired goal: {obs['desired_goal']}")
    print(f"Initial distance: {np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']):.4f}")
    
    # Test reward computation
    reward = env.unwrapped.compute_reward(
        obs['achieved_goal'], 
        obs['desired_goal'], 
        {}
    )
    print(f"\nInitial reward: {reward}")
    
    # Test what happens when goals match
    reward_success = env.unwrapped.compute_reward(
        obs['desired_goal'], 
        obs['desired_goal'], 
        {}
    )
    print(f"Reward when goal achieved: {reward_success}")
    
    # Run a few random actions
    print("\n" + "-"*60)
    print("Running 10 random actions...")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Distance to goal: {distance:.4f}")
        print(f"  Success: {info.get('is_success', False)}")
        
        if terminated or truncated:
            print("  Episode ended!")
            break
    
    # Test with zero actions (no movement)
    print("\n" + "-"*60)
    print("Testing with zero actions...")
    
    obs, info = env.reset(seed=42)
    initial_distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
    
    for i in range(5):
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        
        print(f"\nStep {i+1} (zero action):")
        print(f"  Reward: {reward}")
        print(f"  Distance: {distance:.4f} (initial: {initial_distance:.4f})")
    
    # Test goal tolerance
    print("\n" + "-"*60)
    print("Testing goal tolerance...")
    
    # Get close to goal
    goal = obs['desired_goal'].copy()
    for tolerance in [0.1, 0.05, 0.01, 0.005, 0.001]:
        achieved = goal + np.array([tolerance, 0, 0])
        reward = env.unwrapped.compute_reward(achieved, goal, {})
        print(f"  Distance {tolerance:.3f}: reward = {reward}")
    
    env.close()
    print("\n✅ Exploration complete!")

def test_different_reward_modes():
    """Test different reward modes with our wrapper."""
    from environments.fetch_wrapper import FetchGoalWrapper
    
    print("\n" + "="*60)
    print("Testing Different Reward Modes")
    print("="*60)
    
    for reward_mode in ["sparse", "dense", "distance"]:
        print(f"\n{reward_mode.upper()} reward mode:")
        
        env = gym.make("FetchReach-v3")
        env = FetchGoalWrapper(env, reward_mode=reward_mode)
        
        obs, info = env.reset(seed=42)
        
        # Take a few steps
        total_reward = 0
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"  Step {i+1}: reward = {reward:.3f}, "
                  f"distance = {info['distance_to_goal']:.3f}, "
                  f"sparse = {info['sparse_reward']}")
        
        print(f"  Total reward: {total_reward:.3f}")
        env.close()

if __name__ == "__main__":
    explore_fetch_reach()
    
    # Import after path setup
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    test_different_reward_modes()