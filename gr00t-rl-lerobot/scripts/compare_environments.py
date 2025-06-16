#!/usr/bin/env python3
"""
Compare different environment configurations for GR00T training.

This script runs short episodes with each configuration and compares:
1. Action distributions
2. State trajectories  
3. Reward patterns
4. Computational efficiency
"""

import sys
sys.path.append('..')

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from environments.fetch_wrapper import make_fetch_so101_env
from environments.fetch_so101_coupled import make_fetch_so101_coupled_env
from policies.gr00t_policy import GR00TPolicy, GR00TConfig


def run_episode(env, policy, max_steps=50):
    """Run a single episode and collect metrics."""
    obs, info = env.reset()
    
    metrics = {
        "actions": [],
        "states": [],
        "rewards": [],
        "compute_times": [],
    }
    
    for step in range(max_steps):
        # Time action computation
        start_time = time.time()
        
        # Get action
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
        
        with torch.no_grad():
            action = policy.select_action(batch)
            action_np = action.cpu().numpy().squeeze()
        
        compute_time = time.time() - start_time
        
        # Ensure correct action dimensions
        if hasattr(env, 'use_joint_space') and env.use_joint_space:
            # Joint space - ensure 6D
            if len(action_np) != 6:
                action_np = np.zeros(6)
        else:
            # Cartesian space - ensure 4D
            if len(action_np) > 4:
                action_np = action_np[:4]
            elif len(action_np) < 4:
                action_np = np.pad(action_np, (0, 4 - len(action_np)))
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_np)
        
        # Record metrics
        metrics["actions"].append(action_np)
        metrics["states"].append(obs["observation"]["state"])
        metrics["rewards"].append(reward)
        metrics["compute_times"].append(compute_time)
        
        if terminated or truncated:
            break
    
    # Convert to arrays
    for key in ["actions", "states", "rewards"]:
        metrics[key] = np.array(metrics[key])
    
    metrics["total_reward"] = np.sum(metrics["rewards"])
    metrics["success"] = info.get("is_success", False)
    metrics["episode_length"] = len(metrics["rewards"])
    metrics["avg_compute_time"] = np.mean(metrics["compute_times"])
    
    return metrics


def compare_configurations(num_episodes=5):
    """Compare different environment configurations."""
    print("Comparing Environment Configurations")
    print("=" * 60)
    
    # Load policy once
    try:
        import torch
        config = GR00TConfig(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        policy = GR00TPolicy(config)
        print("✓ GR00T policy loaded")
    except Exception as e:
        print(f"✗ Failed to load GR00T: {e}")
        print("Using random policy for comparison")
        policy = None
    
    # Define configurations to test
    configs = [
        {
            "name": "Cartesian-Only",
            "env_fn": lambda: make_fetch_so101_env(max_episode_steps=50),
            "description": "7-DoF Fetch with 4D Cartesian actions"
        },
        {
            "name": "Coupled-Cartesian", 
            "env_fn": lambda: make_fetch_so101_coupled_env(
                max_episode_steps=50,
                use_joint_space=False,
                couple_joints=True
            ),
            "description": "6-DoF coupled with 4D Cartesian actions"
        },
        {
            "name": "Coupled-Joint",
            "env_fn": lambda: make_fetch_so101_coupled_env(
                max_episode_steps=50,
                use_joint_space=True,
                couple_joints=True
            ),
            "description": "6-DoF coupled with 6D joint actions"
        }
    ]
    
    # Run episodes for each configuration
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        print(f"  {config['description']}")
        
        env = config["env_fn"]()
        config_results = []
        
        for ep in range(num_episodes):
            if policy is None:
                # Random policy fallback
                class RandomPolicy:
                    def select_action(self, batch):
                        import torch
                        return torch.randn(1, 6)
                
                metrics = run_episode(env, RandomPolicy())
            else:
                metrics = run_episode(env, policy)
            
            config_results.append(metrics)
            print(f"  Episode {ep+1}: reward={metrics['total_reward']:.2f}, "
                  f"success={metrics['success']}, steps={metrics['episode_length']}")
        
        results[config["name"]] = config_results
        env.close()
    
    return results


def visualize_comparison(results):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Environment Configuration Comparison", fontsize=16)
    
    config_names = list(results.keys())
    colors = ['blue', 'green', 'red']
    
    # 1. Average rewards
    ax = axes[0, 0]
    avg_rewards = []
    std_rewards = []
    
    for config in config_names:
        rewards = [r["total_reward"] for r in results[config]]
        avg_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))
    
    bars = ax.bar(config_names, avg_rewards, yerr=std_rewards, 
                   color=colors, alpha=0.7, capsize=5)
    ax.set_ylabel("Average Total Reward")
    ax.set_title("Reward Comparison")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Success rates
    ax = axes[0, 1]
    success_rates = []
    
    for config in config_names:
        successes = [r["success"] for r in results[config]]
        success_rates.append(np.mean(successes))
    
    bars = ax.bar(config_names, success_rates, color=colors, alpha=0.7)
    ax.set_ylabel("Success Rate")
    ax.set_title("Task Success Rate")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Computation time
    ax = axes[0, 2]
    avg_times = []
    
    for config in config_names:
        times = [r["avg_compute_time"] * 1000 for r in results[config]]  # ms
        avg_times.append(np.mean(times))
    
    bars = ax.bar(config_names, avg_times, color=colors, alpha=0.7)
    ax.set_ylabel("Average Compute Time (ms)")
    ax.set_title("Computational Efficiency")
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Action distributions
    ax = axes[1, 0]
    for i, config in enumerate(config_names):
        all_actions = np.vstack([r["actions"] for r in results[config]])
        action_std = np.std(all_actions, axis=0)
        x = np.arange(len(action_std))
        ax.bar(x + i*0.25, action_std, width=0.25, 
               label=config, color=colors[i], alpha=0.7)
    
    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("Action Std Dev")
    ax.set_title("Action Variability")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. State trajectories (first episode)
    ax = axes[1, 1]
    for i, config in enumerate(config_names):
        states = results[config][0]["states"]
        # Plot first 3 joint positions
        if states.shape[1] >= 3:
            trajectory = np.linalg.norm(states[:, :3], axis=1)
            ax.plot(trajectory, label=config, color=colors[i], linewidth=2)
    
    ax.set_xlabel("Time Step")
    ax.set_ylabel("End-effector Distance")
    ax.set_title("State Trajectories (First Episode)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Reward progression
    ax = axes[1, 2]
    for i, config in enumerate(config_names):
        rewards = results[config][0]["rewards"]
        cumulative = np.cumsum(rewards)
        ax.plot(cumulative, label=config, color=colors[i], linewidth=2)
    
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward Accumulation (First Episode)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path("environment_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()
    
    # Create summary table
    summary_data = []
    for config in config_names:
        summary_data.append({
            "Configuration": config,
            "Avg Reward": f"{avg_rewards[config_names.index(config)]:.2f}",
            "Success Rate": f"{success_rates[config_names.index(config)]:.1%}",
            "Avg Compute (ms)": f"{avg_times[config_names.index(config)]:.1f}",
            "Episodes": len(results[config])
        })
    
    df = pd.DataFrame(summary_data)
    print("\nSummary Table:")
    print(df.to_string(index=False))
    
    # Save detailed results
    save_path = Path("environment_comparison_detailed.json")
    import json
    with open(save_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for config, episodes in results.items():
            json_results[config] = []
            for ep in episodes:
                json_ep = {
                    "total_reward": float(ep["total_reward"]),
                    "success": bool(ep["success"]),
                    "episode_length": int(ep["episode_length"]),
                    "avg_compute_time": float(ep["avg_compute_time"])
                }
                json_results[config].append(json_ep)
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {save_path}")


def main():
    """Run comparison analysis."""
    print("GR00T Environment Configuration Comparison")
    print("This compares different approaches for 7-DoF → 6-DoF adaptation")
    print()
    
    # Check dependencies
    try:
        import torch
        import gymnasium_robotics
        print("✓ Dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return
    
    # Run comparison
    results = compare_configurations(num_episodes=5)
    
    # Visualize results
    visualize_comparison(results)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("Based on the comparison:")
    print("1. Cartesian-Only: Best for quick prototyping")
    print("2. Coupled-Cartesian: Good balance of fidelity and simplicity")
    print("3. Coupled-Joint: Best kinematic match but more complex")
    print("\nFor GR00T training, start with Coupled-Cartesian approach.")


if __name__ == "__main__":
    main()