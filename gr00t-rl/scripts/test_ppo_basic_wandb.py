#!/usr/bin/env python3
"""
Basic PPO test with WandB logging - simplified version.
"""

import os
import time
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import wandb
from dotenv import load_dotenv

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from utils.logging import get_system_metrics


def test_basic_ppo():
    """Basic PPO test with WandB logging."""
    print("\n" + "="*50)
    print("Basic PPO Test with WandB Logging")
    print("="*50)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize WandB
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"ppo_basic_test_{int(time.time())}",
        config={
            "algorithm": "PPO",
            "test_type": "basic_functionality",
            "environment": "Pendulum-v1",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        tags=["test", "ppo", "basic", "gr00t-rl", "gr00t-ppo-testing"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create single environment
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"Environment: Pendulum-v1")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Log environment info
    wandb.log({
        "env/name": "Pendulum-v1",
        "env/observation_dim": obs_dim,
        "env/action_dim": act_dim,
    })
    
    # Create model
    model = PPOGr00tActorCriticV2(
        observation_space=env.observation_space,
        action_dim=act_dim,
        use_multimodal_encoder=False,
        device=device
    ).to(device)
    
    # Calculate parameters
    actor_params = sum(p.numel() for p in model.actor.parameters())
    critic_params = sum(p.numel() for p in model.critic.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel created successfully!")
    print(f"Actor parameters: {actor_params:,}")
    print(f"Critic parameters: {critic_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Log model info
    wandb.log({
        "model/actor_parameters": actor_params,
        "model/critic_parameters": critic_params,
        "model/total_parameters": total_params,
        "model/device": str(device),
    })
    
    # Test forward pass
    print("\nTesting forward pass...")
    obs, _ = env.reset()
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Test action generation
        actions, log_probs, entropy, values = model.get_action_and_value(obs_tensor)
        
        print(f"Actions shape: {actions.shape}")
        print(f"Log probs shape: {log_probs.shape}")
        print(f"Values shape: {values.shape}")
        print(f"Entropy: {entropy.item():.4f}")
        
        # Log forward pass results
        wandb.log({
            "test/forward_pass_success": True,
            "test/action_mean": actions.mean().item(),
            "test/action_std": actions.std().item(),
            "test/value_mean": values.mean().item(),
            "test/entropy": entropy.item(),
        })
    
    # Test multiple steps
    print("\nTesting environment interaction...")
    episode_reward = 0
    episode_length = 0
    
    for step in range(100):
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs_tensor)
        
        action_np = action.squeeze(0).cpu().numpy()  # Only squeeze batch dimension
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        
        if done:
            break
            
        obs = next_obs
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    
    print(f"\nEpisode completed!")
    print(f"Episode reward: {episode_reward:.2f}")
    print(f"Episode length: {episode_length}")
    
    # Log episode results
    wandb.log({
        "test/episode_reward": episode_reward,
        "test/episode_length": episode_length,
    })
    
    # Get system metrics
    system_metrics = get_system_metrics()
    wandb.log(system_metrics)
    
    # Log GPU memory if available
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        
        wandb.log({
            "gpu/memory_used_gb": gpu_memory_used,
            "gpu/memory_reserved_gb": gpu_memory_reserved,
        })
        
        print(f"\nGPU Memory:")
        print(f"Used: {gpu_memory_used:.2f} GB")
        print(f"Reserved: {gpu_memory_reserved:.2f} GB")
    
    # Test complete
    wandb.log({
        "test/complete": True,
        "test/status": "success"
    })
    
    print("\n✅ Basic PPO test completed successfully!")
    
    # Cleanup
    env.close()
    wandb.finish()
    
    return True


def main():
    """Run the basic test."""
    print("Basic PPO Test Suite")
    print("===================")
    
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not found")
        print("Setting offline mode...")
        os.environ["WANDB_MODE"] = "offline"
    
    try:
        success = test_basic_ppo()
        if success:
            print("\n✅ All tests passed!")
            print(f"\nView results at: https://wandb.ai/{os.getenv('WANDB_ENTITY', 'wild-ai')}/{os.getenv('WANDB_PROJECT', 'pippa')}")
        return success
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()