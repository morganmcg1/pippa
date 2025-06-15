#!/usr/bin/env python3
"""
Test PPO implementation with Isaac Lab Cartpole environment.
This script verifies our PPO can work with Isaac Lab environments.
"""

import os
import sys
import time
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Check if we're in Isaac Lab environment
try:
    # Import Isaac Lab modules
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab_tasks.classic_control.cartpole import CartpoleEnvCfg
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    print("WARNING: Isaac Lab not available. Using Gymnasium Cartpole instead.")
    ISAAC_LAB_AVAILABLE = False
    import gymnasium as gym

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from configs.ppo_config_v2 import PPOConfigV2
from utils.buffers import PPORolloutBuffer
from utils.logging import get_system_metrics

# Import WandB for tracking
import wandb
from dotenv import load_dotenv


def create_cartpole_env():
    """Create Cartpole environment (Isaac Lab or Gymnasium)."""
    if ISAAC_LAB_AVAILABLE:
        print("Creating Isaac Lab Cartpole environment...")
        # Create Isaac Lab Cartpole config
        env_cfg = CartpoleEnvCfg()
        env_cfg.scene.num_envs = 1  # Single environment for testing
        
        # Create environment
        env = ManagerBasedRLEnv(cfg=env_cfg)
        print(f"Isaac Lab environment created: {env}")
        return env, True
    else:
        print("Creating Gymnasium Cartpole environment...")
        env = gym.make("CartPole-v1")
        return env, False


def test_ppo_cartpole():
    """Test PPO on Cartpole environment."""
    print("\n" + "="*60)
    print("Testing PPO with Cartpole Environment")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize WandB
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"ppo_cartpole_test_{int(time.time())}",
        config={
            "algorithm": "PPO",
            "environment": "Cartpole",
            "isaac_lab": ISAAC_LAB_AVAILABLE,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        tags=["test", "ppo", "cartpole", "gr00t-rl", "gr00t-ppo-testing"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env, is_isaac = create_cartpole_env()
    
    # Get observation and action spaces
    if is_isaac:
        obs_space = env.observation_space
        act_space = env.action_space
        num_envs = env.num_envs
    else:
        obs_space = env.observation_space
        act_space = env.action_space
        num_envs = 1
    
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    print(f"Number of environments: {num_envs}")
    
    # Log environment info
    wandb.log({
        "env/name": "Cartpole",
        "env/isaac_lab": is_isaac,
        "env/num_envs": num_envs,
        "env/obs_shape": obs_space.shape,
        "env/act_shape": act_space.shape if hasattr(act_space, 'shape') else act_space.n
    })
    
    # Create PPO model
    # Determine if action space is discrete or continuous
    is_discrete = hasattr(act_space, 'n')
    if is_discrete:
        action_dim = act_space.n  # Discrete action space
    else:
        action_dim = act_space.shape[0]  # Continuous action space
    
    model = PPOGr00tActorCriticV2(
        observation_space=obs_space,
        action_dim=action_dim,
        use_multimodal_encoder=False,
        device=device
    ).to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    wandb.log({
        "model/total_parameters": total_params,
        "model/device": str(device)
    })
    
    print(f"Model created: {total_params:,} parameters")
    
    # Test environment interaction
    print("\nTesting environment interaction...")
    
    # Reset environment
    if is_isaac:
        obs_dict = env.reset()
        obs = obs_dict["policy"]  # Isaac Lab returns dict with "policy" key
    else:
        obs, _ = env.reset()
    
    episode_reward = 0
    episode_length = 0
    
    # Run one episode
    for step in range(500):  # Max 500 steps
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        
        # Get action from model
        with torch.no_grad():
            if hasattr(act_space, 'shape'):
                # Continuous action space
                action, _, _, value = model.get_action_and_value(obs_tensor)
                action_np = action.squeeze(0).cpu().numpy()
            else:
                # Discrete action space - need to sample from logits
                # For discrete actions, the model outputs action logits
                logits = model.actor(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
                action_np = action.cpu().numpy()[0]
                value = model.critic(obs_tensor)
        
        # Step environment
        if is_isaac:
            obs_dict = env.step(torch.tensor([action_np]))
            obs = obs_dict["policy"][0].cpu().numpy()
            reward = obs_dict["reward"][0].item()
            done = obs_dict["terminated"][0].item() or obs_dict["truncated"][0].item()
        else:
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        
        if done:
            break
    
    print(f"\nEpisode completed!")
    print(f"Episode reward: {episode_reward:.2f}")
    print(f"Episode length: {episode_length}")
    
    # Log results
    wandb.log({
        "test/episode_reward": episode_reward,
        "test/episode_length": episode_length,
        "test/success": True
    })
    
    # Get system metrics
    system_metrics = get_system_metrics()
    wandb.log(system_metrics)
    
    # Cleanup
    env.close()
    wandb.finish()
    
    print("\n✅ PPO Cartpole test completed successfully!")
    print(f"View results at: https://wandb.ai/{os.getenv('WANDB_ENTITY', 'wild-ai')}/{os.getenv('WANDB_PROJECT', 'pippa')}")
    
    return True


def main():
    """Main entry point."""
    print("PPO Isaac Lab Integration Test")
    print("==============================")
    
    # Load environment variables
    load_dotenv()
    
    # Ensure we have WandB credentials
    if not os.getenv("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not found")
        os.environ["WANDB_MODE"] = "offline"
    
    try:
        success = test_ppo_cartpole()
        return success
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()