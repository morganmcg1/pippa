#!/usr/bin/env python3
"""
Test PPO implementation with Isaac Lab or Gymnasium Pendulum environment.
This uses continuous action space which is supported by our GaussianActor.
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
ISAAC_LAB_AVAILABLE = False
try:
    # Import Isaac Lab modules
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.utils import configclass
    from omni.isaac.lab_tasks.manager_based.classic.pendulum import PendulumEnvCfg
    ISAAC_LAB_AVAILABLE = True
    print("Isaac Lab modules loaded successfully!")
except ImportError as e:
    print(f"WARNING: Isaac Lab not available: {e}")
    print("Using Gymnasium Pendulum instead.")
    import gymnasium as gym

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from configs.ppo_config_v2 import PPOConfigV2
from utils.buffers import PPORolloutBuffer
from utils.logging import get_system_metrics
from utils.normalization import ObservationNormalizer

# Import WandB for tracking
import wandb
from dotenv import load_dotenv


def create_pendulum_env():
    """Create Pendulum environment (Isaac Lab or Gymnasium)."""
    if ISAAC_LAB_AVAILABLE:
        print("Creating Isaac Lab Pendulum environment...")
        try:
            # Create Isaac Lab Pendulum config
            env_cfg = PendulumEnvCfg()
            env_cfg.scene.num_envs = 4  # 4 parallel environments
            
            # Create environment
            env = ManagerBasedRLEnv(cfg=env_cfg)
            print(f"Isaac Lab environment created: {env}")
            return env, True
        except Exception as e:
            print(f"Failed to create Isaac Lab environment: {e}")
            print("Falling back to Gymnasium...")
            ISAAC_LAB_AVAILABLE = False
    
    print("Creating Gymnasium Pendulum environment...")
    env = gym.make("Pendulum-v1")
    # Wrap with observation normalizer
    env = ObservationNormalizer(env, device="cuda" if torch.cuda.is_available() else "cpu")
    return env, False


def test_ppo_pendulum():
    """Test PPO on Pendulum environment."""
    print("\n" + "="*60)
    print("Testing PPO with Pendulum Environment")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize WandB
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"ppo_pendulum_isaac_test_{int(time.time())}",
        config={
            "algorithm": "PPO",
            "environment": "Pendulum",
            "isaac_lab": ISAAC_LAB_AVAILABLE,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        tags=["test", "ppo", "pendulum", "gr00t-rl", "gr00t-ppo-testing", "isaac-lab"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env, is_isaac = create_pendulum_env()
    
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
        "env/name": "Pendulum",
        "env/isaac_lab": is_isaac,
        "env/num_envs": num_envs,
        "env/obs_shape": obs_space.shape,
        "env/act_shape": act_space.shape
    })
    
    # Create PPO model
    action_dim = act_space.shape[0]
    
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
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    
    # Test environment interaction
    print("\nTesting environment interaction...")
    
    # Reset environment
    if is_isaac:
        obs_dict = env.reset()
        obs = obs_dict["policy"]  # Isaac Lab returns dict with "policy" key
    else:
        obs, _ = env.reset()
        obs = np.expand_dims(obs, 0)  # Add batch dimension
    
    # Track metrics
    episode_rewards = []
    episode_lengths = []
    
    # Run multiple steps to test rollout collection
    num_steps = 128
    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_values = []
    all_log_probs = []
    
    print(f"Collecting {num_steps} steps...")
    
    for step in range(num_steps):
        # Convert observation to tensor
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().to(device)
        else:
            obs_tensor = obs.to(device)
        
        # Get action from model
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(obs_tensor)
            action_np = action.cpu().numpy()
        
        # Store data
        all_obs.append(obs_tensor)
        all_actions.append(action)
        all_rewards.append(torch.zeros(num_envs if is_isaac else 1))
        all_dones.append(torch.zeros(num_envs if is_isaac else 1))
        all_values.append(value)
        all_log_probs.append(log_prob)
        
        # Step environment
        if is_isaac:
            obs_dict = env.step(action_np)
            obs = obs_dict["policy"]
            rewards = obs_dict["reward"]
            dones = obs_dict["terminated"] | obs_dict["truncated"]
            
            # Update stored rewards/dones
            all_rewards[-1] = torch.from_numpy(rewards).to(device)
            all_dones[-1] = torch.from_numpy(dones.astype(np.float32)).to(device)
            
            # Track episodes
            for i in range(num_envs):
                if dones[i]:
                    episode_rewards.append(obs_dict["episode_reward"][i])
                    episode_lengths.append(obs_dict["episode_length"][i])
        else:
            obs, reward, terminated, truncated, info = env.step(action_np[0])
            done = terminated or truncated
            
            # Update stored rewards/dones
            all_rewards[-1] = torch.tensor([reward]).to(device)
            all_dones[-1] = torch.tensor([float(done)]).to(device)
            
            # Track episode
            if done:
                episode_rewards.append(info.get("episode", {}).get("r", 0))
                episode_lengths.append(info.get("episode", {}).get("l", 0))
                obs, _ = env.reset()
            
            obs = np.expand_dims(obs, 0)  # Add batch dimension
    
    print(f"\nCollected {num_steps} steps successfully!")
    
    # Test PPO update
    print("\nTesting PPO update...")
    
    # Stack all data
    batch_obs = torch.stack(all_obs)
    batch_actions = torch.stack(all_actions)
    batch_log_probs = torch.stack(all_log_probs)
    batch_values = torch.stack(all_values)
    batch_rewards = torch.stack(all_rewards)
    batch_dones = torch.stack(all_dones)
    
    # Compute advantages (simplified - no GAE for this test)
    with torch.no_grad():
        next_value = model.get_value(obs_tensor)
        returns = batch_rewards + 0.99 * next_value * (1 - batch_dones)
        advantages = returns - batch_values
        
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Flatten batch
    batch_size = num_steps * (num_envs if is_isaac else 1)
    batch_obs = batch_obs.reshape(batch_size, -1)
    batch_actions = batch_actions.reshape(batch_size, -1)
    batch_log_probs = batch_log_probs.reshape(batch_size)
    batch_advantages = advantages.reshape(batch_size)
    batch_returns = returns.reshape(batch_size)
    batch_values = batch_values.reshape(batch_size)
    
    # Perform one PPO update
    # Get current policy outputs
    new_log_probs, new_values, entropy = model.evaluate_actions(batch_obs, batch_actions)
    
    # Compute ratio
    ratio = torch.exp(new_log_probs - batch_log_probs)
    
    # Compute losses
    pg_loss1 = -batch_advantages * ratio
    pg_loss2 = -batch_advantages * torch.clamp(ratio, 0.8, 1.2)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    
    value_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()
    entropy_loss = entropy.mean()
    
    total_loss = pg_loss + 0.5 * value_loss - 0.01 * entropy_loss
    
    # Optimize
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    print(f"PPO update completed!")
    print(f"  Policy loss: {pg_loss.item():.4f}")
    print(f"  Value loss: {value_loss.item():.4f}")
    print(f"  Entropy: {entropy_loss.item():.4f}")
    
    # Log results
    wandb.log({
        "test/steps_collected": num_steps,
        "test/episodes_completed": len(episode_rewards),
        "test/policy_loss": pg_loss.item(),
        "test/value_loss": value_loss.item(),
        "test/entropy": entropy_loss.item(),
        "test/mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0,
        "test/success": True
    })
    
    # Get system metrics
    system_metrics = get_system_metrics()
    wandb.log(system_metrics)
    
    # Cleanup
    env.close()
    wandb.finish()
    
    print("\n✅ PPO Pendulum test completed successfully!")
    print(f"View results at: https://wandb.ai/{os.getenv('WANDB_ENTITY', 'wild-ai')}/{os.getenv('WANDB_PROJECT', 'pippa')}")
    
    return True


def main():
    """Main entry point."""
    print("PPO Isaac Lab Integration Test (Pendulum)")
    print("=========================================")
    
    # Load environment variables
    load_dotenv()
    
    # Ensure we have WandB credentials
    if not os.getenv("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not found")
        os.environ["WANDB_MODE"] = "offline"
    
    try:
        success = test_ppo_pendulum()
        return success
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to finish WandB run if it was started
        try:
            wandb.finish()
        except:
            pass
            
        return False


if __name__ == "__main__":
    main()