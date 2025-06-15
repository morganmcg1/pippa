#!/usr/bin/env python3
"""
PPO test with comprehensive WandB logging.
Tests functionality and logs all metrics to wandb.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

import wandb
from dotenv import load_dotenv

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from configs.ppo_config_v2 import PPOConfigV2
from utils.logging import get_system_metrics
from utils.buffers import PPORolloutBuffer
from utils.normalization import VecNormalize
from environments.vec_isaac_env import make_vec_env, DummyVecEnv


def test_pendulum_with_logging(num_steps=1000):
    """Test PPO on Pendulum-v1 with full WandB logging."""
    print("\n" + "="*50)
    print("Testing PPO V2 on Pendulum-v1 with WandB Logging")
    print("="*50)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize WandB
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"ppo_test_pendulum_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "algorithm": "PPO",
            "environment": "Pendulum-v1",
            "test_type": "functionality",
            "num_steps": num_steps,
            "num_envs": 4,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        tags=["test", "ppo", "pendulum", "gr00t-rl"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create vectorized environments
    num_envs = 4
    envs = make_vec_env(
        "Pendulum-v1",
        n_envs=num_envs,
        vec_env_cls=DummyVecEnv
    )
    
    # Add normalization wrapper
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True)
    
    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]
    
    # Log environment info
    wandb.log({
        "env/observation_dim": obs_dim,
        "env/action_dim": act_dim,
        "env/num_envs": num_envs
    })
    
    # Create model
    model = PPOGr00tActorCriticV2(
        observation_space=envs.observation_space,
        action_dim=act_dim,
        use_multimodal_encoder=False,
        device=device
    ).to(device)
    
    # Log model info
    actor_params = sum(p.numel() for p in model.actor.parameters())
    critic_params = sum(p.numel() for p in model.critic.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    wandb.log({
        "model/actor_parameters": actor_params,
        "model/critic_parameters": critic_params,
        "model/total_parameters": total_params
    })
    
    print(f"Model created: {total_params:,} parameters")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        eps=1e-5  # Important PPO detail
    )
    
    # Create rollout buffer
    buffer = PPORolloutBuffer(
        buffer_size=128,  # Steps per env before update
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # Training metrics
    global_step = 0
    start_time = time.time()
    episode_rewards = []
    episode_lengths = []
    
    # Reset environments
    obs_data = envs.reset()
    if isinstance(obs_data, tuple):
        obs = obs_data[0]  # New gym API returns (obs, info)
    else:
        obs = obs_data
    
    # Collect some rollouts and update
    for iteration in range(10):  # 10 updates for testing
        iteration_start = time.time()
        
        # Collect rollouts
        for step in range(128):  # Steps per update
            obs_tensor = torch.from_numpy(obs).float().to(device)
            
            with torch.no_grad():
                actions, log_probs, _, values = model.get_action_and_value(obs_tensor)
            
            # Step environment
            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, infos = envs.step(actions_np)
            
            # Store in buffer
            buffer.add(
                obs=obs,
                actions=actions_np,
                rewards=rewards,
                dones=dones,
                values=values.cpu().numpy(),
                log_probs=log_probs.cpu().numpy()
            )
            
            # Track episode statistics
            for info in infos:
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])
                    
                    # Log episode metrics
                    wandb.log({
                        "charts/episodic_return": info["episode"]["r"],
                        "charts/episodic_length": info["episode"]["l"],
                        "charts/global_step": global_step
                    }, step=global_step)
            
            obs = next_obs
            global_step += num_envs
        
        # Compute returns
        with torch.no_grad():
            last_values = model.get_value(torch.from_numpy(next_obs).float().to(device))
        buffer.compute_returns_and_advantages(
            last_values=last_values,
            dones=torch.from_numpy(dones).to(device)
        )
        
        # Get batch data
        batch_data = buffer.get()
        
        # Perform PPO update
        pg_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clipfracs = []
        
        # Multiple epochs
        for epoch in range(10):
            # Mini-batch updates
            for mb_inds in np.array_split(np.random.permutation(buffer.buffer_size * num_envs), 4):
                mb_obs = batch_data['observations'][mb_inds]
                mb_actions = batch_data['actions'][mb_inds]
                mb_log_probs = batch_data['log_probs'][mb_inds]
                mb_advantages = batch_data['advantages'][mb_inds]
                mb_returns = batch_data['returns'][mb_inds]
                mb_values = batch_data['values'][mb_inds]
                
                # Get current policy outputs
                new_log_probs, entropy, new_values = model.evaluate_actions(mb_obs, mb_actions)
                
                # Ratio for PPO
                logratio = new_log_probs - mb_log_probs
                ratio = logratio.exp()
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - 0.01 * entropy_loss + 0.5 * v_loss
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                # Track metrics
                pg_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > 0.2).float().mean().item()
                    approx_kls.append(approx_kl)
                    clipfracs.append(clipfrac)
        
        # Clear buffer
        buffer.reset()
        
        # Calculate iteration time
        iteration_time = time.time() - iteration_start
        sps = (128 * num_envs) / iteration_time
        
        # Get system metrics
        system_metrics = get_system_metrics()
        
        # Log all metrics
        metrics = {
            "train/policy_loss": np.mean(pg_losses),
            "train/value_loss": np.mean(value_losses),
            "train/entropy_loss": np.mean(entropy_losses),
            "train/approx_kl": np.mean(approx_kls),
            "train/clipfrac": np.mean(clipfracs),
            "train/learning_rate": 3e-4,
            "train/updates": iteration + 1,
            "charts/SPS": sps,
            "charts/global_step": global_step,
            "time/iteration_time": iteration_time,
            "time/total_time": time.time() - start_time,
            **system_metrics
        }
        
        if episode_rewards:
            metrics.update({
                "charts/mean_episodic_return": np.mean(episode_rewards[-100:]),
                "charts/mean_episodic_length": np.mean(episode_lengths[-100:]),
            })
        
        wandb.log(metrics, step=global_step)
        
        # Console output
        print(f"\nIteration {iteration + 1}/10")
        print(f"  Global step: {global_step}")
        print(f"  SPS: {sps:.0f}")
        print(f"  Policy loss: {metrics['train/policy_loss']:.4f}")
        print(f"  Value loss: {metrics['train/value_loss']:.4f}")
        print(f"  Entropy: {metrics['train/entropy_loss']:.4f}")
        print(f"  Approx KL: {metrics['train/approx_kl']:.4f}")
        if episode_rewards:
            print(f"  Mean return: {metrics['charts/mean_episodic_return']:.2f}")
    
    # Final summary
    final_metrics = {
        "test/final_mean_return": np.mean(episode_rewards[-100:]) if episode_rewards else 0,
        "test/total_episodes": len(episode_rewards),
        "test/total_steps": global_step,
        "test/total_time": time.time() - start_time,
        "test/success": True
    }
    wandb.log(final_metrics)
    
    print(f"\nTest completed successfully!")
    print(f"Final mean return: {final_metrics['test/final_mean_return']:.2f}")
    print(f"Total episodes: {final_metrics['test/total_episodes']}")
    print(f"Total steps: {final_metrics['test/total_steps']}")
    
    # Cleanup
    envs.close()
    wandb.finish()
    
    return True


def test_multimodal_with_logging():
    """Test multi-modal support with WandB logging."""
    print("\n" + "="*50)
    print("Testing Multi-modal PPO with WandB Logging")
    print("="*50)
    
    # Initialize WandB
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"ppo_test_multimodal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "algorithm": "PPO",
            "test_type": "multimodal",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        tags=["test", "ppo", "multimodal", "gr00t-rl"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create multi-modal observation space
    obs_space = gym.spaces.Dict({
        'proprioception': gym.spaces.Box(-1, 1, shape=(12,)),
        'vision': gym.spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8),
        'language': gym.spaces.Box(-1, 1, shape=(768,))
    })
    act_space = gym.spaces.Box(-1, 1, shape=(6,))
    
    # Log configuration
    wandb.log({
        "env/modalities": list(obs_space.spaces.keys()),
        "env/proprioception_dim": 12,
        "env/vision_shape": [84, 84, 3],
        "env/language_dim": 768,
        "env/action_dim": 6
    })
    
    try:
        # Create model
        model = PPOGr00tActorCriticV2(
            observation_space=obs_space,
            action_dim=act_space.shape[0],
            use_multimodal_encoder=True,
            vision_dim=256,
            proprio_dim=12,
            language_dim=768,
            device=device
        ).to(device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        wandb.log({
            "model/total_parameters": total_params,
            "model/multimodal": True
        })
        
        print(f"Multi-modal model created: {total_params:,} parameters")
        
        # Test forward pass
        batch_size = 4
        obs = {
            'proprioception': torch.randn(batch_size, 12).to(device),
            'vision': torch.randn(batch_size, 3, 84, 84).to(device),
            'language': torch.randn(batch_size, 768).to(device)
        }
        
        with torch.no_grad():
            actions, log_probs, entropy, values = model.get_action_and_value(obs)
        
        # Log test results
        wandb.log({
            "test/multimodal_forward_pass": True,
            "test/action_shape": list(actions.shape),
            "test/value_shape": list(values.shape),
            "test/entropy": entropy.mean().item()
        })
        
        print("Multi-modal test passed!")
        success = True
        
    except Exception as e:
        print(f"Multi-modal test failed: {e}")
        wandb.log({
            "test/multimodal_forward_pass": False,
            "test/error": str(e)
        })
        success = False
    
    wandb.finish()
    return success


def main():
    """Run all tests with WandB logging."""
    print("PPO Tests with WandB Logging")
    print("============================")
    
    # Load environment variables
    load_dotenv()
    
    # Ensure we have WandB credentials
    if not os.getenv("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not found in environment")
        print("Running in offline mode - results will not be uploaded to WandB")
        os.environ["WANDB_MODE"] = "offline"
    
    tests = [
        ("Pendulum Training Test", test_pendulum_with_logging),
        ("Multi-modal Test", test_multimodal_with_logging)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    all_passed = True
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed! Check WandB for detailed metrics.")
        print(f"Project: https://wandb.ai/{os.getenv('WANDB_ENTITY', 'wild-ai')}/{os.getenv('WANDB_PROJECT', 'pippa')}")
    else:
        print("\n❌ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    main()