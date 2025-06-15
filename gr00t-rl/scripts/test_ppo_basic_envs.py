#!/usr/bin/env python3
"""
Test PPO on multiple basic environments.
Based on the working test_ppo_wandb.py, tests CartPole, Pendulum, and others.
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


def test_environment(env_name, num_iterations=10, num_envs=4, use_wandb=True):
    """Test PPO on a specific environment."""
    print(f"\n{'='*60}")
    print(f"Testing PPO V2 on {env_name}")
    print(f"{'='*60}")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize WandB if requested
    if use_wandb:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "pippa"),
            entity=os.getenv("WANDB_ENTITY", "wild-ai"),
            name=f"ppo_test_{env_name.lower().replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "algorithm": "PPO",
                "environment": env_name,
                "test_type": "basic_env_test",
                "num_iterations": num_iterations,
                "num_envs": num_envs,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            tags=["test", "ppo", env_name.lower(), "gr00t-rl", "gr00t-ppo-testing"]
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create vectorized environments
        envs = make_vec_env(
            env_name,
            n_envs=num_envs,
            vec_env_cls=DummyVecEnv
        )
        
        # Check if continuous or discrete
        is_continuous = isinstance(envs.action_space, gym.spaces.Box)
        
        # Add normalization wrapper for continuous control
        if is_continuous:
            envs = VecNormalize(envs, norm_obs=True, norm_reward=True)
        
        # Get dimensions
        if isinstance(envs.observation_space, gym.spaces.Box):
            obs_dim = envs.observation_space.shape[0]
        else:
            obs_dim = envs.observation_space.n
            
        if is_continuous:
            act_dim = envs.action_space.shape[0]
        else:
            act_dim = envs.action_space.n
        
        print(f"Environment: {env_name}")
        print(f"  Observation space: {envs.observation_space}")
        print(f"  Action space: {envs.action_space}")
        print(f"  Continuous: {is_continuous}")
        
        # Log environment info
        if use_wandb:
            wandb.log({
                "env/name": env_name,
                "env/observation_dim": obs_dim,
                "env/action_dim": act_dim,
                "env/num_envs": num_envs,
                "env/is_continuous": is_continuous
            })
        
        # Create model
        model = PPOGr00tActorCriticV2(
            observation_space=envs.observation_space,
            action_dim=act_dim,
            use_multimodal_encoder=False,
            continuous=is_continuous,  # This might need to be added to the model
            device=device
        ).to(device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        if use_wandb:
            wandb.log({
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
            gae_lambda=0.95,
            n_envs=num_envs
        )
        
        # Training metrics
        global_step = 0
        start_time = time.time()
        episode_rewards = []
        episode_lengths = []
        
        # Reset environments
        obs_data = envs.reset()
        if isinstance(obs_data, tuple):
            obs = obs_data[0]
        else:
            obs = obs_data
        
        # Collect some rollouts and update
        for iteration in range(num_iterations):
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
                    obs=obs_tensor,
                    action=actions,
                    reward=torch.from_numpy(rewards).to(device),
                    done=torch.from_numpy(dones).to(device),
                    value=values,
                    log_prob=log_probs
                )
                
                # Track episode statistics
                for info in infos:
                    if "episode" in info:
                        episode_rewards.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
                        
                        if use_wandb:
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
            
            # Perform PPO update (simplified for testing)
            pg_losses = []
            value_losses = []
            
            # Multiple epochs
            for epoch in range(5):  # Fewer epochs for testing
                # Mini-batch updates
                batch_size = 128 * num_envs
                minibatch_size = batch_size // 4
                
                for mb_start in range(0, batch_size, minibatch_size):
                    mb_end = mb_start + minibatch_size
                    mb_inds = np.arange(mb_start, mb_end)
                    
                    mb_obs = batch_data['observations'][mb_inds]
                    mb_actions = batch_data['actions'][mb_inds]
                    mb_log_probs = batch_data['log_probs'][mb_inds]
                    mb_advantages = batch_data['advantages'][mb_inds]
                    mb_returns = batch_data['returns'][mb_inds]
                    mb_values = batch_data['values'][mb_inds]
                    
                    # Get current policy outputs
                    new_log_probs, new_values, entropy = model.evaluate_actions(mb_obs, mb_actions)
                    
                    # Ratio for PPO
                    logratio = new_log_probs - mb_log_probs
                    ratio = logratio.exp()
                    
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    v_loss = 0.5 * ((new_values.flatten() - mb_returns) ** 2).mean()
                    
                    # Total loss
                    loss = pg_loss + 0.5 * v_loss - 0.01 * entropy
                    
                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    
                    pg_losses.append(pg_loss.item())
                    value_losses.append(v_loss.item())
            
            # Clear buffer
            buffer.reset()
            
            # Calculate iteration time
            iteration_time = time.time() - iteration_start
            sps = (128 * num_envs) / iteration_time
            
            # Log metrics
            if use_wandb:
                metrics = {
                    "train/policy_loss": np.mean(pg_losses),
                    "train/value_loss": np.mean(value_losses),
                    "train/learning_rate": 3e-4,
                    "charts/SPS": sps,
                    "charts/global_step": global_step,
                }
                
                if episode_rewards:
                    metrics.update({
                        "charts/mean_episodic_return": np.mean(episode_rewards[-100:]),
                        "charts/mean_episodic_length": np.mean(episode_lengths[-100:]),
                    })
                
                wandb.log(metrics, step=global_step)
            
            # Console output
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            print(f"  SPS: {sps:.0f}")
            print(f"  Policy loss: {np.mean(pg_losses):.4f}")
            print(f"  Value loss: {np.mean(value_losses):.4f}")
            if episode_rewards:
                print(f"  Mean return: {np.mean(episode_rewards[-100:]):.2f}")
        
        # Final summary
        success = True
        final_return = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else (np.mean(episode_rewards) if episode_rewards else 0)
        
        print(f"\nTest completed successfully!")
        print(f"Final mean return: {final_return:.2f}")
        print(f"Total episodes: {len(episode_rewards)}")
        print(f"Total steps: {global_step}")
        
        if use_wandb:
            wandb.log({
                "test/final_mean_return": final_return,
                "test/total_episodes": len(episode_rewards),
                "test/total_steps": global_step,
                "test/success": True
            })
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        success = False
        if use_wandb:
            wandb.log({
                "test/success": False,
                "test/error": str(e)
            })
    
    finally:
        # Cleanup
        try:
            envs.close()
        except:
            pass
        
        if use_wandb:
            wandb.finish()
    
    return success


def test_isaac_pendulum():
    """Test Isaac Lab Pendulum if available, otherwise use regular Pendulum."""
    print(f"\n{'='*60}")
    print("Testing Isaac Lab Integration")
    print(f"{'='*60}")
    
    # Check if Isaac Lab is available
    try:
        # Try to import Isaac Lab
        import isaaclab
        from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
        ISAAC_LAB_AVAILABLE = True
        print("✅ Isaac Lab is available!")
    except ImportError:
        ISAAC_LAB_AVAILABLE = False
        print("⚠️  Isaac Lab not available, using standard Gym environment")
    
    if ISAAC_LAB_AVAILABLE:
        # TODO: Create Isaac Lab wrapper and test
        print("Isaac Lab integration not yet implemented")
        return False
    else:
        # Fall back to regular Pendulum
        return test_environment("Pendulum-v1", num_iterations=5, use_wandb=False)


def main():
    """Run tests on multiple environments."""
    print("PPO Multi-Environment Test Suite")
    print("================================")
    
    # Load environment variables
    load_dotenv()
    
    # Test environments
    test_configs = [
        # ("CartPole-v1", 10, True),    # Discrete control
        ("Pendulum-v1", 10, True),     # Continuous control
        # ("Ant-v4", 5, True),          # Multi-dimensional continuous
    ]
    
    results = []
    
    for env_name, num_iterations, use_wandb in test_configs:
        try:
            success = test_environment(env_name, num_iterations, use_wandb=use_wandb)
            results.append((env_name, success))
        except Exception as e:
            print(f"\n❌ {env_name} test failed with exception: {e}")
            results.append((env_name, False))
    
    # Test Isaac Lab integration
    isaac_success = test_isaac_pendulum()
    results.append(("Isaac Lab Integration", isaac_success))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    all_passed = True
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    main()