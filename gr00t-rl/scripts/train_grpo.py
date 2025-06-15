#!/usr/bin/env python3
"""
GRPO training script for GR00T.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.grpo_gr00t import GRPOGr00t, GRPORollout
from environments.isaac_env import IsaacGR00TEnv
from configs.grpo_config import GRPOConfig
from utils.rewards import PickAndPlaceReward, PushToGoalReward
from utils.logging import compute_grpo_health_metrics, compute_gradient_norm, get_system_metrics

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available. Install with: pip install wandb")

from dotenv import load_dotenv
load_dotenv()


def train_grpo(config: GRPOConfig):
    """Main GRPO training function."""
    
    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Initialize WandB
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.exp_name,
            config=config.__dict__,
            tags=config.wandb_tags
        )
    
    # Create environment
    env = IsaacGR00TEnv(
        env_name=config.env_name,
        device=config.device
    )
    
    # Create model
    model = GRPOGr00t(
        model_path=config.model_path,
        embodiment_tag=config.embodiment_tag,
        num_rollouts_per_update=config.num_rollouts_per_update,
        freeze_backbone=config.freeze_backbone,
        temperature=config.temperature,
        beta=config.beta,
        device=config.device
    )
    
    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.learning_rate)
    
    # Training variables
    total_episodes = 0
    episode_buffer = []
    
    # Training loop
    print(f"Starting GRPO training for {config.total_episodes} episodes...")
    print(f"Collecting {config.num_rollouts_per_update} rollouts per update")
    
    while total_episodes < config.total_episodes:
        # Get initial observation
        initial_obs, _ = env.reset()
        
        # Generate rollouts from same initial state
        rollouts = model.generate_rollouts(
            env,
            initial_obs,
            max_steps=config.max_episode_steps
        )
        
        # Update episode count
        total_episodes += len(rollouts)
        episode_buffer.extend(rollouts)
        
        # Log individual rollout stats
        for i, rollout in enumerate(rollouts):
            if config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'episode/reward': rollout.episode_return,
                    'episode/length': len(rollout.rewards),
                    'episode/rollout_idx': i,
                }, step=total_episodes - len(rollouts) + i)
        
        # Update policy when we have enough episodes
        if len(episode_buffer) >= config.update_interval:
            # Group rollouts by initial state (for GRPO)
            # In this implementation, we assume all rollouts in buffer share same initial
            rollout_groups = [episode_buffer[:config.num_rollouts_per_update]]
            episode_buffer = episode_buffer[config.num_rollouts_per_update:]
            
            # Update on each group
            for rollout_group in rollout_groups:
                # Update policy
                info = model.update_policy(
                    rollout_group,
                    optimizer,
                    num_epochs=config.n_epochs,
                    clip_epsilon=config.clip_range
                )
                
                # Log training stats with enhanced metrics
                if config.use_wandb and WANDB_AVAILABLE:
                    # Basic metrics
                    logs = {
                        'train/loss': info['loss'],
                        'train/mean_return': info['mean_return'],
                        'train/std_return': info['std_return'],
                        'train/max_return': info['max_return'],
                        'train/min_return': info['min_return'],
                        'train/episodes': total_episodes,
                    }
                    
                    # Add gradient norm
                    grad_norm = compute_gradient_norm(p for p in model.parameters() if p.grad is not None)
                    logs['train/grad_norm'] = grad_norm
                    
                    # Add GRPO health metrics
                    returns = torch.tensor([r.episode_return for r in rollout_group])
                    if returns.std() > 0:
                        advantages = (returns - returns.mean()) / returns.std()
                    else:
                        advantages = returns - returns.mean()
                    
                    health_metrics = compute_grpo_health_metrics(rollout_group, advantages)
                    logs.update(health_metrics)
                    
                    # Add learning rate
                    logs['train/learning_rate'] = optimizer.param_groups[0]['lr']
                    
                    # Add KL divergence if available
                    if 'kl_divergence' in info:
                        logs['train/kl_divergence'] = info['kl_divergence']
                    
                    # Add system metrics periodically
                    if total_episodes % 10 == 0:
                        system_metrics = get_system_metrics()
                        logs.update(system_metrics)
                    
                    wandb.log(logs, step=total_episodes)
                
                print(f"Episodes {total_episodes}: "
                      f"Mean Return = {info['mean_return']:.2f} ± {info['std_return']:.2f}, "
                      f"Max = {info['max_return']:.2f}")
        
        # Save checkpoint
        if total_episodes % config.save_interval == 0:
            checkpoint_path = f"checkpoints/{config.exp_name}_ep_{total_episodes}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            model.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Evaluation
        if total_episodes % config.eval_interval == 0:
            eval_rewards = []
            
            print(f"\nEvaluating for {config.eval_episodes} episodes...")
            for eval_ep in range(config.eval_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                steps = 0
                
                while not done and steps < config.max_episode_steps:
                    with torch.no_grad():
                        # Use deterministic action for evaluation
                        action_output = model.gr00t_policy.get_action(obs)
                        action = action_output["action"]
                        
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    steps += 1
                
                eval_rewards.append(episode_reward)
                
            mean_eval_reward = np.mean(eval_rewards)
            std_eval_reward = np.std(eval_rewards)
            
            if config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'eval/mean_reward': mean_eval_reward,
                    'eval/std_reward': std_eval_reward,
                    'eval/episodes': total_episodes,
                }, step=total_episodes)
            
            print(f"Evaluation: Mean Reward = {mean_eval_reward:.2f} ± {std_eval_reward:.2f}")
    
    print("Training completed!")
    
    # Final save
    final_checkpoint_path = f"checkpoints/{config.exp_name}_final.pt"
    model.save(final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")
    
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    config = GRPOConfig()
    train_grpo(config)