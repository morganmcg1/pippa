#!/usr/bin/env python3
"""
PPO training script for GR00T.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.ppo_gr00t import PPOGr00tActorCritic
from environments.isaac_env import IsaacGR00TEnv
from configs.ppo_config import PPOConfig
from utils.rewards import PickAndPlaceReward, PushToGoalReward
from utils.logging import compute_ppo_diagnostics, compute_gradient_norm, get_system_metrics

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available. Install with: pip install wandb")

from dotenv import load_dotenv
load_dotenv()


class PPOBuffer:
    """Simple buffer for PPO training."""
    
    def __init__(self, size: int, obs_shape: Dict[str, tuple], action_dim: int, device: str):
        self.size = size
        self.device = device
        
        # Storage
        self.observations = {}
        for key, shape in obs_shape.items():
            self.observations[key] = torch.zeros((size, *shape), device=device)
            
        self.actions = torch.zeros((size, action_dim), device=device)
        self.rewards = torch.zeros(size, device=device)
        self.values = torch.zeros(size, device=device)
        self.log_probs = torch.zeros(size, device=device)
        self.dones = torch.zeros(size, device=device)
        self.advantages = torch.zeros(size, device=device)
        self.returns = torch.zeros(size, device=device)
        
        self.ptr = 0
        self.path_start_idx = 0
        
    def store(
        self, 
        obs: Dict[str, torch.Tensor], 
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool
    ):
        """Store a transition."""
        for key, val in obs.items():
            if key in self.observations:
                self.observations[key][self.ptr] = val
                
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
        
    def finish_path(self, last_value: float = 0.0, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute advantages and returns for a completed path."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        
        # Append last value for GAE computation
        values = torch.cat([values, torch.tensor([last_value], device=self.device)])
        
        # Compute GAE
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + values[:-1]
        
        self.path_start_idx = self.ptr
        
    def get(self):
        """Get all data and reset buffer."""
        assert self.ptr == self.size
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        return {
            'observations': self.observations,
            'actions': self.actions,
            'values': self.values,
            'log_probs': self.log_probs,
            'advantages': self.advantages,
            'returns': self.returns
        }


def train_ppo(config: PPOConfig):
    """Main PPO training function."""
    
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
    model = PPOGr00tActorCritic(
        model_path=config.model_path,
        embodiment_tag=config.embodiment_tag,
        freeze_backbone=config.freeze_backbone,
        freeze_action_head=config.freeze_action_head,
        device=config.device
    )
    
    # Create optimizers
    actor_params = []
    critic_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'critic' in name:
                critic_params.append(param)
            else:
                actor_params.append(param)
                
    actor_optimizer = optim.Adam(actor_params, lr=config.learning_rate)
    critic_optimizer = optim.Adam(critic_params, lr=config.critic_learning_rate)
    
    # Create buffer
    obs_shape = {
        'ego_view': (224, 224, 3),
        'joint_positions': (7,),
        'joint_velocities': (7,),
    }
    
    buffer = PPOBuffer(
        size=config.buffer_size,
        obs_shape=obs_shape,
        action_dim=env.action_dim,
        device=config.device
    )
    
    # Training variables
    total_steps = 0
    episode_count = 0
    
    # Training loop
    print(f"Starting PPO training for {config.total_timesteps} timesteps...")
    
    for update in range(config.n_updates):
        # Collect rollouts
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(config.n_steps):
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = model.get_action(obs, deterministic=False)
                
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            buffer.store(obs, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Handle episode end
            if done:
                # Compute returns and advantages
                buffer.finish_path(last_value=0.0, gamma=config.gamma, gae_lambda=config.gae_lambda)
                
                # Log episode stats
                if config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'episode/reward': episode_reward,
                        'episode/length': episode_length,
                        'episode/count': episode_count,
                    }, step=total_steps)
                    
                print(f"Episode {episode_count}: Reward = {episode_reward:.2f}, Length = {episode_length}")
                
                # Reset
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_count += 1
            else:
                obs = next_obs
                
        # Compute final value for incomplete episode
        if not done:
            with torch.no_grad():
                last_value = model.get_value(obs)
            buffer.finish_path(last_value=last_value.item(), gamma=config.gamma, gae_lambda=config.gae_lambda)
            
        # Get training data
        data = buffer.get()
        
        # PPO update
        for epoch in range(config.n_epochs):
            # Sample mini-batches
            indices = np.random.permutation(config.buffer_size)
            
            for start in range(0, config.buffer_size, config.batch_size):
                end = start + config.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = {k: v[batch_indices] for k, v in data['observations'].items()}
                batch_actions = data['actions'][batch_indices]
                batch_log_probs = data['log_probs'][batch_indices]
                batch_advantages = data['advantages'][batch_indices]
                batch_returns = data['returns'][batch_indices]
                batch_values = data['values'][batch_indices]
                
                # Evaluate actions
                new_log_probs, new_values, entropy = model.evaluate_actions(batch_obs, batch_actions)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * batch_advantages
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(new_values, batch_returns)
                
                # Total loss
                total_loss = actor_loss + config.ent_coef * entropy_loss + config.vf_coef * value_loss
                
                # Update actor
                actor_optimizer.zero_grad()
                (actor_loss + config.ent_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(actor_params, config.max_grad_norm)
                actor_optimizer.step()
                
                # Update critic
                critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, config.max_grad_norm)
                critic_optimizer.step()
                
        # Log training stats with enhanced metrics
        if update % config.log_interval == 0:
            if config.use_wandb and WANDB_AVAILABLE:
                # Basic metrics
                logs = {
                    'train/actor_loss': actor_loss.item(),
                    'train/value_loss': value_loss.item(),
                    'train/entropy': entropy.mean().item(),
                    'train/total_steps': total_steps,
                }
                
                # Add gradient norms
                actor_grad_norm = compute_gradient_norm(actor_params)
                critic_grad_norm = compute_gradient_norm(critic_params)
                logs['train/actor_grad_norm'] = actor_grad_norm
                logs['train/critic_grad_norm'] = critic_grad_norm
                
                # Add learning rates
                logs['train/actor_learning_rate'] = actor_optimizer.param_groups[0]['lr']
                logs['train/critic_learning_rate'] = critic_optimizer.param_groups[0]['lr']
                
                # Add PPO diagnostics (using last batch data)
                ppo_data = {
                    'ratio': ratio,
                    'advantages': batch_advantages,
                    'clip_range': config.clip_range,
                    'values': batch_values,
                    'returns': batch_returns,
                    'old_log_probs': batch_log_probs,
                    'new_log_probs': new_log_probs
                }
                ppo_diagnostics = compute_ppo_diagnostics(**ppo_data)
                logs.update(ppo_diagnostics)
                
                # Add system metrics periodically
                if update % 10 == 0:
                    system_metrics = get_system_metrics()
                    logs.update(system_metrics)
                
                wandb.log(logs, step=total_steps)
                
        # Save checkpoint
        if update % config.save_interval == 0:
            checkpoint_path = f"checkpoints/{config.exp_name}_step_{total_steps}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            model.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
        # Evaluation
        if update % config.eval_interval == 0:
            eval_rewards = []
            
            for _ in range(config.eval_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    with torch.no_grad():
                        action, _, _ = model.get_action(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    
                eval_rewards.append(episode_reward)
                
            mean_eval_reward = np.mean(eval_rewards)
            
            if config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'eval/mean_reward': mean_eval_reward,
                    'eval/std_reward': np.std(eval_rewards),
                }, step=total_steps)
                
            print(f"Evaluation: Mean Reward = {mean_eval_reward:.2f}")
            
    print("Training completed!")
    
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    config = PPOConfig()
    train_ppo(config)