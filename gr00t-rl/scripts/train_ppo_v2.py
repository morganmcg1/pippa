#!/usr/bin/env python3
"""
Improved PPO training script with all 37 implementation details.
Based on CleanRL and ICLR blog post best practices.
"""

import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import wandb
import gymnasium as gym

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.ppo_config_v2 import PPOConfigV2
from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from environments.vec_isaac_env import make_vec_env, SubprocVecEnv, DummyVecEnv
from utils.buffers import PPORolloutBuffer
from utils.normalization import VecNormalize
from utils.logging import get_system_metrics


def set_random_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable CUBLAS deterministic mode for better reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)


def linear_schedule(initial_value: float) -> callable:
    """
    Linear learning rate schedule.
    One of the 37 implementation details: anneal learning rate.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class PPOTrainerV2:
    """
    Improved PPO trainer with all 37 implementation details.
    """
    
    def __init__(self, config: PPOConfigV2):
        self.config = config
        
        # Set seeds
        set_random_seed(config.seed, config.torch_deterministic)
        
        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.cuda else "cpu"
        )
        
        # Create environments
        self.envs = self._create_envs()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup mixed precision training
        self.use_amp = config.cuda and config.mixed_precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training enabled (AMP)")
        
        # Create buffer
        self.buffer = PPORolloutBuffer(
            buffer_size=config.n_steps,
            observation_space=self.envs.observation_space,
            action_space=self.envs.action_space,
            device=self.device,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            n_envs=config.num_envs
        )
        
        # Logging
        self.start_time = time.time()
        self.global_step = 0
        self.num_updates = 0
        
        # Setup logging
        if config.track:
            self._setup_logging()
            
    def _create_envs(self):
        """Create vectorized environments with normalization."""
        # Create vectorized environments
        vec_env_cls = SubprocVecEnv if self.config.num_envs > 1 else DummyVecEnv
        
        # Create envs with proper seeding
        def make_env(rank: int):
            def _init():
                env = gym.make(self.config.env_name)
                # Seed each environment with a different seed
                env.reset(seed=self.config.seed + rank)
                return env
            return _init
        
        envs = make_vec_env(
            env_id=self.config.env_name,
            n_envs=self.config.num_envs,
            seed=self.config.seed,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs={"start_method": "spawn"},  # For CUDA compatibility
            env_kwargs={"seed": self.config.seed}  # Pass seed to env constructor if supported
        )
        
        # Apply normalization wrapper
        if self.config.norm_obs or self.config.norm_reward:
            envs = VecNormalize(
                envs,
                training=True,
                norm_obs=self.config.norm_obs,
                norm_reward=self.config.norm_reward,
                clip_obs=self.config.clip_obs,
                clip_reward=self.config.clip_reward,
                gamma=self.config.gamma,
                epsilon=1e-8,
                device=self.device
            )
            
        return envs
    
    def _create_agent(self):
        """Create the actor-critic agent."""
        # Determine observation and action spaces
        obs_space = self.envs.observation_space
        act_space = self.envs.action_space
        
        # Get action dimension
        if hasattr(act_space, 'shape'):
            action_dim = act_space.shape[0]
        else:
            action_dim = 1
            
        # Create agent
        agent = PPOGr00tActorCriticV2(
            observation_space=obs_space,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
            use_layer_norm=self.config.use_layer_norm,
            log_std_init=self.config.log_std_init,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
            use_multimodal_encoder=self.config.use_multimodal_encoder,
            vision_dim=self.config.vision_dim,
            proprio_dim=self.config.proprio_dim,
            language_dim=self.config.language_dim,
            encoder_output_dim=self.config.encoder_output_dim,
            device=self.device
        ).to(self.device)
        
        return agent
    
    def _create_optimizer(self):
        """
        Create optimizer with specific Adam epsilon.
        Implementation detail: Use epsilon=1e-5 instead of default 1e-8.
        """
        return optim.Adam(
            self.agent.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon  # 1e-5 instead of 1e-8
        )
    
    def _setup_logging(self):
        """Setup WandB and tensorboard logging."""
        # Create run name
        run_name = f"{self.config.exp_name}_{self.config.seed}_{int(time.time())}"
        
        # Initialize WandB
        wandb.init(
            project=self.config.wandb_project_name,
            entity=self.config.wandb_entity,
            name=run_name,
            config=vars(self.config),
            tags=self.config.wandb_tags + ["gr00t-ppo-testing"] if "gr00t-ppo-testing" not in self.config.wandb_tags else self.config.wandb_tags,
            save_code=True,
        )
        
        # Create tensorboard writer
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(self.config).items()])
            ),
        )
        
    def collect_rollouts(self):
        """
        Collect rollouts for n_steps.
        Implementation details:
        - No gradient computation during rollout
        - Proper action clipping
        - Episode statistics tracking
        """
        obs = self.last_obs
        dones = self.last_dones
        
        # Episode statistics
        episode_rewards = []
        episode_lengths = []
        episode_successes = []  # Track success rate for robotics tasks
        
        for step in range(self.config.n_steps):
            self.global_step += self.config.num_envs
            
            # Convert to torch tensors
            obs_tensor = obs
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
            elif isinstance(obs, dict):
                obs_tensor = {
                    k: torch.from_numpy(v).float().to(self.device) 
                    for k, v in obs.items()
                }
                
            # Get action and value
            with torch.no_grad():
                action, log_prob, _, value = self.agent.get_action_and_value(
                    obs_tensor, deterministic=False
                )
                
            # Clip actions (implementation detail)
            action = torch.clamp(action, -self.config.action_clip, self.config.action_clip)
            
            # Step environment
            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
            
            # Store transition
            self.buffer.add(
                obs=obs_tensor,
                action=action,
                reward=torch.from_numpy(reward).to(self.device),
                done=torch.from_numpy(done).to(self.device),
                value=value,
                log_prob=log_prob
            )
            
            # Track episode statistics
            for idx, i in enumerate(info):
                if "episode" in i:
                    episode_rewards.append(i["episode"]["r"])
                    episode_lengths.append(i["episode"]["l"])
                    
                    # Track success if available (common in robotics tasks)
                    if "is_success" in i:
                        episode_successes.append(float(i["is_success"]))
                    elif "success" in i:
                        episode_successes.append(float(i["success"]))
                    
                    # Log immediately for real-time monitoring
                    if self.config.track:
                        log_dict = {
                            "charts/episodic_return": i["episode"]["r"],
                            "charts/episodic_length": i["episode"]["l"],
                            "global_step": self.global_step,
                        }
                        
                        # Log success if available
                        if "is_success" in i or "success" in i:
                            success = i.get("is_success", i.get("success", 0))
                            log_dict["charts/episodic_success"] = float(success)
                        
                        wandb.log(log_dict)
                        
            # Update observations
            obs = next_obs
            dones = done
            
        # Store last values for bootstrapping
        self.last_obs = obs
        self.last_dones = dones
        
        # Get last values for GAE computation
        with torch.no_grad():
            obs_tensor = obs
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
            elif isinstance(obs, dict):
                obs_tensor = {
                    k: torch.from_numpy(v).float().to(self.device) 
                    for k, v in obs.items()
                }
            last_values = self.agent.get_value(obs_tensor)
            
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(
            last_values=last_values,
            dones=torch.from_numpy(dones).to(self.device)
        )
        
        return episode_rewards, episode_lengths, episode_successes
    
    def update(self):
        """
        Perform PPO update with all implementation details.
        Key details:
        - Normalize advantages at batch level (not mini-batch)
        - Clip value function loss
        - Use separate value function coefficient
        - Gradient clipping
        - Early stopping based on KL divergence
        """
        # Get all data from buffer (advantages already normalized at batch level)
        batch_data = self.buffer.get()
        
        # Learning rate annealing
        if self.config.anneal_lr:
            frac = 1.0 - (self.num_updates / self.config.num_updates)
            lr = linear_schedule(self.config.learning_rate)(frac)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = self.config.learning_rate
            
        # Initialize metrics
        pg_losses = []
        value_losses = []
        entropy_losses = []
        old_approx_kls = []
        approx_kls = []
        clipfracs = []
        
        # Optimization epochs
        for epoch in range(self.config.n_epochs):
            # Create random indices for mini-batches
            # Important: shuffle for each epoch to avoid overfitting
            b_inds = np.arange(self.config.batch_size)
            np.random.shuffle(b_inds)
            
            # Mini-batch updates - each sample appears in exactly one mini-batch per epoch
            for start in range(0, self.config.batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_inds = b_inds[start:end]
                
                # Get mini-batch data
                mb_obs = batch_data['observations']
                if isinstance(mb_obs, dict):
                    mb_obs = {k: v[mb_inds] for k, v in mb_obs.items()}
                else:
                    mb_obs = mb_obs[mb_inds]
                    
                mb_actions = batch_data['actions'][mb_inds]
                mb_log_probs = batch_data['log_probs'][mb_inds]
                mb_advantages = batch_data['advantages'][mb_inds]
                mb_returns = batch_data['returns'][mb_inds]
                mb_values = batch_data['values'][mb_inds]
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        new_log_probs, new_values, entropy = self.agent.evaluate_actions(
                            mb_obs, mb_actions
                        )
                else:
                    new_log_probs, new_values, entropy = self.agent.evaluate_actions(
                        mb_obs, mb_actions
                    )
                
                # Ratio for PPO
                log_ratio = new_log_probs - mb_log_probs
                ratio = torch.exp(log_ratio)
                
                # KL divergence for early stopping
                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    old_approx_kls.append(old_approx_kl.item())
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    approx_kls.append(approx_kl.item())
                    clipfrac = ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
                    clipfracs.append(clipfrac.item())
                
                # Policy loss (PPO clip)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                new_values = new_values.view(-1)
                if self.config.clip_range_vf is not None:
                    # Value clipping (implementation detail)
                    v_loss_unclipped = (new_values - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        new_values - mb_values,
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = (
                    pg_loss 
                    + self.config.vf_coef * v_loss 
                    - self.config.ent_coef * entropy_loss
                )
                
                # Optimization step with mixed precision
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    # Scale loss and backward
                    self.scaler.scale(loss).backward()
                    
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), 
                        self.config.max_grad_norm
                    )
                    
                    # Step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    
                    # Gradient clipping (implementation detail)
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), 
                        self.config.max_grad_norm
                    )
                    
                    self.optimizer.step()
                
                # Store losses
                pg_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                
            # Early stopping based on KL divergence
            if self.config.target_kl is not None:
                # Use mean KL from this epoch
                mean_kl = np.mean(approx_kls[-len(mb_inds):]) if approx_kls else 0
                if mean_kl > self.config.target_kl:
                    print(f"Early stopping at epoch {epoch} due to KL divergence: {mean_kl:.4f} > {self.config.target_kl}")
                    break
                    
        # Compute explained variance
        y_pred = batch_data['values'].cpu().numpy()
        y_true = batch_data['returns'].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Return metrics
        return {
            "train/policy_loss": np.mean(pg_losses),
            "train/value_loss": np.mean(value_losses),
            "train/entropy_loss": np.mean(entropy_losses),
            "train/approx_kl": np.mean(approx_kls),
            "train/clipfrac": np.mean(clipfracs),
            "train/explained_variance": explained_var,
            "train/learning_rate": lr,
        }
    
    def train(self):
        """Main training loop."""
        # Initialize environments
        self.last_obs = self.envs.reset()
        self.last_dones = np.zeros((self.config.num_envs,), dtype=bool)
        
        # Training loop
        for update in range(1, self.config.num_updates + 1):
            self.num_updates = update
            
            # Collect rollouts
            episode_rewards, episode_lengths, episode_successes = self.collect_rollouts()
            
            # Perform PPO update
            update_metrics = self.update()
            
            # Clear buffer
            self.buffer.reset()
            
            # Logging
            if update % self.config.log_interval == 0:
                # Add system metrics
                system_metrics = get_system_metrics()
                update_metrics.update(system_metrics)
                
                # Add timing metrics
                time_elapsed = time.time() - self.start_time
                update_metrics["charts/SPS"] = int(self.global_step / time_elapsed)
                update_metrics["charts/global_step"] = self.global_step
                
                # Log to WandB
                if self.config.track:
                    wandb.log(update_metrics)
                    
                # Console output
                print(f"Update {update}/{self.config.num_updates}")
                print(f"  Global step: {self.global_step}")
                print(f"  SPS: {update_metrics['charts/SPS']}")
                print(f"  Policy loss: {update_metrics['train/policy_loss']:.4f}")
                print(f"  Value loss: {update_metrics['train/value_loss']:.4f}")
                print(f"  Approx KL: {update_metrics['train/approx_kl']:.4f}")
                
                if episode_rewards:
                    log_dict = {
                        "charts/mean_episodic_return": np.mean(episode_rewards),
                        "charts/mean_episodic_length": np.mean(episode_lengths),
                    }
                    
                    # Add success rate if available
                    if episode_successes:
                        success_rate = np.mean(episode_successes)
                        log_dict["charts/success_rate"] = success_rate
                        print(f"  Episode return: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
                        print(f"  Success rate: {success_rate:.2%}")
                    else:
                        print(f"  Episode return: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
                    
                    if self.config.track:
                        wandb.log(log_dict)
                    
            # Save checkpoint
            if update % self.config.save_interval == 0:
                checkpoint_dir = Path(f"checkpoints/{self.config.exp_name}")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_path = checkpoint_dir / f"checkpoint_{update}.pt"
                self.agent.save(checkpoint_path)
                
                # Save normalizer state if using normalization
                if hasattr(self.envs, 'obs_rms'):
                    normalizer_path = checkpoint_dir / f"normalizer_{update}.pt"
                    torch.save({
                        'obs_rms': self.envs.obs_rms,
                        'return_rms': getattr(self.envs, 'return_rms', None)
                    }, normalizer_path)
                    
                print(f"Saved checkpoint to {checkpoint_path}")
                
        # Cleanup
        self.envs.close()
        if self.config.track:
            wandb.finish()


def main():
    """Main entry point."""
    # Create config
    config = PPOConfigV2()
    
    # Override from command line if needed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=config.env_name)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--total-timesteps", type=int, default=config.total_timesteps)
    parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
    parser.add_argument("--num-envs", type=int, default=config.num_envs)
    parser.add_argument("--num-steps", type=int, default=config.n_steps)
    parser.add_argument("--anneal-lr", type=bool, default=config.anneal_lr)
    parser.add_argument("--track", type=bool, default=config.track)
    parser.add_argument("--cuda", type=bool, default=config.cuda)
    parser.add_argument("--capture-video", type=bool, default=config.capture_video)
    
    args = parser.parse_args()
    
    # Update config
    config.env_name = args.env
    config.seed = args.seed
    config.total_timesteps = args.total_timesteps
    config.learning_rate = args.learning_rate
    config.num_envs = args.num_envs
    config.n_steps = args.num_steps
    config.anneal_lr = args.anneal_lr
    config.track = args.track
    config.cuda = args.cuda
    config.capture_video = args.capture_video
    
    # Re-calculate derived values
    config.__post_init__()
    
    # Create trainer
    trainer = PPOTrainerV2(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()