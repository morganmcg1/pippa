#!/usr/bin/env python3
"""
Simple training script using our PPO implementation with Isaac Lab environments.
This script uses the Isaac Gym wrapper to make Isaac Lab envs compatible with our PPO.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

import wandb
from dotenv import load_dotenv

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from configs.ppo_config_v2 import PPOConfigV2
from environments.isaac_gym_wrapper import create_isaac_env
from utils.buffers import PPORolloutBuffer
from utils.normalization import VecNormalize
from utils.logging import get_system_metrics


class SimpleIsaacTrainer:
    """Simple trainer for Isaac Lab environments using our PPO."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # Create environment
        print(f"Creating environment: {args.env}")
        self.env = create_isaac_env(
            args.env,
            num_envs=args.num_envs,
            device=str(self.device)
        )
        
        # Add normalization for continuous control
        if isinstance(self.env.action_space, gym.spaces.Box):
            self.env = VecNormalize(
                self.env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0
            )
        
        # Get dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0] if hasattr(self.env.action_space, 'shape') else 1
        
        print(f"Environment created:")
        print(f"  Using Isaac Lab: {self.env.is_isaac}")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.act_dim}")
        print(f"  Num envs: {self.env.num_envs}")
        
        # Create model
        self.model = PPOGr00tActorCriticV2(
            observation_space=self.env.observation_space,
            action_dim=self.act_dim,
            hidden_dims=[256, 256],
            use_multimodal_encoder=False,
            device=self.device
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model created: {total_params:,} parameters")
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            eps=1e-5
        )
        
        # Create buffer
        self.buffer = PPORolloutBuffer(
            buffer_size=args.num_steps,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
            gamma=0.99,
            gae_lambda=0.95,
            n_envs=self.env.num_envs
        )
        
        # Initialize WandB if requested
        if args.track:
            self._init_wandb()
            
        # Training state
        self.global_step = 0
        self.start_time = time.time()
        
    def _init_wandb(self):
        """Initialize WandB logging."""
        load_dotenv()
        
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "pippa"),
            entity=os.getenv("WANDB_ENTITY", "wild-ai"),
            name=f"isaac_ppo_{self.args.env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(self.args),
            tags=["isaac-lab", "ppo", self.args.env.lower(), "gr00t-rl"]
        )
        
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.args.total_timesteps} timesteps")
        
        # Reset environment
        obs = self.env.reset()
        episode_rewards = []
        episode_lengths = []
        
        num_updates = self.args.total_timesteps // (self.args.num_steps * self.env.num_envs)
        
        for update in range(num_updates):
            # Collect rollouts
            for step in range(self.args.num_steps):
                self.global_step += self.env.num_envs
                
                # Convert to tensor
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
                
                # Get action
                with torch.no_grad():
                    actions, log_probs, _, values = self.model.get_action_and_value(obs_tensor)
                
                # Step environment
                actions_np = actions.cpu().numpy()
                next_obs, rewards, dones, infos = self.env.step(actions_np)
                
                # Store in buffer
                self.buffer.add(
                    obs=obs_tensor,
                    action=actions,
                    reward=torch.from_numpy(rewards).to(self.device),
                    done=torch.from_numpy(dones).to(self.device),
                    value=values,
                    log_prob=log_probs
                )
                
                # Track episodes
                if isinstance(infos, dict):
                    infos = [infos]  # Single env
                    
                for info in infos:
                    if "episode" in info:
                        episode_rewards.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
                        
                        if self.args.track:
                            wandb.log({
                                "charts/episodic_return": info["episode"]["r"],
                                "charts/episodic_length": info["episode"]["l"],
                                "global_step": self.global_step
                            })
                
                obs = next_obs
                
            # Compute returns
            with torch.no_grad():
                last_values = self.model.get_value(torch.from_numpy(next_obs).float().to(self.device))
            
            self.buffer.compute_returns_and_advantages(
                last_values=last_values,
                dones=torch.from_numpy(dones).to(self.device)
            )
            
            # PPO update
            self._ppo_update()
            
            # Clear buffer
            self.buffer.reset()
            
            # Logging
            if update % 10 == 0:
                fps = int(self.global_step / (time.time() - self.start_time))
                
                print(f"\nUpdate {update}/{num_updates}")
                print(f"  Timestep: {self.global_step}/{self.args.total_timesteps}")
                print(f"  FPS: {fps}")
                
                if episode_rewards:
                    mean_return = np.mean(episode_rewards[-100:])
                    mean_length = np.mean(episode_lengths[-100:])
                    print(f"  Mean return: {mean_return:.2f}")
                    print(f"  Mean length: {mean_length:.0f}")
                    
                    if self.args.track:
                        wandb.log({
                            "charts/mean_episodic_return": mean_return,
                            "charts/mean_episodic_length": mean_length,
                            "charts/fps": fps,
                            "global_step": self.global_step
                        })
                        
            # Save checkpoint
            if update % 100 == 0:
                self._save_checkpoint(update)
                
        print("\nTraining completed!")
        self.env.close()
        
        if self.args.track:
            wandb.finish()
            
    def _ppo_update(self):
        """Perform PPO update."""
        batch_data = self.buffer.get()
        
        # Training metrics
        pg_losses = []
        value_losses = []
        entropy_losses = []
        
        # Multiple epochs
        for epoch in range(self.args.num_epochs):
            # Create mini-batches
            batch_size = self.args.num_steps * self.env.num_envs
            minibatch_size = batch_size // self.args.num_minibatches
            
            indices = np.random.permutation(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]
                
                # Get mini-batch data
                mb_obs = batch_data['observations'][mb_inds]
                mb_actions = batch_data['actions'][mb_inds]
                mb_log_probs = batch_data['log_probs'][mb_inds]
                mb_advantages = batch_data['advantages'][mb_inds]
                mb_returns = batch_data['returns'][mb_inds]
                mb_values = batch_data['values'][mb_inds]
                
                # Get current policy outputs
                new_log_probs, new_values, entropy = self.model.evaluate_actions(mb_obs, mb_actions)
                
                # Calculate ratio
                logratio = new_log_probs - mb_log_probs
                ratio = logratio.exp()
                
                # Policy loss (clipped PPO)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((new_values.flatten() - mb_returns) ** 2).mean()
                
                # Total loss
                loss = pg_loss + self.args.vf_coef * v_loss - self.args.ent_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                pg_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy.item())
                
        # Log training metrics
        if self.args.track:
            wandb.log({
                "train/policy_loss": np.mean(pg_losses),
                "train/value_loss": np.mean(value_losses),
                "train/entropy": np.mean(entropy_losses),
                "train/learning_rate": self.args.learning_rate,
                "global_step": self.global_step
            })
            
    def _save_checkpoint(self, update):
        """Save model checkpoint."""
        checkpoint_dir = Path(f"checkpoints/{self.args.env}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "update": update,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "args": self.args
        }
        
        path = checkpoint_dir / f"checkpoint_{update}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train PPO on Isaac Lab environments")
    
    # Environment
    parser.add_argument("--env", type=str, default="Isaac-Cartpole-v0",
                        help="Environment name (Isaac-* or standard Gym)")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="Number of parallel environments")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="Total timesteps to train")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="Number of steps per rollout")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of PPO epochs")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="Number of mini-batches")
    
    # Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum gradient norm")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--track", action="store_true",
                        help="Enable WandB tracking")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Add missing gym import
    global gym
    import gymnasium as gym
    
    # Create trainer
    trainer = SimpleIsaacTrainer(args)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()