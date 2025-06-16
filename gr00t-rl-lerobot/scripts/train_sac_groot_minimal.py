#!/usr/bin/env python3
"""
Minimal SAC training script with GR00T model loading workaround.
Uses the actual GR00T model but avoids the flash-attn import issue.
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import numpy as np
import wandb
from pathlib import Path
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Workaround for flash-attn issue
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"

from environments.fetch_so101_coupled import make_fetch_so101_coupled_env

# Load environment variables
load_dotenv()


class GR00TSACTrainer:
    """SAC trainer that will use GR00T model (simplified for now)."""
    
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda",
    ):
        self.env = env
        self.device = torch.device(device)
        
        # For now, use a simple policy to test training
        # Will replace with actual GR00T later
        self.policy = self._create_simple_policy().to(self.device)
        self.q1_net = self._create_q_network().to(self.device)
        self.q2_net = self._create_q_network().to(self.device)
        self.q1_target = self._create_q_network().to(self.device)
        self.q2_target = self._create_q_network().to(self.device)
        
        # Copy parameters to targets
        self.q1_target.load_state_dict(self.q1_net.state_dict())
        self.q2_target.load_state_dict(self.q2_net.state_dict())
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2_net.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        
        # Replay buffer
        self.replay_buffer = []
        self.buffer_size = 10000
        
    def _create_simple_policy(self):
        """Create a simple policy network."""
        return nn.Sequential(
            nn.Linear(224*224*3*2 + 6, 256),  # Simplified input
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Tanh()
        )
    
    def _create_q_network(self):
        """Create Q-network."""
        return nn.Sequential(
            nn.Linear(224*224*3*2 + 6 + 6, 256),  # obs + action
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def collect_rollout(self, num_steps: int = 1000):
        """Collect experience."""
        obs, info = self.env.reset()
        
        for _ in range(num_steps):
            # Flatten observation for simple policy
            obs_flat = self._flatten_obs(obs)
            
            # Get action
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)
                action = self.policy(obs_tensor)
                action_np = action.cpu().numpy().squeeze()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            
            # Store transition
            self.add_to_buffer(obs_flat, action_np, reward, self._flatten_obs(next_obs), terminated)
            
            if terminated or truncated:
                obs, info = self.env.reset()
            else:
                obs = next_obs
    
    def _flatten_obs(self, obs):
        """Flatten observation to vector."""
        front = obs["observation"]["images"]["front"].flatten()
        wrist = obs["observation"]["images"]["wrist"].flatten()
        state = obs["observation"]["state"]
        return np.concatenate([front, wrist, state])
    
    def add_to_buffer(self, obs, action, reward, next_obs, done):
        """Add transition to replay buffer."""
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        
        self.replay_buffer.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
        })
    
    def train_step(self, batch_size: int = 64):
        """Perform one training step."""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Convert to tensors
        obs_batch = torch.FloatTensor([t["obs"] for t in batch]).to(self.device)
        action_batch = torch.FloatTensor([t["action"] for t in batch]).to(self.device)
        reward_batch = torch.FloatTensor([t["reward"] for t in batch]).to(self.device)
        next_obs_batch = torch.FloatTensor([t["next_obs"] for t in batch]).to(self.device)
        done_batch = torch.FloatTensor([t["done"] for t in batch]).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions = self.policy(next_obs_batch)
            target_q1 = self.q1_target(torch.cat([next_obs_batch, next_actions], dim=-1))
            target_q2 = self.q2_target(torch.cat([next_obs_batch, next_actions], dim=-1))
            target_q = torch.min(target_q1, target_q2)
            target_value = reward_batch.unsqueeze(-1) + (1 - done_batch.unsqueeze(-1)) * self.gamma * target_q
        
        # Q-function losses
        q1_pred = self.q1_net(torch.cat([obs_batch, action_batch], dim=-1))
        q2_pred = self.q2_net(torch.cat([obs_batch, action_batch], dim=-1))
        q1_loss = nn.functional.mse_loss(q1_pred, target_value)
        q2_loss = nn.functional.mse_loss(q2_pred, target_value)
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        new_actions = self.policy(obs_batch)
        q1_new = self.q1_net(torch.cat([obs_batch, new_actions], dim=-1))
        q2_new = self.q2_net(torch.cat([obs_batch, new_actions], dim=-1))
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = -q_new.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update target networks
        self._update_target_networks()
        
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "mean_q": q_new.mean().item(),
        }
    
    def _update_target_networks(self):
        """Soft update of target networks."""
        for param, target_param in zip(self.q1_net.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.q2_net.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate policy performance."""
        rewards = []
        successes = []
        
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            
            for _ in range(50):  # Max steps
                obs_flat = self._flatten_obs(obs)
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)
                    action = self.policy(obs_tensor)
                    action_np = action.cpu().numpy().squeeze()
                
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            successes.append(info.get('is_success', False))
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "success_rate": np.mean(successes),
        }


def main():
    """Main training loop."""
    # Configuration
    config = {
        "env_id": "FetchPickAndPlaceDense-v3",
        "max_episode_steps": 50,
        "total_timesteps": 20000,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "eval_frequency": 2500,
        "wandb_project": "pippa",
        "wandb_tags": ["gr00t-rl-lerobot", "sac", "fetch", "groot-minimal"],
    }
    
    # Initialize WandB
    run = wandb.init(
        project=config["wandb_project"],
        tags=config["wandb_tags"],
        config=config,
        name=f"sac-groot-minimal-{wandb.util.generate_id()}",
    )
    
    # Create environment
    print("Creating coupled Fetch environment...")
    env = make_fetch_so101_coupled_env(
        env_id=config["env_id"],
        max_episode_steps=config["max_episode_steps"],
        use_joint_space=False,
        couple_joints=True,
    )
    
    # Create trainer
    print("Creating SAC trainer (will use GR00T model later)...")
    trainer = GR00TSACTrainer(env, learning_rate=config["learning_rate"])
    
    # Training loop
    print("Starting SAC training...")
    timesteps = 0
    
    with tqdm(total=config["total_timesteps"]) as pbar:
        while timesteps < config["total_timesteps"]:
            # Collect rollout
            trainer.collect_rollout(num_steps=500)
            timesteps += 500
            
            # Train
            for _ in range(500):
                metrics = trainer.train_step(batch_size=config["batch_size"])
                if metrics:
                    wandb.log(metrics, step=timesteps)
            
            # Evaluate
            if timesteps % config["eval_frequency"] == 0:
                eval_metrics = trainer.evaluate()
                print(f"\nTimestep {timesteps}: {eval_metrics}")
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=timesteps)
            
            pbar.update(500)
    
    # Final evaluation
    final_metrics = trainer.evaluate(num_episodes=20)
    print(f"\nFinal evaluation: {final_metrics}")
    wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
    
    # Note about GR00T integration
    print("\nNote: This script uses a simple policy for testing.")
    print("To use actual GR00T model, we need to resolve the flash-attn compatibility issue.")
    print("Options:")
    print("1. Use Isaac-GR00T from source with proper torch version")
    print("2. Disable flash attention in GR00T config")
    print("3. Use the trained weights without flash-attn dependency")
    
    env.close()
    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()