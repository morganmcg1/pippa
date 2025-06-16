#!/usr/bin/env python3
"""
Simple SAC training script for Fetch environment without GR00T policy.
Uses a basic CNN policy to test the training pipeline.
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import wandb
from pathlib import Path
from tqdm import tqdm
import json
import os
from dotenv import load_dotenv

from environments.fetch_so101_coupled import make_fetch_so101_coupled_env

# Load environment variables
load_dotenv()


class SimpleCNNPolicy(nn.Module):
    """Simple CNN policy for Fetch environment."""
    
    def __init__(self, action_dim: int = 6):
        super().__init__()
        
        # Vision encoder (shared for both cameras)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(256 * 2 + 128, 256),  # 2 cameras + state
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        # Extract inputs
        front_img = batch["observation"]["images"]["front"]
        wrist_img = batch["observation"]["images"]["wrist"]
        state = batch["observation"]["state"]
        
        # Normalize images
        front_img = front_img.float() / 255.0
        wrist_img = wrist_img.float() / 255.0
        
        # Encode images
        front_features = self.vision_encoder(front_img.permute(0, 3, 1, 2))
        wrist_features = self.vision_encoder(wrist_img.permute(0, 3, 1, 2))
        
        # Encode state
        state_features = self.state_encoder(state.float())
        
        # Combine and predict actions
        combined = torch.cat([front_features, wrist_features, state_features], dim=-1)
        actions = torch.tanh(self.policy_head(combined))  # Actions in [-1, 1]
        
        return actions
    
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action with exploration noise."""
        with torch.no_grad():
            actions = self.forward(batch)
            # Add exploration noise
            noise = torch.randn_like(actions) * 0.1
            actions = torch.clamp(actions + noise, -1.0, 1.0)
        return actions


class SimpleSACTrainer:
    """Minimal SAC trainer."""
    
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
        
        # Create policy and critics
        self.policy = SimpleCNNPolicy().to(self.device)
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
        
    def _create_q_network(self):
        """Create Q-network."""
        class QNetwork(nn.Module):
            def __init__(self, obs_dim=406, action_dim=6):  # 200+200+6 = 406
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim + action_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                )
                
            def forward(self, obs_features, action):
                x = torch.cat([obs_features, action], dim=-1)
                return self.net(x)
        
        return QNetwork()
    
    def collect_rollout(self, num_steps: int = 1000):
        """Collect experience."""
        obs, info = self.env.reset()
        
        for _ in range(num_steps):
            # Convert observation
            batch = self._obs_to_batch(obs)
            
            # Get action
            action = self.policy.select_action(batch)
            action_np = action.cpu().numpy().squeeze()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            
            # Store transition
            self.add_to_buffer(obs, action_np, reward, next_obs, terminated)
            
            if terminated or truncated:
                obs, info = self.env.reset()
            else:
                obs = next_obs
    
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
        obs_batch = self._stack_observations([t["obs"] for t in batch])
        action_batch = torch.FloatTensor([t["action"] for t in batch]).to(self.device)
        reward_batch = torch.FloatTensor([t["reward"] for t in batch]).to(self.device)
        next_obs_batch = self._stack_observations([t["next_obs"] for t in batch])
        done_batch = torch.FloatTensor([t["done"] for t in batch]).to(self.device)
        
        # Get features
        with torch.no_grad():
            obs_features = self._get_features(obs_batch)
            next_obs_features = self._get_features(next_obs_batch)
        
        # Update critics
        with torch.no_grad():
            next_actions = self.policy(next_obs_batch)
            target_q1 = self.q1_target(next_obs_features, next_actions)
            target_q2 = self.q2_target(next_obs_features, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = reward_batch.unsqueeze(-1) + (1 - done_batch.unsqueeze(-1)) * self.gamma * target_q
        
        # Q-function losses
        q1_pred = self.q1_net(obs_features, action_batch)
        q2_pred = self.q2_net(obs_features, action_batch)
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
        q1_new = self.q1_net(obs_features.detach(), new_actions)
        q2_new = self.q2_net(obs_features.detach(), new_actions)
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
    
    def _obs_to_batch(self, obs):
        """Convert single observation to batch format."""
        return {
            "observation": {
                "images": {
                    "front": torch.from_numpy(obs["observation"]["images"]["front"]).unsqueeze(0).to(self.device),
                    "wrist": torch.from_numpy(obs["observation"]["images"]["wrist"]).unsqueeze(0).to(self.device),
                },
                "state": torch.from_numpy(obs["observation"]["state"]).unsqueeze(0).to(self.device),
            }
        }
    
    def _stack_observations(self, obs_list):
        """Stack list of observations into batch."""
        return {
            "observation": {
                "images": {
                    "front": torch.stack([
                        torch.from_numpy(o["observation"]["images"]["front"])
                        for o in obs_list
                    ]).to(self.device),
                    "wrist": torch.stack([
                        torch.from_numpy(o["observation"]["images"]["wrist"])
                        for o in obs_list
                    ]).to(self.device),
                },
                "state": torch.stack([
                    torch.from_numpy(o["observation"]["state"])
                    for o in obs_list
                ]).to(self.device),
            }
        }
    
    def _get_features(self, batch):
        """Extract features using simplified approach."""
        # For simplicity, just flatten everything
        front = batch["observation"]["images"]["front"].to(self.device)
        wrist = batch["observation"]["images"]["wrist"].to(self.device)
        state = batch["observation"]["state"].to(self.device)
        
        # Simple feature extraction
        front_flat = front.view(front.size(0), -1)[:, :200]  # Take first 200 dims
        wrist_flat = wrist.view(wrist.size(0), -1)[:, :200]
        state_flat = state.view(state.size(0), -1)
        
        return torch.cat([front_flat, wrist_flat, state_flat * 50], dim=-1)  # Scale state
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate policy performance."""
        rewards = []
        successes = []
        
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            
            for _ in range(50):  # Max steps
                batch = self._obs_to_batch(obs)
                with torch.no_grad():
                    action = self.policy(batch)
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
        "total_timesteps": 50000,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "eval_frequency": 2500,
        "wandb_project": "pippa",
        "wandb_tags": ["gr00t-rl-lerobot", "sac", "fetch", "simple-cnn"],
    }
    
    # Initialize WandB
    run = wandb.init(
        project=config["wandb_project"],
        tags=config["wandb_tags"],
        config=config,
        name=f"sac-fetch-simple-{wandb.util.generate_id()}",
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
    trainer = SimpleSACTrainer(env, learning_rate=config["learning_rate"])
    
    # Training loop
    print("Starting SAC training with simple CNN policy...")
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
    
    env.close()
    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()