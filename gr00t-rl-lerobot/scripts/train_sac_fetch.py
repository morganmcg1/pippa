"""
SAC training script for GR00T policy on Fetch environment.

This demonstrates how to use reinforcement learning to fine-tune
the GR00T policy on the Fetch Pick and Place task.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from typing import Dict, Optional
import wandb
from pathlib import Path
from tqdm import tqdm
import json

from environments.fetch_wrapper import make_fetch_so101_env
from environments.fetch_so101_coupled import make_fetch_so101_coupled_env
from policies.gr00t_policy import GR00TPolicy, GR00TConfig


class SACTrainer:
    """
    Simplified SAC trainer for GR00T policy.
    
    In a full implementation, this would use LeRobot's SAC,
    but for now we implement a minimal version.
    """
    
    def __init__(
        self,
        env,
        policy: GR00TPolicy,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "cuda",
    ):
        self.env = env
        self.policy = policy
        self.device = torch.device(device)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Temperature parameter
        
        # Create Q-networks (critics)
        self.q1_net = self._create_q_network().to(self.device)
        self.q2_net = self._create_q_network().to(self.device)
        self.q1_target = self._create_q_network().to(self.device)
        self.q2_target = self._create_q_network().to(self.device)
        
        # Copy parameters to targets
        self.q1_target.load_state_dict(self.q1_net.state_dict())
        self.q2_target.load_state_dict(self.q2_net.state_dict())
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )
        self.q1_optimizer = torch.optim.Adam(
            self.q1_net.parameters(), lr=learning_rate
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2_net.parameters(), lr=learning_rate
        )
        
        # Replay buffer (simplified)
        self.replay_buffer = []
        self.buffer_size = 100000
        
    def _create_q_network(self):
        """Create a Q-network (critic)."""
        import torch.nn as nn
        
        class QNetwork(nn.Module):
            def __init__(self, obs_dim=512 * 2 + 256, action_dim=6):
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
        """Collect experience in the environment."""
        obs, info = self.env.reset()
        
        for _ in range(num_steps):
            # Convert observation
            batch = self._obs_to_batch(obs)
            
            # Get action
            with torch.no_grad():
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
    
    def train_step(self, batch_size: int = 256):
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
        
        # Get features from observations
        obs_features = self._get_features(obs_batch)
        next_obs_features = self._get_features(next_obs_batch)
        
        # Update critics
        with torch.no_grad():
            # Sample next actions
            next_actions = self.policy.select_action(next_obs_batch)
            
            # Compute target Q-values
            target_q1 = self.q1_target(next_obs_features, next_actions)
            target_q2 = self.q2_target(next_obs_features, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = reward_batch.unsqueeze(-1) + (1 - done_batch.unsqueeze(-1)) * self.gamma * target_q
        
        # Q-function losses
        q1_pred = self.q1_net(obs_features, action_batch)
        q2_pred = self.q2_net(obs_features, action_batch)
        q1_loss = torch.nn.functional.mse_loss(q1_pred, target_value)
        q2_loss = torch.nn.functional.mse_loss(q2_pred, target_value)
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        new_actions = self.policy.select_action(obs_batch)
        q1_new = self.q1_net(obs_features, new_actions)
        q2_new = self.q2_net(obs_features, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = -q_new.mean()  # Maximize Q-value
        
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
                    "front": torch.from_numpy(obs["observation"]["images"]["front"]).float(),
                    "wrist": torch.from_numpy(obs["observation"]["images"]["wrist"]).float(),
                },
                "state": torch.from_numpy(obs["observation"]["state"]).float(),
            }
        }
    
    def _stack_observations(self, obs_list):
        """Stack list of observations into batch."""
        return {
            "observation": {
                "images": {
                    "front": torch.stack([
                        torch.from_numpy(o["observation"]["images"]["front"]).float()
                        for o in obs_list
                    ]),
                    "wrist": torch.stack([
                        torch.from_numpy(o["observation"]["images"]["wrist"]).float()
                        for o in obs_list
                    ]),
                },
                "state": torch.stack([
                    torch.from_numpy(o["observation"]["state"]).float()
                    for o in obs_list
                ]),
            }
        }
    
    def _get_features(self, batch):
        """Extract features from observations using policy encoder."""
        with torch.no_grad():
            # Use policy's encoders
            front_img = batch["observation"]["images"]["front"].to(self.device)
            wrist_img = batch["observation"]["images"]["wrist"].to(self.device)
            state = batch["observation"]["state"].to(self.device)
            
            # Normalize images
            front_img = front_img / 255.0
            wrist_img = wrist_img / 255.0
            
            # Encode
            front_features = self.policy.vision_encoder(front_img.permute(0, 3, 1, 2))
            wrist_features = self.policy.vision_encoder(wrist_img.permute(0, 3, 1, 2))
            state_features = self.policy.state_encoder(state)
            
            # Combine
            features = torch.cat([front_features, wrist_features, state_features], dim=-1)
            
        return features
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate policy performance."""
        rewards = []
        successes = []
        
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.env._max_episode_steps):
                batch = self._obs_to_batch(obs)
                with torch.no_grad():
                    action = self.policy.select_action(batch)
                    action_np = action.cpu().numpy().squeeze()
                
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            # Check if object reached goal (simplified success metric)
            successes.append(info.get("is_success", reward > -1.0))
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "success_rate": np.mean(successes),
        }


def main(
    env_type: str = "cartesian",  # "cartesian" or "coupled"
    use_joint_space: bool = False,
):
    """Main training loop.
    
    Args:
        env_type: Type of environment wrapper to use
        use_joint_space: Whether to use joint-space actions (only for coupled)
    """
    # Configuration
    config = {
        "env_type": env_type,
        "use_joint_space": use_joint_space,
        "env_id": "FetchPickAndPlaceDense-v3",
        "max_episode_steps": 50,
        "total_timesteps": 100000,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "eval_frequency": 5000,
        "save_frequency": 10000,
        "wandb_project": "pippa",
        "wandb_tags": ["gr00t-rl-lerobot", "sac", "fetch", f"env-{env_type}"],
    }
    
    # Initialize WandB
    run = wandb.init(
        project=config["wandb_project"],
        tags=config["wandb_tags"],
        config=config,
        name=f"gr00t-sac-fetch-{env_type}-{wandb.util.generate_id()}",
    )
    
    # Create environment based on type
    if env_type == "coupled":
        env = make_fetch_so101_coupled_env(
            env_id=config["env_id"],
            max_episode_steps=config["max_episode_steps"],
            use_joint_space=use_joint_space,
            couple_joints=True,
        )
        print(f"Using coupled environment with {'joint' if use_joint_space else 'Cartesian'} actions")
    else:
        env = make_fetch_so101_env(
            env_id=config["env_id"],
            max_episode_steps=config["max_episode_steps"],
        )
        print("Using Cartesian-only environment")
    
    # Create policy
    policy_config = GR00TConfig()
    policy = GR00TPolicy(policy_config)
    
    # Create trainer
    trainer = SACTrainer(env, policy, learning_rate=config["learning_rate"])
    
    # Training loop
    print("Starting SAC training...")
    timesteps = 0
    
    with tqdm(total=config["total_timesteps"]) as pbar:
        while timesteps < config["total_timesteps"]:
            # Collect rollout
            trainer.collect_rollout(num_steps=1000)
            timesteps += 1000
            
            # Train
            for _ in range(1000):
                metrics = trainer.train_step(batch_size=config["batch_size"])
                wandb.log(metrics, step=timesteps)
            
            # Evaluate
            if timesteps % config["eval_frequency"] == 0:
                eval_metrics = trainer.evaluate()
                print(f"\nTimestep {timesteps}: {eval_metrics}")
                wandb.log({"eval/" + k: v for k, v in eval_metrics.items()}, step=timesteps)
            
            # Save checkpoint
            if timesteps % config["save_frequency"] == 0:
                save_path = Path(f"checkpoints/sac_fetch_{timesteps}")
                save_path.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")
            
            pbar.update(1000)
    
    # Final evaluation
    final_metrics = trainer.evaluate(num_episodes=50)
    print(f"\nFinal evaluation: {final_metrics}")
    wandb.log({"final/" + k: v for k, v in final_metrics.items()})
    
    # Save final model
    policy.save_pretrained("checkpoints/sac_fetch_final")
    
    env.close()
    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GR00T with SAC on Fetch")
    parser.add_argument(
        "--env-type", 
        type=str, 
        default="cartesian",
        choices=["cartesian", "coupled"],
        help="Environment type: cartesian (7-DoF hidden) or coupled (6-DoF sim)"
    )
    parser.add_argument(
        "--use-joint-space",
        action="store_true",
        help="Use joint-space actions (only for coupled env)"
    )
    args = parser.parse_args()
    
    main(env_type=args.env_type, use_joint_space=args.use_joint_space)