#!/usr/bin/env python3
"""
Training script for Isaac Lab integration with our GR00T PPO implementation.
This bridges our PPO with Isaac Lab's environment infrastructure.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
import wandb

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "IsaacLab"))
sys.path.append(str(Path(__file__).parent.parent.parent / "rsl_rl"))

from algorithms.gr00t_wrapper import GR00TActorCritic
from configs.isaac_lab_ppo_cfg import (
    GR00TTestPPOCfg, 
    GR00TFrozenPPOCfg,
    GR00TReachingPPOCfg,
    GR00TManipulationPPOCfg
)
from utils.logging import get_system_metrics


class IsaacLabPPOTrainer:
    """
    Trainer that connects our GR00T PPO to Isaac Lab environments.
    """
    
    def __init__(self, env_name: str, cfg: Any, args: argparse.Namespace):
        self.env_name = env_name
        self.cfg = cfg
        self.args = args
        
        # Set device
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create environment
        self.env = self._create_environment()
        
        # Create model
        self.model = self._create_model()
        
        # Create PPO algorithm
        self.ppo = self._create_ppo_algorithm()
        
        # Initialize logging
        if args.track:
            self._init_wandb()
            
        # Training state
        self.global_step = 0
        self.start_time = time.time()
        
    def _create_environment(self):
        """Create Isaac Lab environment."""
        try:
            # Import Isaac Lab components
            from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
            from isaaclab.utils.config import configclass
            from isaaclab_tasks import isaaclab_task_registry
            
            # Get environment config
            env_cfg = isaaclab_task_registry.get_cfgs(self.env_name)
            
            # Create environment
            env = DirectRLEnv(cfg=env_cfg, device=self.device)
            
            print(f"Created environment: {self.env_name}")
            print(f"  Observation shape: {env.observation_space.shape}")
            print(f"  Action shape: {env.action_space.shape}")
            print(f"  Num envs: {env.num_envs}")
            
            return env
            
        except Exception as e:
            print(f"Failed to create Isaac Lab environment: {e}")
            print("Falling back to mock environment for testing")
            return self._create_mock_environment()
            
    def _create_mock_environment(self):
        """Create mock environment for testing without Isaac Lab."""
        class MockEnv:
            def __init__(self, num_envs=16):
                self.num_envs = num_envs
                self.num_obs = 45
                self.num_actions = 12
                self.observation_space = type('obj', (object,), {
                    'shape': (self.num_obs,)
                })
                self.action_space = type('obj', (object,), {
                    'shape': (self.num_actions,)
                })
                self.device = "cpu"
                
            def reset(self):
                return torch.randn(self.num_envs, self.num_obs)
                
            def step(self, actions):
                obs = torch.randn(self.num_envs, self.num_obs)
                rewards = torch.randn(self.num_envs)
                dones = torch.zeros(self.num_envs, dtype=torch.bool)
                info = {}
                return obs, rewards, dones, info
                
        return MockEnv()
        
    def _create_model(self):
        """Create GR00T actor-critic model."""
        # Get dimensions from environment
        num_actor_obs = self.env.observation_space.shape[0]
        num_critic_obs = getattr(self.env, 'num_privileged_obs', num_actor_obs)
        num_actions = self.env.action_space.shape[0]
        
        # Create model
        model = GR00TActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            **vars(self.cfg.policy_cfg),
            device=self.device
        ).to(self.device)
        
        return model
        
    def _create_ppo_algorithm(self):
        """Create PPO algorithm (rsl_rl compatible)."""
        try:
            from rsl_rl.algorithms import PPO
            from rsl_rl.modules import ActorCritic
            
            # Create PPO with our model
            ppo = PPO(
                actor_critic=self.model,
                num_learning_epochs=self.cfg.algorithm_cfg.num_learning_epochs,
                num_mini_batches=self.cfg.algorithm_cfg.num_mini_batches,
                clip_param=self.cfg.algorithm_cfg.clip_param,
                gamma=self.cfg.algorithm_cfg.gamma,
                lam=self.cfg.algorithm_cfg.lam,
                value_loss_coef=self.cfg.algorithm_cfg.value_loss_coef,
                entropy_coef=self.cfg.algorithm_cfg.entropy_coef,
                learning_rate=self.cfg.algorithm_cfg.learning_rate,
                max_grad_norm=self.cfg.algorithm_cfg.max_grad_norm,
                use_clipped_value_loss=self.cfg.algorithm_cfg.use_clipped_value_loss,
                schedule=self.cfg.algorithm_cfg.schedule,
                desired_kl=self.cfg.algorithm_cfg.desired_kl,
                device=self.device
            )
            
            print("Created rsl_rl PPO algorithm")
            return ppo
            
        except Exception as e:
            print(f"Failed to create rsl_rl PPO: {e}")
            print("Using mock PPO for testing")
            return self._create_mock_ppo()
            
    def _create_mock_ppo(self):
        """Create mock PPO for testing."""
        class MockPPO:
            def __init__(self, model, cfg):
                self.model = model
                self.cfg = cfg
                self.optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=cfg.learning_rate,
                    eps=cfg.adam_epsilon
                )
                
            def update(self, obs, actions, rewards, dones, values, log_probs):
                # Mock update
                loss = torch.tensor(0.1)
                return {"loss": loss.item()}
                
        return MockPPO(self.model, self.cfg.algorithm_cfg)
        
    def _init_wandb(self):
        """Initialize WandB logging."""
        wandb.init(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            name=f"{self.cfg.experiment_name}_{self.env_name}_{int(time.time())}",
            config={
                "env_name": self.env_name,
                "experiment": self.cfg.experiment_name,
                **vars(self.cfg),
                **vars(self.args)
            },
            tags=["isaac-lab", "ppo", "gr00t", self.env_name]
        )
        
    def train(self):
        """Main training loop."""
        print(f"\nStarting training: {self.cfg.experiment_name}")
        print(f"Environment: {self.env_name}")
        print(f"Max iterations: {self.cfg.max_iterations}")
        print(f"Steps per env: {self.cfg.num_steps_per_env}")
        
        # Reset environment
        obs = self.env.reset()
        
        # Training loop
        for iteration in range(self.cfg.max_iterations):
            # Collect rollouts
            rollout_data = self.collect_rollouts(obs)
            obs = rollout_data["last_obs"]
            
            # Update policy
            update_info = self.update_policy(rollout_data)
            
            # Logging
            if iteration % self.cfg.log_interval == 0:
                self.log_training(iteration, rollout_data, update_info)
                
            # Save checkpoint
            if iteration % self.cfg.save_interval == 0:
                self.save_checkpoint(iteration)
                
            # Evaluation
            if iteration % self.cfg.eval_interval == 0:
                self.evaluate(iteration)
                
        print("\nTraining completed!")
        
    def collect_rollouts(self, start_obs):
        """Collect rollouts from environment."""
        rollout_data = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "log_probs": []
        }
        
        obs = start_obs
        
        for step in range(self.cfg.num_steps_per_env):
            # Get actions from policy
            with torch.no_grad():
                if hasattr(self.model, 'get_actions_log_prob'):
                    actions, log_probs = self.model.get_actions_log_prob(obs)
                else:
                    actions = self.model.act(obs)
                    log_probs = torch.zeros(actions.shape[0])
                    
                values = self.model.get_value(obs)
                
            # Step environment
            next_obs, rewards, dones, info = self.env.step(actions)
            
            # Store data
            rollout_data["obs"].append(obs)
            rollout_data["actions"].append(actions)
            rollout_data["rewards"].append(rewards)
            rollout_data["dones"].append(dones)
            rollout_data["values"].append(values)
            rollout_data["log_probs"].append(log_probs)
            
            obs = next_obs
            self.global_step += self.env.num_envs
            
        # Stack data
        for key in rollout_data:
            if key != "last_obs":
                rollout_data[key] = torch.stack(rollout_data[key])
                
        rollout_data["last_obs"] = obs
        
        return rollout_data
        
    def update_policy(self, rollout_data):
        """Update policy using PPO."""
        # Compute returns and advantages
        with torch.no_grad():
            last_values = self.model.get_value(rollout_data["last_obs"])
            
        # Simple return calculation (can be improved with GAE)
        returns = []
        advantages = []
        
        next_value = last_values
        for step in reversed(range(self.cfg.num_steps_per_env)):
            reward = rollout_data["rewards"][step]
            done = rollout_data["dones"][step]
            value = rollout_data["values"][step]
            
            # Simple TD return
            ret = reward + self.cfg.algorithm_cfg.gamma * next_value * (~done)
            advantage = ret - value
            
            returns.insert(0, ret)
            advantages.insert(0, advantage)
            next_value = value
            
        returns = torch.stack(returns)
        advantages = torch.stack(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        if hasattr(self.ppo, 'update'):
            update_info = self.ppo.update(
                rollout_data["obs"].reshape(-1, *rollout_data["obs"].shape[2:]),
                rollout_data["actions"].reshape(-1, *rollout_data["actions"].shape[2:]),
                returns.reshape(-1),
                advantages.reshape(-1),
                rollout_data["values"].reshape(-1),
                rollout_data["log_probs"].reshape(-1)
            )
        else:
            # Mock update
            update_info = {"loss": 0.1, "policy_loss": 0.05, "value_loss": 0.05}
            
        return update_info
        
    def log_training(self, iteration, rollout_data, update_info):
        """Log training metrics."""
        # Compute metrics
        fps = self.global_step / (time.time() - self.start_time)
        
        metrics = {
            "train/iteration": iteration,
            "train/global_step": self.global_step,
            "train/fps": fps,
            "train/mean_reward": rollout_data["rewards"].mean().item(),
            "train/mean_value": rollout_data["values"].mean().item(),
            **update_info,
            **get_system_metrics()
        }
        
        # Console logging
        print(f"\nIteration {iteration}/{self.cfg.max_iterations}")
        print(f"  Global step: {self.global_step}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Mean reward: {metrics['train/mean_reward']:.3f}")
        
        for key, value in update_info.items():
            print(f"  {key}: {value:.4f}")
            
        # WandB logging
        if self.args.track:
            wandb.log(metrics, step=self.global_step)
            
    def save_checkpoint(self, iteration):
        """Save training checkpoint."""
        checkpoint_dir = Path(f"checkpoints/{self.cfg.experiment_name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "iteration": iteration,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "config": self.cfg
        }
        
        if hasattr(self.ppo, 'optimizer'):
            checkpoint["optimizer_state_dict"] = self.ppo.optimizer.state_dict()
            
        path = checkpoint_dir / f"checkpoint_{iteration}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
    def evaluate(self, iteration):
        """Evaluate policy."""
        print(f"\nEvaluating at iteration {iteration}...")
        
        eval_rewards = []
        
        for ep in range(self.cfg.eval_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    action = self.model.act_inference(obs)
                    
                obs, reward, dones, info = self.env.step(action)
                episode_reward += reward.sum().item()
                done = dones.any()
                
            eval_rewards.append(episode_reward / self.env.num_envs)
            
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        
        if self.args.track:
            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/std_reward": std_reward
            }, step=self.global_step)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    
    # Environment
    parser.add_argument("--env", type=str, default="Isaac-Reach-Franka-v0",
                        help="Isaac Lab environment name")
    
    # Configuration
    parser.add_argument("--config", type=str, default="test",
                        choices=["test", "frozen", "reaching", "manipulation"],
                        help="Configuration preset")
    
    # Logging
    parser.add_argument("--track", action="store_true", 
                        help="Enable WandB tracking")
    parser.add_argument("--wandb-project", type=str, default="gr00t-isaac-lab")
    parser.add_argument("--wandb-entity", type=str, default=None)
    
    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration
    configs = {
        "test": GR00TTestPPOCfg(),
        "frozen": GR00TFrozenPPOCfg(),
        "reaching": GR00TReachingPPOCfg(),
        "manipulation": GR00TManipulationPPOCfg()
    }
    cfg = configs[args.config]
    
    # Override device
    cfg.device = args.device
    cfg.policy_cfg.device = args.device
    
    # Create trainer
    trainer = IsaacLabPPOTrainer(args.env, cfg, args)
    
    # Train
    trainer.train()
    
    # Cleanup
    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()