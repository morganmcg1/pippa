#!/usr/bin/env python3
"""
PPO configuration for GR00T training.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    """PPO hyperparameters for GR00T training."""
    
    # Model configuration
    model_path: str = "nvidia/GR00T-N1.5-3B"
    embodiment_tag: str = "GR1"
    freeze_backbone: bool = True
    freeze_action_head: bool = False
    critic_hidden_dim: int = 512
    
    # Environment configuration
    env_name: str = "FrankaCubeLift-v0"
    num_envs: int = 8  # Parallel environments
    max_episode_steps: int = 200
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    n_steps: int = 2048  # Steps per environment before update
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # Value function clipping
    normalize_advantage: bool = True
    ent_coef: float = 0.01  # Entropy coefficient
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5
    
    # Training configuration
    total_timesteps: int = 1_000_000
    log_interval: int = 1
    save_interval: int = 10
    eval_interval: int = 10
    eval_episodes: int = 10
    
    # Experiment configuration
    exp_name: str = "ppo_gr00t_cubelift"
    seed: int = 42
    device: str = "cuda"
    
    # WandB configuration
    use_wandb: bool = True
    wandb_project: str = "gr00t-rl"
    wandb_entity: Optional[str] = None
    wandb_tags: list = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.wandb_tags is None:
            self.wandb_tags = ["ppo", "gr00t", self.env_name]
            
        # Adjust steps based on number of environments
        self.buffer_size = self.n_steps * self.num_envs
        self.n_updates = self.total_timesteps // self.buffer_size