#!/usr/bin/env python3
"""
GRPO configuration for GR00T training.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GRPOConfig:
    """GRPO hyperparameters for GR00T training."""
    
    # Model configuration
    model_path: str = "nvidia/GR00T-N1.5-3B"
    embodiment_tag: str = "GR1"
    freeze_backbone: bool = True
    
    # Environment configuration
    env_name: str = "FrankaCubeLift-v0"
    max_episode_steps: int = 200
    
    # GRPO hyperparameters
    num_rollouts_per_update: int = 8  # Number of rollouts from same initial state
    learning_rate: float = 5e-5  # Lower than PPO since no critic to train
    n_epochs: int = 3  # Epochs per batch of rollouts
    temperature: float = 1.0  # Action sampling temperature
    beta: float = 0.0  # KL penalty (0 for Dr GRPO)
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    
    # Training configuration
    total_episodes: int = 10000  # Total episodes (not timesteps)
    update_interval: int = 8  # Update after N episodes
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 100
    eval_episodes: int = 10
    
    # Experiment configuration
    exp_name: str = "grpo_gr00t_cubelift"
    seed: int = 42
    device: str = "cuda"
    
    # WandB configuration
    use_wandb: bool = True
    wandb_project: str = "gr00t-rl"
    wandb_entity: Optional[str] = None
    wandb_tags: list = None
    
    # GRPO-specific options
    normalize_rewards: bool = True  # Normalize advantages
    use_kl_penalty: bool = False  # Whether to use KL penalty
    reward_scale: float = 1.0  # Scale rewards before normalization
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.wandb_tags is None:
            self.wandb_tags = ["grpo", "gr00t", self.env_name]
            
        # Ensure consistency
        if self.use_kl_penalty:
            assert self.beta > 0, "beta must be > 0 when using KL penalty"
        else:
            self.beta = 0.0
            
        # Number of updates
        self.n_updates = self.total_episodes // self.update_interval