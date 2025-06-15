#!/usr/bin/env python3
"""
Improved PPO configuration with all 37 implementation details.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class PPOConfigV2:
    """Improved PPO configuration following CleanRL best practices."""
    
    # Environment configuration
    env_name: str = "FrankaCubeLift-v0"
    num_envs: int = 4  # Number of parallel environments
    max_episode_steps: int = 1000
    capture_video: bool = False
    video_freq: int = 50  # Capture video every N episodes
    
    # Model architecture
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "tanh"  # One of: tanh, relu, elu, leaky_relu
    use_layer_norm: bool = False  # Layer norm for robotics
    
    # Gaussian policy settings
    log_std_init: float = 0.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    
    # Multi-modal settings
    use_multimodal_encoder: bool = True
    vision_dim: Optional[int] = None  # Set if using pre-extracted features
    proprio_dim: Optional[int] = 14  # Joint positions + velocities
    language_dim: Optional[int] = 768  # Language embedding dimension
    encoder_output_dim: int = 256
    
    # PPO hyperparameters (37 implementation details)
    learning_rate: float = 3e-4
    anneal_lr: bool = True  # Linear annealing to 0
    adam_epsilon: float = 1e-5  # Different from default 1e-8!
    n_steps: int = 2048  # Steps per environment before update
    batch_size: int = 64  # Mini-batch size
    n_epochs: int = 10  # Update epochs
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_range: float = 0.2  # PPO clip parameter
    clip_range_vf: Optional[float] = None  # Value function clipping
    normalize_advantage: bool = True  # At mini-batch level
    ent_coef: float = 0.0  # Entropy coefficient (0 for continuous)
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5  # Global gradient clipping
    target_kl: Optional[float] = None  # Early stopping if KL too high
    
    # Normalization
    norm_obs: bool = True  # Normalize observations
    norm_reward: bool = True  # Normalize rewards
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    
    # Training configuration
    total_timesteps: int = 1_000_000
    log_interval: int = 1  # Log every N updates
    save_interval: int = 100  # Save every N updates
    eval_interval: int = 100  # Evaluate every N updates
    eval_episodes: int = 10  # Episodes for evaluation
    
    # Experiment configuration
    exp_name: str = "ppo_v2_gr00t"
    seed: int = 1
    torch_deterministic: bool = True  # For reproducibility
    cuda: bool = True
    device: str = "cuda"
    
    # Logging configuration
    track: bool = True  # Use WandB
    wandb_project_name: str = "gr00t-rl"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[list] = None
    capture_video_freq: int = 100  # Capture video every N episodes
    log_video: bool = True  # Log videos to WandB
    
    # Advanced features
    use_lstm: bool = False  # Use LSTM for partial observability
    lstm_hidden_size: int = 256
    use_sde: bool = False  # State-dependent exploration
    sde_sample_freq: int = -1  # Sample new noise at each step
    
    # Safety and constraints
    action_clip: float = 1.0  # Clip actions to [-1, 1]
    reward_scale: float = 1.0  # Scale rewards
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.wandb_tags is None:
            self.wandb_tags = ["ppo-v2", "gr00t", self.env_name]
            
        # Calculate derived values
        self.batch_size = int(self.num_envs * self.n_steps)
        self.minibatch_size = int(self.batch_size // self.n_epochs)
        self.num_updates = self.total_timesteps // self.batch_size
        
        # Ensure batch size is divisible by minibatch size
        assert self.batch_size % self.minibatch_size == 0, \
            f"Batch size {self.batch_size} must be divisible by minibatch size {self.minibatch_size}"