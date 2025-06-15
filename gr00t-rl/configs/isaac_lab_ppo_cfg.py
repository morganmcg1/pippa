#!/usr/bin/env python3
"""
PPO configuration for Isaac Lab with GR00T integration.
Based on rsl_rl PPO config structure.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class GR00TPPOActorCriticCfg:
    """Configuration for GR00T actor-critic."""
    
    # GR00T settings
    gr00t_model_path: str = "nvidia/GR00T-N1.5-3B"
    embodiment_tag: str = "GR1"
    freeze_backbone: bool = True
    use_gr00t: bool = True  # Can disable for testing
    
    # Network architecture (fallback if GR00T unavailable)
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    
    # Action noise
    init_noise_std: float = 1.0
    action_clip: float = 1.0
    
    # Device
    device: str = "cuda"


@dataclass
class GR00TPPOAlgorithmCfg:
    """PPO algorithm configuration with conservative settings for large model."""
    
    # PPO core settings
    clip_param: float = 0.1  # Smaller than usual (0.2) for large model
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0  # No entropy for continuous control
    use_clipped_value_loss: bool = True
    
    # Training settings
    num_learning_epochs: int = 3  # Fewer epochs for large model
    num_mini_batches: int = 4
    learning_rate: float = 1e-5  # Much smaller for fine-tuning
    schedule: str = "adaptive"  # or "fixed"
    
    # GAE settings
    gamma: float = 0.99
    lam: float = 0.95
    
    # KL divergence settings
    desired_kl: float = 0.01
    max_kl: float = 0.03  # Stop training if KL too high
    
    # Gradient clipping
    max_grad_norm: float = 0.5
    
    # Adam epsilon (critical detail from ICLR blog)
    adam_epsilon: float = 1e-5


@dataclass
class GR00TPPORunnerCfg:
    """Runner configuration for Isaac Lab integration."""
    
    # Environment interaction
    num_steps_per_env: int = 24  # Steps before PPO update
    max_iterations: int = 1500
    
    # Experiment settings
    experiment_name: str = "gr00t_ppo"
    run_name: Optional[str] = None
    
    # Logging and saving
    save_interval: int = 50
    log_interval: int = 1
    
    # Normalization
    empirical_normalization: bool = True  # Use running mean/std
    
    # Evaluation
    eval_interval: int = 50
    eval_episodes: int = 10
    
    # Checkpointing
    resume: bool = False
    resume_path: Optional[str] = None
    
    # Algorithm and policy
    policy_cfg: GR00TPPOActorCriticCfg = field(default_factory=GR00TPPOActorCriticCfg)
    algorithm_cfg: GR00TPPOAlgorithmCfg = field(default_factory=GR00TPPOAlgorithmCfg)
    
    # Device settings
    device: str = "cuda"
    num_threads: int = 4  # CPU threads for parallel envs
    
    # Memory management for large model
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False  # Enable if memory constrained


# Preset configurations for different scenarios

@dataclass
class GR00TTestPPOCfg(GR00TPPORunnerCfg):
    """Test configuration with small MLP instead of GR00T."""
    
    def __post_init__(self):
        self.experiment_name = "gr00t_test_mlp"
        self.policy_cfg.use_gr00t = False
        self.policy_cfg.actor_hidden_dims = (128, 128)
        self.policy_cfg.critic_hidden_dims = (128, 128)
        self.algorithm_cfg.learning_rate = 3e-4  # Standard LR for small network


@dataclass
class GR00TFrozenPPOCfg(GR00TPPORunnerCfg):
    """Configuration with frozen GR00T backbone."""
    
    def __post_init__(self):
        self.experiment_name = "gr00t_frozen"
        self.policy_cfg.freeze_backbone = True
        self.algorithm_cfg.num_learning_epochs = 5  # Can do more epochs


@dataclass
class GR00TLoRAPPOCfg(GR00TPPORunnerCfg):
    """Configuration for LoRA fine-tuning (future implementation)."""
    
    def __post_init__(self):
        self.experiment_name = "gr00t_lora"
        self.policy_cfg.freeze_backbone = False  # But will use LoRA
        # TODO: Add LoRA config parameters


@dataclass
class GR00TProgressiveUnfreezeCfg(GR00TPPORunnerCfg):
    """Configuration for progressive unfreezing."""
    
    def __post_init__(self):
        self.experiment_name = "gr00t_progressive"
        self.unfreeze_schedule = {
            0: ["action_head"],
            100: ["transformer.layers[-2:]"],
            200: ["transformer.layers[-4:]"],
            300: ["transformer.layers[-8:]"],
        }


# Task-specific configurations

@dataclass
class GR00TReachingPPOCfg(GR00TPPORunnerCfg):
    """Configuration for simple reaching task."""
    
    def __post_init__(self):
        self.experiment_name = "gr00t_reaching"
        self.num_steps_per_env = 16  # Shorter episodes
        self.max_iterations = 500


@dataclass 
class GR00TManipulationPPOCfg(GR00TPPORunnerCfg):
    """Configuration for manipulation tasks."""
    
    def __post_init__(self):
        self.experiment_name = "gr00t_manipulation"
        self.num_steps_per_env = 32
        self.algorithm_cfg.clip_param = 0.05  # Even more conservative


@dataclass
class GR00THumanoidPPOCfg(GR00TPPORunnerCfg):
    """Configuration for full humanoid control."""
    
    def __post_init__(self):
        self.experiment_name = "gr00t_humanoid"
        self.num_steps_per_env = 48  # Longer episodes
        self.max_iterations = 3000
        self.policy_cfg.init_noise_std = 0.5  # Less exploration