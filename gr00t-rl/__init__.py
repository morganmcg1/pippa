"""GR00T-RL: Reinforcement Learning for NVIDIA GR00T Robot Foundation Model."""

__version__ = "0.1.0"

# Import key components for easier access
from .algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from .algorithms.gr00t_wrapper import GR00TActorCritic
from .configs.ppo_config_v2 import PPOConfigV2
from .utils.buffers import PPORolloutBuffer
from .utils.normalization import VecNormalize, ObservationNormalizer, RewardNormalizer

__all__ = [
    "PPOGr00tActorCriticV2",
    "GR00TActorCritic", 
    "PPOConfigV2",
    "PPORolloutBuffer",
    "VecNormalize",
    "ObservationNormalizer",
    "RewardNormalizer",
]