#!/usr/bin/env python3
"""
GR00T Policy Wrapper for RL training.
This wraps the GR00T N1.5 model to work with PPO/GRPO algorithms.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

# Import GR00T components
# Note: These imports assume Isaac-GR00T is installed or in PYTHONPATH
try:
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.data.schema import EmbodimentTag
    from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
    GROOT_AVAILABLE = True
except ImportError:
    print("Warning: Isaac-GR00T not found. Please install it first.")
    GROOT_AVAILABLE = False


class GR00TRLPolicy(nn.Module):
    """
    Wrapper for GR00T N1.5 model to work with RL algorithms (PPO/GRPO).
    
    This class adapts the GR00T model interface to work with standard RL training loops
    by providing actor-critic functionality.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "nvidia/GR00T-N1.5-3B",
        action_dim: int = 4,  # Default for Fetch environments
        device: str = "cuda",
        embodiment_tag: str = "new_embodiment",
        use_pretrained_vision: bool = True,
        use_pretrained_language: bool = True,
        freeze_vision: bool = True,
        freeze_language: bool = True,
        add_value_head: bool = True,
    ):
        """
        Initialize GR00T policy for RL.
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
            action_dim: Dimension of action space
            device: Device to run model on
            embodiment_tag: Embodiment tag for the robot
            use_pretrained_vision: Whether to use pretrained vision encoder
            use_pretrained_language: Whether to use pretrained language model
            freeze_vision: Whether to freeze vision encoder during training
            freeze_language: Whether to freeze language model during training
            add_value_head: Whether to add a value head for actor-critic training
        """
        super().__init__()
        
        if not GROOT_AVAILABLE:
            raise ImportError("Isaac-GR00T is required but not installed.")
        
        self.device = device
        self.action_dim = action_dim
        self.embodiment_tag = EmbodimentTag(embodiment_tag)
        
        # Load GR00T model
        print(f"Loading GR00T model from {model_name_or_path}...")
        self.groot_model = GR00T_N1_5.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            tune_llm=not freeze_language,
            tune_visual=not freeze_vision,
            tune_projector=True,  # Always tune projector for RL
            tune_diffusion_model=True,  # Always tune action head for RL
        )
        
        # Set compute dtype
        self.groot_model.compute_dtype = "bfloat16"
        self.groot_model.config.compute_dtype = "bfloat16"
        
        # Move model to device
        self.groot_model = self.groot_model.to(device)
        
        # Add value head if needed
        if add_value_head:
            # Get hidden size from backbone
            hidden_size = self._get_hidden_size()
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ).to(device)
        else:
            self.value_head = None
    
    def _get_hidden_size(self) -> int:
        """Get the hidden size from the backbone model."""
        # This depends on the specific backbone used
        # For Eagle backbone, we need to check the config
        if hasattr(self.groot_model.backbone, 'hidden_size'):
            return self.groot_model.backbone.hidden_size
        elif hasattr(self.groot_model.backbone, 'config') and hasattr(self.groot_model.backbone.config, 'hidden_size'):
            return self.groot_model.backbone.config.hidden_size
        else:
            # Default hidden size for vision-language models
            return 1024
    
    def _prepare_groot_inputs(self, observations: torch.Tensor) -> Dict[str, Any]:
        """
        Convert standard RL observations to GR00T model inputs.
        
        Args:
            observations: Tensor of shape (batch_size, obs_dim) or dict of observations
            
        Returns:
            Dictionary formatted for GR00T model
        """
        batch_size = observations.shape[0] if isinstance(observations, torch.Tensor) else observations['observation'].shape[0]
        
        # GR00T expects specific input format
        groot_inputs = {
            "embodiment_id": torch.tensor([self.embodiment_tag.value] * batch_size).to(self.device),
        }
        
        # Handle different observation formats
        if isinstance(observations, dict):
            # Goal-conditioned observations (e.g., from Fetch)
            if 'observation' in observations and 'desired_goal' in observations:
                # Concatenate observation and goal
                obs = observations['observation']
                goal = observations['desired_goal']
                full_obs = torch.cat([obs, goal], dim=-1)
            else:
                # Use all values from dict
                full_obs = torch.cat(list(observations.values()), dim=-1)
        else:
            # Simple observation tensor
            full_obs = observations
        
        # For now, we'll use the observation as proprioception
        # In a real setup, you'd properly separate vision, proprioception, etc.
        groot_inputs["proprioception"] = full_obs
        
        # Add dummy vision input if needed (GR00T expects vision)
        # In real usage, this would be actual camera images
        dummy_vision = torch.zeros(
            (batch_size, 3, 224, 224),  # Standard vision input size
            device=self.device,
            dtype=torch.float32
        )
        groot_inputs["vision"] = dummy_vision
        
        return groot_inputs
    
    def get_action_distribution(self, observations: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get action distribution from GR00T model.
        
        Args:
            observations: Input observations
            
        Returns:
            Action distribution
        """
        # Prepare inputs for GR00T
        groot_inputs = self._prepare_groot_inputs(observations)
        
        # Get outputs from GR00T model
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.groot_model(**groot_inputs)
        
        # Extract action predictions
        if "action_pred" in outputs:
            action_mean = outputs["action_pred"]
        else:
            raise ValueError("GR00T model did not return action predictions")
        
        # GR00T uses flow matching which outputs deterministic actions
        # For RL, we need to add some exploration noise
        # Create a Gaussian distribution around the predicted actions
        action_std = torch.ones_like(action_mean) * 0.1  # Fixed std for now
        
        # Ensure correct shape (batch_size, action_dim)
        if len(action_mean.shape) == 3:  # (batch, horizon, action_dim)
            # Take only the first action in the horizon
            action_mean = action_mean[:, 0, :]
            action_std = action_std[:, 0, :]
        
        # Ensure action dimension matches
        if action_mean.shape[-1] != self.action_dim:
            # Project to correct action dimension
            action_mean = action_mean[..., :self.action_dim]
            action_std = action_std[..., :self.action_dim]
        
        return torch.distributions.Normal(action_mean, action_std)
    
    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Get value estimates from the value head.
        
        Args:
            observations: Input observations
            
        Returns:
            Value estimates
        """
        if self.value_head is None:
            raise ValueError("Value head not initialized. Set add_value_head=True.")
        
        # Get backbone features
        groot_inputs = self._prepare_groot_inputs(observations)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Get backbone features
            backbone_outputs = self.groot_model.backbone(**groot_inputs)
            
            # Extract features (this depends on backbone output format)
            if isinstance(backbone_outputs, dict) and "backbone_features" in backbone_outputs:
                features = backbone_outputs["backbone_features"]
            else:
                # Fallback - use last hidden states or pooled output
                features = backbone_outputs
            
            # Pool features if needed
            if len(features.shape) > 2:
                # Global average pooling
                features = features.mean(dim=1)
        
        # Convert to float32 for value head
        features = features.float()
        
        # Get value prediction
        values = self.value_head(features)
        
        return values
    
    def get_action_and_value(
        self,
        observations: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Args:
            observations: Input observations
            action: Optional action to evaluate (if None, sample new action)
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        # Get action distribution
        action_dist = self.get_action_distribution(observations)
        
        # Sample or evaluate action
        if action is None:
            action = action_dist.sample()
        
        # Calculate log probability and entropy
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1)
        
        # Get value estimate
        value = self.get_value(observations) if self.value_head is not None else torch.zeros_like(log_prob)
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for compatibility with existing training code.
        
        Returns:
            Tuple of (action, value)
        """
        action, _, _, value = self.get_action_and_value(observations)
        return action, value


class GR00TRLPolicyLite(nn.Module):
    """
    Lightweight version of GR00T policy that doesn't require the full model.
    This is useful for testing the integration without loading the 3B parameter model.
    """
    
    def __init__(
        self,
        observation_space,
        action_dim: int,
        hidden_dims: Tuple[int, int] = (256, 256),
        device: str = "cuda"
    ):
        """Initialize lightweight policy for testing."""
        super().__init__()
        
        self.device = device
        self.action_dim = action_dim
        
        # Calculate input dimension
        if hasattr(observation_space, 'shape'):
            input_dim = observation_space.shape[0]
        else:
            input_dim = observation_space.n
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim * 2)  # Mean and log_std
        ).to(device)
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        ).to(device)
    
    def get_action_and_value(self, obs, action=None):
        """Get action and value, optionally evaluating a given action."""
        # Get actor output
        actor_out = self.actor(obs)
        action_mean = actor_out[:, :self.action_dim]
        action_log_std = actor_out[:, self.action_dim:]
        action_std = torch.exp(action_log_std.clamp(-5, 2))
        
        # Create distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Sample or evaluate action
        if action is None:
            action = dist.sample()
        
        # Get log prob and entropy
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Get value
        value = self.critic(obs).squeeze(-1)
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs):
        """Get value estimate."""
        return self.critic(obs).squeeze(-1)