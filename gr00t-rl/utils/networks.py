#!/usr/bin/env python3
"""
Neural network utilities for PPO implementation.
Includes proper initialization and network architectures.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional, Dict, Any


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a layer with orthogonal initialization.
    
    Args:
        layer: nn.Module layer to initialize
        std: Standard deviation for initialization
        bias_const: Constant for bias initialization
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GaussianActor(nn.Module):
    """
    Gaussian policy for continuous action spaces.
    Outputs mean and log_std for action distribution.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: nn.Module = nn.Tanh,
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        use_layer_norm: bool = False
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(layer_init(nn.Linear(prev_dim, hidden_dim)))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
            
        self.trunk = nn.Sequential(*layers)
        
        # Output layer for mean
        self.mean_layer = layer_init(nn.Linear(prev_dim, action_dim), std=0.01)
        
        # State-independent log_std (learned parameter, not network output)
        # This is one of the 37 implementation details
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean and log_std.
        
        Args:
            obs: Observation tensor
            
        Returns:
            mean: Mean of action distribution
            log_std: Log standard deviation of action distribution
        """
        features = self.trunk(obs)
        mean = self.mean_layer(features)
        
        # Expand log_std to match batch size
        batch_size = obs.shape[0]
        log_std = self.log_std.expand(batch_size, -1)
        
        # Clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action(
        self, 
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.
        
        Args:
            obs: Observation
            deterministic: If True, return mean action
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            entropy: Entropy of distribution
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.rsample()  # Reparameterization trick
            
        # Compute log probability
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Ensure action is in valid range (assumes [-1, 1])
        action = torch.tanh(action)
        
        # Correct log_prob for tanh squashing
        log_prob -= (2 * (np.log(2) - action - torch.nn.functional.softplus(-2 * action))).sum(dim=-1)
        
        return action, log_prob, entropy
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            
        Returns:
            log_prob: Log probabilities
            entropy: Entropy of distribution
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Create distribution
        dist = Normal(mean, std)
        
        # Inverse tanh to get pre-squashed actions
        # Clamp to avoid numerical issues
        actions = torch.clamp(actions, -0.999999, 0.999999)
        pre_tanh_actions = 0.5 * (actions.log1p() - (-actions).log1p())
        
        # Compute log probability
        log_prob = dist.log_prob(pre_tanh_actions).sum(dim=-1)
        
        # Correct for tanh squashing
        log_prob -= (2 * (np.log(2) - pre_tanh_actions - torch.nn.functional.softplus(-2 * pre_tanh_actions))).sum(dim=-1)
        
        # Compute entropy
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class Critic(nn.Module):
    """
    Value function network.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: nn.Module = nn.Tanh,
        use_layer_norm: bool = False
    ):
        super().__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(layer_init(nn.Linear(prev_dim, hidden_dim)))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
            
        layers.append(layer_init(nn.Linear(prev_dim, 1), std=1.0))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning value estimate.
        
        Args:
            obs: Observation tensor
            
        Returns:
            value: Value estimate
        """
        return self.net(obs)


class MultiModalEncoder(nn.Module):
    """
    Encoder for multi-modal observations (vision + proprioception + language).
    """
    
    def __init__(
        self,
        vision_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        language_dim: Optional[int] = None,
        output_dim: int = 256,
        use_vision_encoder: bool = True
    ):
        super().__init__()
        
        self.use_vision = vision_dim is not None
        self.use_proprio = proprio_dim is not None
        self.use_language = language_dim is not None
        
        encoders = []
        total_dim = 0
        
        # Vision encoder (CNN)
        if self.use_vision and use_vision_encoder:
            self.vision_encoder = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),  # Assuming 84x84 input
                nn.ReLU()
            )
            total_dim += 512
        elif self.use_vision:
            # If vision features are already extracted
            total_dim += vision_dim
            
        # Proprioception encoder
        if self.use_proprio:
            self.proprio_encoder = nn.Sequential(
                layer_init(nn.Linear(proprio_dim, 128)),
                nn.ReLU(),
                layer_init(nn.Linear(128, 128)),
                nn.ReLU()
            )
            total_dim += 128
            
        # Language encoder (simple MLP, assumes pre-embedded)
        if self.use_language:
            self.language_encoder = nn.Sequential(
                layer_init(nn.Linear(language_dim, 128)),
                nn.ReLU(),
                layer_init(nn.Linear(128, 128)),
                nn.ReLU()
            )
            total_dim += 128
            
        # Fusion layer
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(total_dim, output_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(output_dim, output_dim)),
            nn.ReLU()
        )
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass encoding multi-modal observations.
        
        Args:
            obs: Dictionary of observations
            
        Returns:
            features: Encoded features
        """
        features = []
        
        if self.use_vision:
            if hasattr(self, 'vision_encoder'):
                vision_features = self.vision_encoder(obs['vision'])
            else:
                vision_features = obs['vision']
            features.append(vision_features)
            
        if self.use_proprio:
            proprio_features = self.proprio_encoder(obs['proprioception'])
            features.append(proprio_features)
            
        if self.use_language:
            language_features = self.language_encoder(obs['language'])
            features.append(language_features)
            
        # Concatenate and fuse
        combined = torch.cat(features, dim=-1)
        return self.fusion(combined)