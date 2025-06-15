#!/usr/bin/env python3
"""
Improved PPO implementation for GR00T N1.5 model.
Incorporates all 37 implementation details from CleanRL and ICLR blog post.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from utils.networks import GaussianActor, Critic, MultiModalEncoder, layer_init


class PPOGr00tActorCriticV2(nn.Module):
    """
    Improved Actor-Critic for GR00T with proper continuous action handling.
    Uses Gaussian policy as a temporary solution until we understand GR00T's flow matching.
    """
    
    def __init__(
        self,
        observation_space: Dict[str, Any],
        action_dim: int,
        # Network architecture
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "tanh",
        use_layer_norm: bool = False,
        # Policy settings
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        # GR00T integration (future)
        use_gr00t_backbone: bool = False,
        gr00t_model_path: Optional[str] = None,
        freeze_gr00t_backbone: bool = True,
        # Multi-modal settings
        use_multimodal_encoder: bool = True,
        vision_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        language_dim: Optional[int] = None,
        encoder_output_dim: int = 256,
        # Device
        device: str = "cuda"
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_dim = action_dim
        self.device = device
        self.use_gr00t_backbone = use_gr00t_backbone
        self.use_multimodal_encoder = use_multimodal_encoder
        
        # Get activation function
        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU
        }[activation.lower()]
        
        # Determine input dimension
        if use_multimodal_encoder:
            # Multi-modal encoder
            self.encoder = MultiModalEncoder(
                vision_dim=vision_dim,
                proprio_dim=proprio_dim,
                language_dim=language_dim,
                output_dim=encoder_output_dim,
                use_vision_encoder=True  # Use CNN for raw images
            )
            actor_input_dim = encoder_output_dim
            critic_input_dim = encoder_output_dim
        else:
            # Simple flattened observation
            if isinstance(observation_space, dict):
                # Assume we'll concatenate all observations
                total_dim = sum(
                    np.prod(space.shape) for space in observation_space.values()
                )
                actor_input_dim = total_dim
                critic_input_dim = total_dim
            else:
                actor_input_dim = np.prod(observation_space.shape)
                critic_input_dim = actor_input_dim
                
        # GR00T backbone integration (future work)
        if use_gr00t_backbone and gr00t_model_path:
            # TODO: Integrate GR00T backbone when we understand its architecture
            # For now, we'll use our own encoder
            pass
            
        # Create actor (Gaussian policy)
        self.actor = GaussianActor(
            input_dim=actor_input_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation_fn,
            log_std_init=log_std_init,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            use_layer_norm=use_layer_norm
        )
        
        # Create critic (value function)
        self.critic = Critic(
            input_dim=critic_input_dim,
            hidden_dims=hidden_dims,
            activation=activation_fn,
            use_layer_norm=use_layer_norm
        )
        
    def _process_observation(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Process observation through encoder if needed."""
        if self.use_multimodal_encoder and isinstance(obs, dict):
            return self.encoder(obs)
        elif isinstance(obs, dict):
            # Concatenate all observations
            obs_list = []
            for key in sorted(obs.keys()):
                if isinstance(obs[key], torch.Tensor):
                    obs_list.append(obs[key].flatten(start_dim=1))
            return torch.cat(obs_list, dim=-1)
        else:
            return obs.flatten(start_dim=1)
    
    def get_value(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Get value estimate."""
        features = self._process_observation(obs)
        return self.critic(features)
    
    def get_action_and_value(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        If action is provided, evaluate it. Otherwise, sample new action.
        """
        features = self._process_observation(obs)
        
        # Get value estimate
        value = self.critic(features)
        
        if action is None:
            # Sample new action
            action, log_prob, entropy = self.actor.get_action(features, deterministic)
        else:
            # Evaluate given action
            log_prob, entropy = self.actor.evaluate_actions(features, action)
            
        return action, log_prob, entropy, value
    
    def get_action(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        features = self._process_observation(obs)
        return self.actor.get_action(features, deterministic)
    
    def evaluate_actions(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given actions."""
        features = self._process_observation(obs)
        
        # Get log probabilities and entropy
        log_probs, entropy = self.actor.evaluate_actions(features, actions)
        
        # Get values
        values = self.critic(features)
        
        return log_probs, values, entropy
    
    def save(self, path: str):
        """Save model state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'encoder_state_dict': self.encoder.state_dict() if hasattr(self, 'encoder') else None,
        }, path)
        
    def load(self, path: str, strict: bool = True):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'], strict=strict)
        self.critic.load_state_dict(checkpoint['critic_state_dict'], strict=strict)
        if hasattr(self, 'encoder') and checkpoint.get('encoder_state_dict') is not None:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=strict)