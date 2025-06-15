#!/usr/bin/env python3
"""
PPO wrapper for GR00T N1.5 model.
Adds a critic head and provides actor-critic interface for PPO training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

# Assuming Isaac-GR00T is installed
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
from gr00t.data.embodiment_tags import EmbodimentTag


class CriticHead(nn.Module):
    """Lightweight critic network for value estimation."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Backbone features [batch_size, feature_dim]
        Returns:
            value: Estimated value [batch_size, 1]
        """
        return self.net(features)


class PPOGr00tActorCritic(nn.Module):
    """
    Actor-Critic wrapper for GR00T model.
    Provides both action predictions and value estimates for PPO.
    """
    
    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "GR1",
        critic_hidden_dim: int = 512,
        freeze_backbone: bool = True,
        freeze_action_head: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        
        # Load GR00T policy
        self.gr00t_policy = Gr00tPolicy.from_pretrained(
            model_path,
            embodiment_tag=embodiment_tag,
            device=device
        )
        
        # Access the underlying model
        self.gr00t_model = self.gr00t_policy.model
        
        # Get backbone output dimension
        # This needs to be determined from the model config
        self.backbone_dim = self._get_backbone_dim()
        
        # Create critic head
        self.critic = CriticHead(self.backbone_dim, critic_hidden_dim)
        
        # Freeze settings
        if freeze_backbone:
            self._freeze_backbone()
        if freeze_action_head:
            self._freeze_action_head()
            
        self.device = device
        
    def _get_backbone_dim(self) -> int:
        """Get the dimension of backbone features."""
        # This is model-specific, GR00T N1.5 typically outputs 2048-dim features
        # You may need to adjust based on actual model
        return 2048
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.gr00t_model.backbone.parameters():
            param.requires_grad = False
            
    def _freeze_action_head(self):
        """Freeze action head parameters."""
        for param in self.gr00t_model.action_head.parameters():
            param.requires_grad = False
    
    def get_backbone_features(self, observations: Dict[str, Any]) -> torch.Tensor:
        """
        Extract backbone features from observations.
        
        Args:
            observations: Dict containing multimodal inputs
        Returns:
            features: Backbone features [batch_size, feature_dim]
        """
        # Process through GR00T backbone
        # This needs to match GR00T's expected input format
        inputs = self.gr00t_policy._prepare_inputs(observations)
        
        # Get backbone output
        with torch.no_grad() if self.gr00t_model.backbone.training == False else torch.enable_grad():
            backbone_output = self.gr00t_model.backbone(**inputs)
            
        # Extract features (typically from the last hidden state)
        features = backbone_output["backbone_features"]
        
        # Pool or select features if needed
        if len(features.shape) > 2:
            # If we have sequence dimension, pool it
            features = features.mean(dim=1)
            
        return features
    
    def get_action(
        self, 
        observations: Dict[str, Any],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from the policy.
        
        Args:
            observations: Environment observations
            deterministic: If True, return mean action
            
        Returns:
            action: Sampled or deterministic action
            action_logprob: Log probability of the action
            value: Estimated value of the state
        """
        # Get backbone features
        features = self.get_backbone_features(observations)
        
        # Get value estimate
        value = self.critic(features)
        
        # Get action from GR00T
        # The action head in GR00T uses flow matching, which is stochastic
        action_output = self.gr00t_policy.get_action(observations)
        
        # Extract action tensor
        action = action_output["action"]
        
        # For log probability, we need to compute it based on the flow matching distribution
        # This is a simplification - actual implementation would need the flow matching likelihood
        action_logprob = self._compute_action_logprob(action, observations)
        
        if deterministic:
            # For deterministic action, we might want to use more denoising steps
            # or return the mean of the distribution
            action = self._get_deterministic_action(observations)
            
        return action, action_logprob, value
    
    def get_value(self, observations: Dict[str, Any]) -> torch.Tensor:
        """
        Get value estimate for observations.
        
        Args:
            observations: Environment observations
        Returns:
            value: Estimated value [batch_size, 1]
        """
        features = self.get_backbone_features(observations)
        return self.critic(features)
    
    def evaluate_actions(
        self,
        observations: Dict[str, Any],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions under current policy.
        
        Args:
            observations: Environment observations
            actions: Actions to evaluate
            
        Returns:
            action_logprobs: Log probabilities of actions
            values: Value estimates
            entropy: Entropy of action distribution
        """
        # Get backbone features
        features = self.get_backbone_features(observations)
        
        # Get value estimate
        values = self.critic(features)
        
        # Compute log probabilities and entropy
        # This is simplified - actual implementation needs proper flow matching likelihood
        action_logprobs = self._compute_action_logprob(actions, observations)
        entropy = self._compute_entropy(observations)
        
        return action_logprobs, values, entropy
    
    def _compute_action_logprob(
        self, 
        actions: torch.Tensor, 
        observations: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute log probability of actions.
        Note: This is a placeholder - actual implementation needs flow matching likelihood.
        """
        # For now, return a dummy value
        # In practice, this would compute the likelihood under the flow matching distribution
        return torch.zeros(actions.shape[0], device=self.device)
    
    def _compute_entropy(self, observations: Dict[str, Any]) -> torch.Tensor:
        """
        Compute entropy of action distribution.
        Note: This is a placeholder - actual implementation needs flow matching entropy.
        """
        # For now, return a dummy value
        batch_size = next(iter(observations.values())).shape[0]
        return torch.ones(batch_size, device=self.device) * 0.1
    
    def _get_deterministic_action(self, observations: Dict[str, Any]) -> torch.Tensor:
        """
        Get deterministic action (e.g., mean of distribution).
        """
        # Use more denoising steps for deterministic action
        original_steps = self.gr00t_policy.denoising_steps
        self.gr00t_policy.denoising_steps = original_steps * 2
        
        action_output = self.gr00t_policy.get_action(observations)
        action = action_output["action"]
        
        # Restore original steps
        self.gr00t_policy.denoising_steps = original_steps
        
        return action
    
    def save(self, path: str):
        """Save the model state."""
        torch.save({
            'critic_state_dict': self.critic.state_dict(),
            'gr00t_state_dict': self.gr00t_model.state_dict(),
        }, path)
        
    def load(self, path: str):
        """Load the model state."""
        checkpoint = torch.load(path)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.gr00t_model.load_state_dict(checkpoint['gr00t_state_dict'])