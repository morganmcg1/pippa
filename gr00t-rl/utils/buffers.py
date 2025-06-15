#!/usr/bin/env python3
"""
Improved buffer implementations for PPO with proper GAE and bootstrapping.
"""

import torch
import numpy as np
from typing import Dict, Union, Optional, Any


class PPORolloutBuffer:
    """
    Rollout buffer for PPO with proper GAE implementation.
    Handles variable episode lengths and proper bootstrapping.
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: Any,
        action_space: Any,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_envs: int = 1
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_envs = n_envs
        
        # Determine dimensions
        if hasattr(action_space, 'shape'):
            self.action_dim = action_space.shape[0]
        else:
            self.action_dim = 1
            
        # Storage
        self.reset()
        
    def reset(self):
        """Reset the buffer."""
        # Core data
        self.observations = {}
        self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), device=self.device)
        self.rewards = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.returns = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.values = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.log_probs = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.advantages = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        self.dones = torch.zeros((self.buffer_size, self.n_envs), device=self.device)
        
        # For proper episode handling
        self.episode_starts = torch.ones((self.n_envs,), dtype=torch.bool, device=self.device)
        
        self.pos = 0
        self.full = False
        
    def add(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor
    ):
        """Add a transition to the buffer."""
        if self.pos == 0:
            # Initialize observation storage on first add
            if isinstance(obs, dict):
                for key, val in obs.items():
                    obs_shape = (self.buffer_size, self.n_envs) + val.shape[1:]
                    self.observations[key] = torch.zeros(obs_shape, device=self.device)
            else:
                obs_shape = (self.buffer_size, self.n_envs) + obs.shape[1:]
                self.observations = torch.zeros(obs_shape, device=self.device)
                
        # Store data
        if isinstance(obs, dict):
            for key, val in obs.items():
                self.observations[key][self.pos] = val.to(self.device)
        else:
            self.observations[self.pos] = obs.to(self.device)
            
        self.actions[self.pos] = action.to(self.device)
        self.rewards[self.pos] = reward.to(self.device)
        self.dones[self.pos] = done.to(self.device)
        self.values[self.pos] = value.to(self.device).flatten()
        self.log_probs[self.pos] = log_prob.to(self.device)
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            
    def compute_returns_and_advantages(
        self, 
        last_values: torch.Tensor,
        dones: torch.Tensor
    ):
        """
        Compute returns and advantages using GAE.
        Properly handles episode boundaries and bootstrapping.
        
        Args:
            last_values: Value estimates for last observation
            dones: Whether episodes are done at last step
        """
        # Convert to correct device
        last_values = last_values.to(self.device).flatten()
        dones = dones.to(self.device)
        
        # GAE computation
        last_gae = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
                
            # TD error
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            
            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
            
        # Compute returns
        self.returns = self.advantages + self.values
        
    def get(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Get all data from buffer with proper reshaping for training.
        
        Args:
            batch_size: If provided, will yield batches of this size
            
        Returns:
            Dictionary of training data
        """
        assert self.full or self.pos > 0, "No data in buffer"
        
        # Determine actual buffer size
        buffer_len = self.buffer_size if self.full else self.pos
        
        # Flatten all data for training
        def _flatten(tensor):
            return tensor[:buffer_len].reshape(-1, *tensor.shape[2:])
            
        # Prepare data dictionary
        data = {
            'actions': _flatten(self.actions),
            'values': _flatten(self.values),
            'log_probs': _flatten(self.log_probs),
            'advantages': _flatten(self.advantages),
            'returns': _flatten(self.returns),
        }
        
        # Handle observations
        if isinstance(self.observations, dict):
            data['observations'] = {
                key: _flatten(val) for key, val in self.observations.items()
            }
        else:
            data['observations'] = _flatten(self.observations)
            
        # Normalize advantages at the batch level (not mini-batch)
        # This is one of the 37 implementation details
        advantages_mean = data['advantages'].mean()
        advantages_std = data['advantages'].std() + 1e-8
        data['advantages'] = (data['advantages'] - advantages_mean) / advantages_std
        
        # If batch_size is provided, yield batches
        if batch_size is not None:
            num_samples = len(data['advantages'])
            indices = np.arange(num_samples)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch = {}
                for key, value in data.items():
                    if isinstance(value, dict):
                        batch[key] = {
                            k: v[batch_indices] for k, v in value.items()
                        }
                    else:
                        batch[key] = value[batch_indices]
                        
                yield batch
        else:
            return data