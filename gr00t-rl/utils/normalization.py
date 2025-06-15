#!/usr/bin/env python3
"""
Normalization utilities for observations and rewards.
Includes running mean/std computation and normalization wrappers.
"""

import numpy as np
import torch
from typing import Union, Dict, Any, Optional
import gymnasium as gym


class RunningMeanStd:
    """
    Tracks running mean and standard deviation of data.
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(
        self, 
        shape: tuple = (), 
        epsilon: float = 1e-8,
        device: str = "cpu"
    ):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)
        self.epsilon = epsilon
        self.device = device
        
    def update(self, x: torch.Tensor):
        """Update running statistics with new data."""
        x = x.to(self.device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def update_from_moments(
        self, 
        batch_mean: torch.Tensor, 
        batch_var: torch.Tensor, 
        batch_count: int
    ):
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        
        self.var = M2 / total_count
        self.count = total_count
        
    def normalize(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Normalize input data."""
        if update and self.training:
            self.update(x)
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize data."""
        return x * torch.sqrt(self.var + self.epsilon) + self.mean
    
    @property
    def training(self):
        """Check if in training mode."""
        return getattr(self, '_training', True)
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self._training = mode
        
    def eval(self):
        """Set evaluation mode."""
        self._training = False


class ObservationNormalizer(gym.ObservationWrapper):
    """
    Gym wrapper that normalizes observations using running statistics.
    """
    
    def __init__(
        self, 
        env: gym.Env,
        epsilon: float = 1e-8,
        clip_obs: float = 10.0,
        device: str = "cpu"
    ):
        super().__init__(env)
        
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        self.device = device
        
        # Initialize running stats
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_rms = {}
            for key, space in self.observation_space.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    self.obs_rms[key] = RunningMeanStd(
                        shape=space.shape,
                        epsilon=epsilon,
                        device=device
                    )
        else:
            self.obs_rms = RunningMeanStd(
                shape=self.observation_space.shape,
                epsilon=epsilon,
                device=device
            )
            
    def observation(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Any:
        """Normalize observation."""
        if isinstance(obs, dict):
            normalized_obs = {}
            for key, val in obs.items():
                if key in self.obs_rms:
                    val_tensor = torch.from_numpy(val).float().to(self.device)
                    normalized_val = self.obs_rms[key].normalize(val_tensor.unsqueeze(0))
                    normalized_val = torch.clamp(normalized_val, -self.clip_obs, self.clip_obs)
                    normalized_obs[key] = normalized_val.squeeze(0).cpu().numpy()
                else:
                    normalized_obs[key] = val
            return normalized_obs
        else:
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            normalized_obs = self.obs_rms.normalize(obs_tensor.unsqueeze(0))
            normalized_obs = torch.clamp(normalized_obs, -self.clip_obs, self.clip_obs)
            return normalized_obs.squeeze(0).cpu().numpy()
    
    def train(self, mode: bool = True):
        """Set training mode for running statistics."""
        if isinstance(self.obs_rms, dict):
            for rms in self.obs_rms.values():
                rms.train(mode)
        else:
            self.obs_rms.train(mode)
            
    def eval(self):
        """Set evaluation mode."""
        self.train(False)


class RewardNormalizer(gym.RewardWrapper):
    """
    Gym wrapper that normalizes rewards using running statistics.
    Can optionally scale rewards by a fixed factor.
    """
    
    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        gamma: float = 0.99,
        clip_reward: Optional[float] = None,
        scale_factor: float = 1.0,
        device: str = "cpu"
    ):
        super().__init__(env)
        
        self.epsilon = epsilon
        self.gamma = gamma
        self.clip_reward = clip_reward
        self.scale_factor = scale_factor
        self.device = device
        
        # Running statistics for returns (not rewards)
        self.return_rms = RunningMeanStd(shape=(), epsilon=epsilon, device=device)
        self.returns = None
        
    def reset(self, **kwargs):
        """Reset environment and return tracker."""
        self.returns = 0.0
        return self.env.reset(**kwargs)
    
    def reward(self, reward: float) -> float:
        """Normalize reward based on return statistics."""
        # Update returns
        self.returns = reward + self.gamma * self.returns
        
        # Normalize using return statistics
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        # During training, we normalize by the std of returns
        if self.return_rms.training:
            self.return_rms.update(torch.tensor([[self.returns]], device=self.device))
            
        # Normalize reward by std of returns
        normalized_reward = reward_tensor / torch.sqrt(self.return_rms.var + self.epsilon)
        normalized_reward *= self.scale_factor
        
        # Clip if specified
        if self.clip_reward is not None:
            normalized_reward = torch.clamp(
                normalized_reward, 
                -self.clip_reward, 
                self.clip_reward
            )
            
        return normalized_reward.item()
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.return_rms.train(mode)
        
    def eval(self):
        """Set evaluation mode."""
        self.train(False)


class VecNormalize:
    """
    Vectorized environment normalizer for multiple parallel environments.
    """
    
    def __init__(
        self,
        venv: Any,  # VecEnv type
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        device: str = "cpu"
    ):
        self.venv = venv
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        
        self.num_envs = getattr(venv, 'num_envs', 1)
        self.returns = torch.zeros(self.num_envs, device=device)
        
        # Initialize observation normalizer
        if norm_obs:
            obs_space = venv.observation_space
            if isinstance(obs_space, gym.spaces.Dict):
                self.obs_rms = {}
                for key, space in obs_space.spaces.items():
                    if isinstance(space, gym.spaces.Box):
                        self.obs_rms[key] = RunningMeanStd(
                            shape=space.shape,
                            epsilon=epsilon,
                            device=device
                        )
            else:
                self.obs_rms = RunningMeanStd(
                    shape=obs_space.shape,
                    epsilon=epsilon,
                    device=device
                )
                
        # Initialize reward normalizer
        if norm_reward:
            self.return_rms = RunningMeanStd(shape=(), epsilon=epsilon, device=device)
            
    def step(self, actions):
        """Step through environments with normalization."""
        obs, rewards, dones, infos = self.venv.step(actions)
        
        # Normalize observations
        if self.norm_obs:
            obs = self._normalize_obs(obs)
            
        # Normalize rewards
        if self.norm_reward:
            rewards = self._normalize_reward(rewards, dones)
            
        return obs, rewards, dones, infos
    
    def reset(self, **kwargs):
        """Reset environments."""
        reset_data = self.venv.reset(**kwargs)
        
        # Handle new Gymnasium API that returns (obs, info)
        if isinstance(reset_data, tuple):
            obs, infos = reset_data
            return_info = True
        else:
            obs = reset_data
            infos = None
            return_info = False
            
        self.returns = torch.zeros(self.num_envs, device=self.device)
        
        if self.norm_obs:
            obs = self._normalize_obs(obs)
            
        if return_info:
            return obs, infos
        else:
            return obs
    
    def _normalize_obs(self, obs):
        """Normalize observations."""
        if isinstance(obs, dict):
            normalized_obs = {}
            for key, val in obs.items():
                if key in self.obs_rms:
                    val_tensor = torch.from_numpy(val).float().to(self.device)
                    if self.training:
                        self.obs_rms[key].update(val_tensor)
                    normalized_val = self.obs_rms[key].normalize(val_tensor, update=False)
                    normalized_val = torch.clamp(normalized_val, -self.clip_obs, self.clip_obs)
                    normalized_obs[key] = normalized_val.cpu().numpy()
                else:
                    normalized_obs[key] = val
            return normalized_obs
        else:
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            if self.training:
                self.obs_rms.update(obs_tensor)
            normalized_obs = self.obs_rms.normalize(obs_tensor, update=False)
            normalized_obs = torch.clamp(normalized_obs, -self.clip_obs, self.clip_obs)
            return normalized_obs.cpu().numpy()
    
    def _normalize_reward(self, rewards, dones):
        """Normalize rewards based on return statistics."""
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        
        # Update returns
        self.returns = rewards_tensor + self.gamma * self.returns
        
        # Update return statistics
        if self.training:
            self.return_rms.update(self.returns.unsqueeze(1))
            
        # Normalize rewards
        normalized_rewards = rewards_tensor / torch.sqrt(self.return_rms.var + self.epsilon)
        
        # Clip rewards
        if self.clip_reward is not None:
            normalized_rewards = torch.clamp(
                normalized_rewards,
                -self.clip_reward,
                self.clip_reward
            )
            
        # Reset returns for done environments
        self.returns[dones] = 0.0
        
        return normalized_rewards.cpu().numpy()
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
    
    @property
    def observation_space(self):
        """Return the observation space of the wrapped environment."""
        return self.venv.observation_space
    
    @property
    def action_space(self):
        """Return the action space of the wrapped environment."""
        return self.venv.action_space
    
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped environment."""
        return getattr(self.venv, name)