#!/usr/bin/env python3
"""
Wrapper for Gymnasium-Robotics Fetch environments to work with our PPO/GRPO implementations.
Handles goal-conditioned observations and provides different reward modes.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Union


class FetchGoalWrapper(gym.Wrapper):
    """
    Wrapper for Fetch environments to work with standard RL algorithms.
    
    Features:
    - Converts dictionary observations to flat tensors
    - Supports different observation modes (with/without goals)
    - Provides sparse and dense reward options
    - Compatible with HER (Hindsight Experience Replay)
    """
    
    def __init__(
        self,
        env: gym.Env,
        observation_mode: str = "observation_goal",  # Options: "observation", "observation_goal", "full"
        reward_mode: str = "sparse",  # Options: "sparse", "dense", "distance"
        goal_in_observation: bool = True,
        normalize_observations: bool = False,
        normalize_goals: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize the Fetch wrapper.
        
        Args:
            env: The Fetch environment to wrap
            observation_mode: How to structure observations
                - "observation": Only robot state
                - "observation_goal": Robot state + desired goal
                - "full": Robot state + desired goal + achieved goal
            reward_mode: Type of reward to use
                - "sparse": Original sparse reward (-1/0)
                - "dense": Negative distance to goal
                - "distance": Distance-based with success bonus
            goal_in_observation: Whether to include goal in observation
            normalize_observations: Whether to normalize observations
            normalize_goals: Whether to normalize goal positions
            device: Device for tensor operations
        """
        super().__init__(env)
        
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.goal_in_observation = goal_in_observation
        self.normalize_observations = normalize_observations
        self.normalize_goals = normalize_goals
        self.device = device
        
        # Get original spaces
        self._observation_space = env.observation_space['observation']
        self._goal_space = env.observation_space['desired_goal']
        self._achieved_goal_space = env.observation_space['achieved_goal']
        
        # Compute new observation space based on mode
        obs_dim = self._observation_space.shape[0]
        goal_dim = self._goal_space.shape[0]
        
        if observation_mode == "observation":
            total_dim = obs_dim
        elif observation_mode == "observation_goal":
            total_dim = obs_dim + goal_dim
        elif observation_mode == "full":
            total_dim = obs_dim + goal_dim + goal_dim  # achieved_goal has same dim as desired_goal
        else:
            raise ValueError(f"Unknown observation_mode: {observation_mode}")
        
        # Create new observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
        
        # Store normalization statistics
        self.obs_mean = np.zeros(obs_dim)
        self.obs_std = np.ones(obs_dim)
        self.goal_mean = np.zeros(goal_dim)
        self.goal_std = np.ones(goal_dim)
        
        # For HER compatibility
        self.last_observation = None
        self.desired_goal = None
        
    def _process_observation(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert dictionary observation to flat array based on observation mode."""
        observation = obs_dict['observation']
        desired_goal = obs_dict['desired_goal']
        achieved_goal = obs_dict['achieved_goal']
        
        # Store for HER
        self.last_observation = observation.copy()
        self.desired_goal = desired_goal.copy()
        
        # Normalize if requested
        if self.normalize_observations:
            observation = (observation - self.obs_mean) / (self.obs_std + 1e-8)
        
        if self.normalize_goals:
            desired_goal = (desired_goal - self.goal_mean) / (self.goal_std + 1e-8)
            achieved_goal = (achieved_goal - self.goal_mean) / (self.goal_std + 1e-8)
        
        # Concatenate based on mode
        if self.observation_mode == "observation":
            return observation
        elif self.observation_mode == "observation_goal":
            return np.concatenate([observation, desired_goal])
        elif self.observation_mode == "full":
            return np.concatenate([observation, desired_goal, achieved_goal])
    
    def _compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict
    ) -> float:
        """Compute reward based on reward mode."""
        # Use environment's compute_reward for sparse reward
        sparse_reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        
        if self.reward_mode == "sparse":
            return sparse_reward
        
        # Compute distance for dense rewards
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        if self.reward_mode == "dense":
            # Negative distance
            return -distance
        
        elif self.reward_mode == "distance":
            # Distance with success bonus
            if sparse_reward == 0:  # Success
                return 10.0 - distance  # Bonus for getting close
            else:
                return -distance
        
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        obs_dict, info = self.env.reset(**kwargs)
        obs = self._process_observation(obs_dict)
        
        # Add goal info to info dict for logging
        info['desired_goal'] = obs_dict['desired_goal'].copy()
        info['achieved_goal'] = obs_dict['achieved_goal'].copy()
        info['initial_distance'] = np.linalg.norm(
            obs_dict['achieved_goal'] - obs_dict['desired_goal']
        )
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        
        # Process observation
        obs = self._process_observation(obs_dict)
        
        # Compute custom reward if needed
        if self.reward_mode != "sparse":
            reward = self._compute_reward(
                obs_dict['achieved_goal'],
                obs_dict['desired_goal'],
                info
            )
        
        # Add extra info for logging
        info['achieved_goal'] = obs_dict['achieved_goal'].copy()
        info['desired_goal'] = obs_dict['desired_goal'].copy()
        info['distance_to_goal'] = np.linalg.norm(
            obs_dict['achieved_goal'] - obs_dict['desired_goal']
        )
        info['sparse_reward'] = self.env.compute_reward(
            obs_dict['achieved_goal'],
            obs_dict['desired_goal'],
            info
        )
        
        return obs, reward, terminated, truncated, info
    
    def update_normalization_stats(
        self,
        observations: np.ndarray,
        goals: Optional[np.ndarray] = None
    ):
        """Update normalization statistics (for training)."""
        if self.normalize_observations and observations is not None:
            self.obs_mean = observations.mean(axis=0)
            self.obs_std = observations.std(axis=0)
        
        if self.normalize_goals and goals is not None:
            self.goal_mean = goals.mean(axis=0)
            self.goal_std = goals.std(axis=0)
    
    def get_achieved_goal(self) -> np.ndarray:
        """Get the current achieved goal (for HER)."""
        # Need to get from underlying environment
        obs_dict = self.env.unwrapped._get_obs()
        return obs_dict['achieved_goal']
    
    def compute_reward_from_goal(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray
    ) -> float:
        """Compute reward for arbitrary goal (for HER)."""
        info = {}  # Empty info for compute_reward
        return self._compute_reward(achieved_goal, desired_goal, info)


class FetchVecWrapper:
    """
    Vectorized version of FetchGoalWrapper for parallel environments.
    """
    
    def __init__(
        self,
        venv,
        observation_mode: str = "observation_goal",
        reward_mode: str = "sparse",
        **kwargs
    ):
        """Initialize vectorized wrapper."""
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        
        # Wrap each environment
        for i in range(self.num_envs):
            self.venv.envs[i] = FetchGoalWrapper(
                self.venv.envs[i],
                observation_mode=observation_mode,
                reward_mode=reward_mode,
                **kwargs
            )
        
        # Update observation space
        self.observation_space = self.venv.envs[0].observation_space
        self.action_space = self.venv.action_space
    
    def reset(self, **kwargs):
        """Reset all environments."""
        return self.venv.reset(**kwargs)
    
    def step(self, actions):
        """Step all environments."""
        return self.venv.step(actions)
    
    def close(self):
        """Close all environments."""
        self.venv.close()
    
    def __getattr__(self, name):
        """Forward other attributes to the vectorized environment."""
        return getattr(self.venv, name)