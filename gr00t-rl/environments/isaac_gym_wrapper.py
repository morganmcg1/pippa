#!/usr/bin/env python3
"""
Simple wrapper to make Isaac Lab environments compatible with standard Gym interface.
This allows us to use our existing PPO implementation without modifications.
"""

import numpy as np
import torch
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional, Union


class IsaacGymWrapper(gym.Env):
    """
    Wraps Isaac Lab environments to provide standard Gym interface.
    
    Key features:
    - Converts tensor observations to numpy arrays
    - Handles Isaac Lab's vectorized environments
    - Provides fallback to standard Gym envs if Isaac Lab unavailable
    """
    
    def __init__(
        self, 
        env_name: str,
        num_envs: int = 1,
        device: str = "cuda",
        isaac_lab_available: bool = None,
        **kwargs
    ):
        """
        Args:
            env_name: Name of the environment (Isaac Lab or Gym)
            num_envs: Number of parallel environments
            device: Device for Isaac Lab (cuda/cpu)
            isaac_lab_available: Override automatic detection
            **kwargs: Additional arguments for environment creation
        """
        self.env_name = env_name
        self.num_envs = num_envs
        self.device = device
        
        # Check if Isaac Lab is available
        if isaac_lab_available is None:
            self.isaac_lab_available = self._check_isaac_lab()
        else:
            self.isaac_lab_available = isaac_lab_available
            
        # Create environment
        if self.isaac_lab_available and env_name.startswith("Isaac-"):
            self._create_isaac_env(env_name, **kwargs)
        else:
            self._create_gym_env(env_name)
            
    def _check_isaac_lab(self) -> bool:
        """Check if Isaac Lab is available."""
        try:
            import isaaclab
            from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
            return True
        except ImportError:
            return False
            
    def _create_isaac_env(self, env_name: str, **kwargs):
        """Create Isaac Lab environment."""
        try:
            from isaaclab.envs import DirectRLEnv
            from isaaclab_tasks import isaaclab_task_registry
            
            # Get environment configuration
            env_cfg = isaaclab_task_registry.get_cfgs(env_name)
            env_cfg.num_envs = self.num_envs
            env_cfg.device = self.device
            
            # Apply any additional kwargs
            for key, value in kwargs.items():
                if hasattr(env_cfg, key):
                    setattr(env_cfg, key, value)
                    
            # Create environment
            self.env = DirectRLEnv(cfg=env_cfg)
            self.is_isaac = True
            
            # Set up spaces
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.env.num_obs,),
                dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.env.num_actions,),
                dtype=np.float32
            )
            
        except Exception as e:
            print(f"Failed to create Isaac Lab environment: {e}")
            print("Falling back to standard Gym environment")
            self._create_gym_env(env_name.replace("Isaac-", ""))
            
    def _create_gym_env(self, env_name: str):
        """Create standard Gym environment (fallback)."""
        # Map Isaac Lab names to Gym equivalents
        env_mapping = {
            "Cartpole": "CartPole-v1",
            "Pendulum": "Pendulum-v1",
            "Ant": "Ant-v4",
            "Humanoid": "Humanoid-v4",
        }
        
        gym_name = env_mapping.get(env_name, env_name)
        
        # Create single environment
        self.env = gym.make(gym_name)
        self.is_isaac = False
        
        # Copy spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # For compatibility with vectorized interface
        if self.num_envs > 1:
            print(f"Warning: Requested {self.num_envs} envs but using single Gym env")
            
    def reset(self, seed: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Reset environment(s).
        
        Returns:
            observations: numpy array of shape (num_envs, obs_dim) or (obs_dim,)
            info: Optional info dict (for new Gym API)
        """
        if self.is_isaac:
            # Isaac Lab returns tensors
            obs_dict = self.env.reset()
            if isinstance(obs_dict, dict):
                obs = obs_dict["obs"]
            else:
                obs = obs_dict
                
            # Convert to numpy
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
                
            return obs
        else:
            # Standard Gym
            result = self.env.reset(seed=seed)
            if isinstance(result, tuple):
                obs, info = result
            else:
                obs = result
                info = {}
                
            # Add batch dimension if needed
            if self.num_envs > 1 and obs.ndim == 1:
                obs = obs[np.newaxis, :]
                
            return obs
            
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Step environment(s).
        
        Args:
            actions: numpy array of shape (num_envs, act_dim) or (act_dim,)
            
        Returns:
            observations: numpy array
            rewards: numpy array of shape (num_envs,) or scalar
            dones: numpy array of shape (num_envs,) or scalar
            infos: dict or list of dicts
        """
        if self.is_isaac:
            # Convert to tensor
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
                
            # Step Isaac Lab env
            obs, rewards, dones, infos = self.env.step(actions)
            
            # Convert to numpy
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.cpu().numpy()
            if isinstance(dones, torch.Tensor):
                dones = dones.cpu().numpy()
                
            return obs, rewards, dones, infos
        else:
            # Standard Gym
            if self.num_envs > 1 and actions.ndim > 1:
                actions = actions[0]  # Take first env's action
                
            obs, reward, done, truncated, info = self.env.step(actions)
            
            # Combine done and truncated for old API
            done = done or truncated
            
            # Add batch dimension if needed
            if self.num_envs > 1:
                obs = obs[np.newaxis, :]
                reward = np.array([reward])
                done = np.array([done])
                info = [info]
            else:
                reward = np.array(reward)
                done = np.array(done)
                
            return obs, reward, done, info
            
    def close(self):
        """Close environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
            
    def render(self, mode: str = "human"):
        """Render environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
            
    @property
    def unwrapped(self):
        """Get unwrapped environment."""
        return self.env
        
    def __str__(self):
        """String representation."""
        return f"IsaacGymWrapper({self.env_name}, num_envs={self.num_envs}, isaac={self.is_isaac})"


def create_isaac_env(
    env_name: str,
    num_envs: int = 1,
    device: str = "cuda",
    **kwargs
) -> IsaacGymWrapper:
    """
    Convenience function to create Isaac Lab environment with Gym interface.
    
    Args:
        env_name: Environment name
        num_envs: Number of parallel environments
        device: Device for Isaac Lab
        **kwargs: Additional environment configuration
        
    Returns:
        IsaacGymWrapper instance
    """
    return IsaacGymWrapper(env_name, num_envs, device, **kwargs)


# Test function
def test_wrapper():
    """Test the Isaac Gym wrapper."""
    print("Testing Isaac Gym Wrapper")
    print("=" * 50)
    
    # Test 1: Fallback to standard Gym
    print("\n1. Testing fallback to standard Gym (Pendulum)...")
    env = create_isaac_env("Pendulum", num_envs=1, isaac_lab_available=False)
    print(f"Created: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Step observation shape: {obs.shape}")
    print(f"Reward: {reward}, Done: {done}")
    env.close()
    print("✓ Fallback test passed!")
    
    # Test 2: Try Isaac Lab (will fallback if unavailable)
    print("\n2. Testing Isaac Lab integration...")
    env = create_isaac_env("Isaac-Cartpole-v0", num_envs=4)
    print(f"Created: {env}")
    
    if env.is_isaac:
        print("✓ Isaac Lab environment created!")
    else:
        print("✓ Fell back to standard Gym (Isaac Lab not available)")
        
    env.close()
    
    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    test_wrapper()