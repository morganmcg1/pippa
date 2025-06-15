#!/usr/bin/env python3
"""
Vectorized environment wrapper for Isaac Lab / Gymnasium environments.
Provides parallel environment execution for efficient PPO training.
"""

import numpy as np
import torch
import gymnasium as gym
from typing import List, Tuple, Dict, Any, Optional, Union
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import cloudpickle


def worker(
    remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: 'CloudPickledData'
):
    """Worker process for parallel environment execution."""
    parent_remote.close()
    env_fn = env_fn_wrapper.fn
    env = env_fn()
    
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                remote.send((obs, reward, done, info))
                
            elif cmd == 'reset':
                obs, info = env.reset(**data)
                remote.send((obs, info))
                
            elif cmd == 'render':
                remote.send(env.render())
                
            elif cmd == 'close':
                env.close()
                remote.close()
                break
                
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
                
            elif cmd == 'env_method':
                method_name, args, kwargs = data
                method = getattr(env, method_name)
                remote.send(method(*args, **kwargs))
                
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
                
            elif cmd == 'set_attr':
                key, value = data
                setattr(env, key, value)
                remote.send(None)
                
        except EOFError:
            break


class CloudPickledData:
    """Wrapper for cloudpickled data."""
    def __init__(self, fn):
        self.fn = fn


class SubprocVecEnv:
    """
    Vectorized environment that runs multiple environments in parallel processes.
    Based on Stable-Baselines3 implementation.
    """
    
    def __init__(self, env_fns: List[callable], start_method: Optional[str] = None):
        self.num_envs = len(env_fns)
        
        if start_method is None:
            # Use 'spawn' for CUDA compatibility
            start_method = 'spawn'
            
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        
        # Start worker processes
        self.ps = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudPickledData(env_fn))
            process = Process(target=worker, args=args, daemon=True)
            process.start()
            self.ps.append(process)
            work_remote.close()
            
        # Get spaces from first environment
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments in parallel."""
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
            
        results = [remote.recv() for remote in self.remotes]
        
        # Handle both old (4-tuple) and new (5-tuple) Gym API
        if len(results[0]) == 4:
            # Old API: obs, reward, done, info
            obs, rewards, dones, infos = zip(*results)
            # Convert to new API: terminated = done, truncated = False
            terminated = dones
            truncated = [False] * len(dones)
        else:
            # New API: obs, reward, terminated, truncated, info
            obs, rewards, terminated, truncated, infos = zip(*results)
        
        return np.stack(obs), np.stack(rewards), np.stack(terminated), np.stack(truncated), list(infos)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments."""
        for remote in self.remotes:
            remote.send(('reset', kwargs))
            
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        
        return np.stack(obs), list(infos)
    
    def close(self):
        """Close all environments and worker processes."""
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the first environment."""
        self.remotes[0].send(('render', None))
        return self.remotes[0].recv()
    
    def env_method(self, method_name: str, *args, indices: Optional[List[int]] = None, **kwargs):
        """Call a method on specified environments."""
        if indices is None:
            indices = range(self.num_envs)
            
        for i in indices:
            self.remotes[i].send(('env_method', (method_name, args, kwargs)))
            
        return [self.remotes[i].recv() for i in indices]
    
    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None):
        """Get an attribute from specified environments."""
        if indices is None:
            indices = range(self.num_envs)
            
        for i in indices:
            self.remotes[i].send(('get_attr', attr_name))
            
        return [self.remotes[i].recv() for i in indices]
    
    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None):
        """Set an attribute on specified environments."""
        if indices is None:
            indices = range(self.num_envs)
            
        for i in indices:
            self.remotes[i].send(('set_attr', (attr_name, value)))
            
        for i in indices:
            self.remotes[i].recv()


class DummyVecEnv:
    """
    Vectorized environment that runs multiple environments sequentially.
    Useful for debugging or when parallelization is not needed.
    """
    
    def __init__(self, env_fns: List[callable]):
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments sequentially."""
        results = []
        for env, action in zip(self.envs, actions):
            result = env.step(action)
            if len(result) == 4:
                # Old API
                obs, reward, done, info = result
                terminated = done
                truncated = False
            else:
                # New API
                obs, reward, terminated, truncated, info = result
            results.append((obs, reward, terminated, truncated, info))
            
        obs, rewards, terminated, truncated, infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(terminated), np.stack(truncated), list(infos)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments."""
        results = []
        for env in self.envs:
            obs, info = env.reset(**kwargs)
            results.append((obs, info))
            
        obs, infos = zip(*results)
        return np.stack(obs), list(infos)
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()
            
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the first environment."""
        return self.envs[0].render(mode=mode)
    
    def env_method(self, method_name: str, *args, indices: Optional[List[int]] = None, **kwargs):
        """Call a method on specified environments."""
        if indices is None:
            indices = range(self.num_envs)
            
        results = []
        for i in indices:
            method = getattr(self.envs[i], method_name)
            results.append(method(*args, **kwargs))
            
        return results
    
    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None):
        """Get an attribute from specified environments."""
        if indices is None:
            indices = range(self.num_envs)
            
        return [getattr(self.envs[i], attr_name) for i in indices]
    
    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None):
        """Set an attribute on specified environments."""
        if indices is None:
            indices = range(self.num_envs)
            
        for i in indices:
            setattr(self.envs[i], attr_name, value)


def make_vec_env(
    env_id: str,
    n_envs: int = 1,
    seed: Optional[int] = None,
    vec_env_cls: type = SubprocVecEnv,
    vec_env_kwargs: Optional[Dict] = None,
    env_kwargs: Optional[Dict] = None,
    wrapper_class: Optional[callable] = None,
    wrapper_kwargs: Optional[Dict] = None
) -> Union[SubprocVecEnv, DummyVecEnv]:
    """
    Create a vectorized environment from an environment id.
    
    Args:
        env_id: Environment ID
        n_envs: Number of parallel environments
        seed: Random seed
        vec_env_cls: Vectorized environment class
        vec_env_kwargs: Arguments for vec env
        env_kwargs: Arguments for environment creation
        wrapper_class: Optional wrapper to apply
        wrapper_kwargs: Arguments for wrapper
        
    Returns:
        Vectorized environment
    """
    if vec_env_kwargs is None:
        vec_env_kwargs = {}
    if env_kwargs is None:
        env_kwargs = {}
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
        
    def make_env(rank: int, seed: int = 0) -> callable:
        def _init() -> gym.Env:
            env = gym.make(env_id, **env_kwargs)
            
            # Seed the environment
            if seed is not None:
                env.reset(seed=seed + rank)
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
            else:
                env.reset()
            
            # Apply wrapper if specified
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
                
            return env
        return _init
    
    # Create environment functions
    env_fns = [make_env(i, seed) for i in range(n_envs)]
    
    # Create vectorized environment
    return vec_env_cls(env_fns, **vec_env_kwargs)