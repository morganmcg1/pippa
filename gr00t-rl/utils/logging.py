#!/usr/bin/env python3
"""
Enhanced logging utilities for GR00T RL training.
Provides comprehensive metrics tracking for debugging and analysis.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
import psutil
import GPUtil


def compute_gradient_norm(parameters) -> float:
    """Compute the L2 norm of gradients."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """
    Compute the explained variance of the value function.
    
    A value close to 1 means the value function is doing a good job
    of predicting returns.
    """
    var_returns = torch.var(returns)
    if var_returns == 0:
        return 0.0
    return 1 - torch.var(returns - values) / var_returns


def get_system_metrics() -> Dict[str, float]:
    """Get system resource usage metrics."""
    metrics = {}
    
    # CPU metrics
    metrics['system/cpu_percent'] = psutil.cpu_percent(interval=0.1)
    metrics['system/memory_percent'] = psutil.virtual_memory().percent
    
    # GPU metrics
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Assuming single GPU
            metrics['system/gpu_memory_used'] = gpu.memoryUsed
            metrics['system/gpu_memory_total'] = gpu.memoryTotal
            metrics['system/gpu_utilization'] = gpu.load * 100
            metrics['system/gpu_temperature'] = gpu.temperature
    except:
        pass  # GPUtil not available or no GPU
        
    return metrics


def compute_grpo_health_metrics(
    rollouts: List[Any],
    advantages: torch.Tensor
) -> Dict[str, float]:
    """
    Compute GRPO-specific health metrics.
    
    Args:
        rollouts: List of rollout data
        advantages: Computed advantages
        
    Returns:
        Dictionary of health metrics
    """
    metrics = {}
    
    # Check for zero standard deviation (critical for GRPO)
    returns = torch.tensor([r.episode_return for r in rollouts])
    if returns.std() == 0:
        metrics['train/frac_zero_std'] = 1.0
    else:
        metrics['train/frac_zero_std'] = 0.0
    
    # Return distribution metrics
    metrics['train/return_mean'] = returns.mean().item()
    metrics['train/return_std'] = returns.std().item()
    metrics['train/return_min'] = returns.min().item()
    metrics['train/return_max'] = returns.max().item()
    
    # Advantage statistics
    metrics['train/advantage_mean'] = advantages.mean().item()
    metrics['train/advantage_std'] = advantages.std().item()
    
    # Diversity metric
    unique_returns = len(torch.unique(returns))
    metrics['train/return_diversity'] = unique_returns / len(returns)
    
    return metrics


def compute_ppo_diagnostics(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    values: torch.Tensor,
    returns: torch.Tensor,
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor
) -> Dict[str, float]:
    """
    Compute PPO-specific diagnostic metrics.
    
    Args:
        ratio: Probability ratio (new/old)
        advantages: Advantages
        clip_range: PPO clip parameter
        values: Value predictions
        returns: Actual returns
        old_log_probs: Log probs from old policy
        new_log_probs: Log probs from new policy
        
    Returns:
        Dictionary of diagnostic metrics
    """
    metrics = {}
    
    # Clip fraction
    clipped = torch.abs(ratio - 1.0) > clip_range
    metrics['train/clip_fraction'] = clipped.float().mean().item()
    
    # KL divergence
    kl = (old_log_probs - new_log_probs).mean()
    metrics['train/kl'] = kl.item()
    
    # Explained variance
    ev = compute_explained_variance(values, returns)
    metrics['train/explained_variance'] = ev
    
    # Advantage statistics
    metrics['train/advantage_mean'] = advantages.mean().item()
    metrics['train/advantage_std'] = advantages.std().item()
    
    # Value function error
    value_error = (values - returns).pow(2).mean()
    metrics['train/value_error'] = value_error.item()
    
    return metrics


def log_action_statistics(actions: torch.Tensor) -> Dict[str, float]:
    """Log statistics about actions taken."""
    metrics = {}
    
    # Basic stats
    metrics['actions/mean'] = actions.mean().item()
    metrics['actions/std'] = actions.std().item()
    metrics['actions/min'] = actions.min().item()
    metrics['actions/max'] = actions.max().item()
    
    # Per-dimension stats if multi-dimensional
    if len(actions.shape) > 1:
        for i in range(actions.shape[-1]):
            metrics[f'actions/dim_{i}_mean'] = actions[..., i].mean().item()
            metrics[f'actions/dim_{i}_std'] = actions[..., i].std().item()
            
    return metrics


def log_reward_components(
    reward_components: Dict[str, float],
    prefix: str = "reward_components"
) -> Dict[str, float]:
    """
    Log individual reward components.
    
    Args:
        reward_components: Dict of component names to values
        prefix: Prefix for metric names
        
    Returns:
        Formatted metrics dict
    """
    metrics = {}
    for name, value in reward_components.items():
        metrics[f"{prefix}/{name}"] = value
    return metrics


def create_comprehensive_logs(
    base_logs: Dict[str, Any],
    model_params: Optional[torch.nn.Module] = None,
    rollouts: Optional[List[Any]] = None,
    ppo_data: Optional[Dict[str, torch.Tensor]] = None,
    include_system: bool = True
) -> Dict[str, Any]:
    """
    Create comprehensive logs combining base metrics with diagnostics.
    
    Args:
        base_logs: Base logging dict
        model_params: Model parameters for gradient norm
        rollouts: GRPO rollouts for health metrics
        ppo_data: PPO diagnostic data
        include_system: Whether to include system metrics
        
    Returns:
        Enhanced logging dictionary
    """
    logs = base_logs.copy()
    
    # Add gradient norm if model provided
    if model_params is not None:
        logs['train/grad_norm'] = compute_gradient_norm(
            p for p in model_params if p.grad is not None
        )
    
    # Add GRPO health metrics if rollouts provided
    if rollouts is not None:
        # Compute advantages for health metrics
        returns = torch.tensor([r.episode_return for r in rollouts])
        if returns.std() > 0:
            advantages = (returns - returns.mean()) / returns.std()
        else:
            advantages = returns - returns.mean()
        grpo_metrics = compute_grpo_health_metrics(rollouts, advantages)
        logs.update(grpo_metrics)
    
    # Add PPO diagnostics if data provided
    if ppo_data is not None:
        ppo_metrics = compute_ppo_diagnostics(**ppo_data)
        logs.update(ppo_metrics)
    
    # Add system metrics
    if include_system:
        system_metrics = get_system_metrics()
        logs.update(system_metrics)
    
    return logs