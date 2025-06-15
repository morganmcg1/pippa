#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) adaptation for GR00T N1.5 model.
Critic-free RL using relative rewards from multiple rollouts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from gr00t.model.policy import Gr00tPolicy
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class GRPORollout:
    """Container for a single rollout's data."""
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    infos: List[Dict[str, Any]]
    episode_return: float


class GRPOGr00t(nn.Module):
    """
    GRPO wrapper for GR00T model.
    Implements critic-free RL using relative rewards.
    """
    
    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "GR1",
        num_rollouts_per_update: int = 8,
        freeze_backbone: bool = True,
        temperature: float = 1.0,
        beta: float = 0.0,  # KL penalty coefficient (0 for Dr GRPO)
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
        
        # Reference model for KL computation (if beta > 0)
        if beta > 0:
            self.ref_policy = Gr00tPolicy.from_pretrained(
                model_path,
                embodiment_tag=embodiment_tag,
                device=device
            )
            # Freeze reference model
            for param in self.ref_policy.model.parameters():
                param.requires_grad = False
        else:
            self.ref_policy = None
        
        # GRPO parameters
        self.num_rollouts_per_update = num_rollouts_per_update
        self.temperature = temperature
        self.beta = beta
        
        # Freeze settings
        if freeze_backbone:
            self._freeze_backbone()
            
        self.device = device
        
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.gr00t_model.backbone.parameters():
            param.requires_grad = False
    
    def generate_rollouts(
        self,
        env,
        initial_obs: Dict[str, Any],
        max_steps: int = 1000
    ) -> List[GRPORollout]:
        """
        Generate multiple rollouts from the same initial state.
        
        Args:
            env: Gymnasium environment
            initial_obs: Initial observation to start from
            max_steps: Maximum steps per rollout
            
        Returns:
            rollouts: List of rollout data
        """
        rollouts = []
        
        for i in range(self.num_rollouts_per_update):
            # Reset to same initial state
            obs = env.reset_to(initial_obs)
            
            # Storage for this rollout
            observations = []
            actions = []
            rewards = []
            dones = []
            infos = []
            
            done = False
            step = 0
            
            while not done and step < max_steps:
                # Get action from policy
                with torch.no_grad():
                    action_output = self.gr00t_policy.get_action(obs)
                    action = action_output["action"]
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                
                obs = next_obs
                step += 1
            
            # Convert to tensors
            rollout = GRPORollout(
                observations=self._stack_observations(observations),
                actions=torch.stack(actions),
                rewards=torch.tensor(rewards, dtype=torch.float32),
                dones=torch.tensor(dones, dtype=torch.float32),
                infos=infos,
                episode_return=sum(rewards)
            )
            
            rollouts.append(rollout)
            
        return rollouts
    
    def compute_grpo_loss(
        self,
        rollouts: List[GRPORollout],
        clip_epsilon: float = 0.2
    ) -> torch.Tensor:
        """
        Compute GRPO loss using relative rewards.
        
        Args:
            rollouts: List of rollouts from same initial state
            clip_epsilon: PPO clipping parameter
            
        Returns:
            loss: GRPO loss to minimize
        """
        # Extract episode returns
        returns = torch.tensor([r.episode_return for r in rollouts])
        
        # Compute relative advantages (GRPO style)
        if returns.std() > 0:
            advantages = (returns - returns.mean()) / returns.std()
        else:
            advantages = returns - returns.mean()
        
        total_loss = 0.0
        num_transitions = 0
        
        # Process each rollout
        for rollout_idx, rollout in enumerate(rollouts):
            advantage = advantages[rollout_idx]
            
            # Get action probabilities under current policy
            action_logprobs = self._compute_action_logprobs(
                rollout.observations,
                rollout.actions
            )
            
            # Get old action probabilities (stored during rollout generation)
            # For simplicity, we'll use the current policy as old policy
            # In practice, you'd store these during rollout generation
            with torch.no_grad():
                old_action_logprobs = action_logprobs.detach()
            
            # Compute probability ratio
            ratio = torch.exp(action_logprobs - old_action_logprobs)
            
            # Clipped surrogate loss (PPO style)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
            
            # Take minimum of clipped and unclipped
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Add KL penalty if beta > 0
            if self.beta > 0 and self.ref_policy is not None:
                ref_logprobs = self._compute_action_logprobs(
                    rollout.observations,
                    rollout.actions,
                    policy=self.ref_policy
                )
                kl_div = (old_action_logprobs - ref_logprobs).mean()
                policy_loss += self.beta * kl_div
            
            total_loss += policy_loss * len(rollout.actions)
            num_transitions += len(rollout.actions)
        
        # Average loss across all transitions
        return total_loss / num_transitions
    
    def _compute_action_logprobs(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        policy: Optional[Gr00tPolicy] = None
    ) -> torch.Tensor:
        """
        Compute log probabilities of actions under policy.
        
        Note: This is a placeholder implementation.
        Actual implementation needs proper flow matching likelihood computation.
        """
        if policy is None:
            policy = self.gr00t_policy
            
        # For now, return dummy log probabilities
        # In practice, this would compute the flow matching likelihood
        batch_size = actions.shape[0]
        return torch.zeros(batch_size, device=self.device)
    
    def _stack_observations(self, observations: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Stack list of observation dicts into batched dict of tensors."""
        batched_obs = {}
        
        for key in observations[0].keys():
            if isinstance(observations[0][key], torch.Tensor):
                batched_obs[key] = torch.stack([obs[key] for obs in observations])
            elif isinstance(observations[0][key], np.ndarray):
                batched_obs[key] = torch.from_numpy(
                    np.stack([obs[key] for obs in observations])
                )
            else:
                # For non-tensor data, just keep as list
                batched_obs[key] = [obs[key] for obs in observations]
                
        return batched_obs
    
    def update_policy(
        self,
        rollouts: List[GRPORollout],
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 1,
        clip_epsilon: float = 0.2
    ) -> Dict[str, float]:
        """
        Update policy using GRPO.
        
        Args:
            rollouts: List of rollouts from same initial state
            optimizer: Optimizer for policy parameters
            num_epochs: Number of epochs to train on batch
            clip_epsilon: PPO clipping parameter
            
        Returns:
            info: Dictionary of training metrics
        """
        losses = []
        
        for epoch in range(num_epochs):
            # Compute loss
            loss = self.compute_grpo_loss(rollouts, clip_epsilon)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.gr00t_model.parameters(), 
                max_norm=0.5
            )
            
            optimizer.step()
            
            losses.append(loss.item())
        
        # Compute metrics
        all_returns = [r.episode_return for r in rollouts]
        
        info = {
            "loss": np.mean(losses),
            "mean_return": np.mean(all_returns),
            "std_return": np.std(all_returns),
            "max_return": np.max(all_returns),
            "min_return": np.min(all_returns),
        }
        
        # Add KL divergence if using reference model
        if self.beta > 0 and self.ref_policy is not None:
            with torch.no_grad():
                kl_total = 0.0
                num_samples = 0
                
                for rollout in rollouts:
                    current_logprobs = self._compute_action_logprobs(
                        rollout.observations,
                        rollout.actions
                    )
                    ref_logprobs = self._compute_action_logprobs(
                        rollout.observations,
                        rollout.actions,
                        policy=self.ref_policy
                    )
                    kl = (current_logprobs - ref_logprobs).mean()
                    kl_total += kl.item() * len(rollout.actions)
                    num_samples += len(rollout.actions)
                
                info["kl_divergence"] = kl_total / num_samples if num_samples > 0 else 0.0
        
        return info
    
    def save(self, path: str):
        """Save the model state."""
        torch.save({
            'gr00t_state_dict': self.gr00t_model.state_dict(),
            'config': {
                'num_rollouts_per_update': self.num_rollouts_per_update,
                'temperature': self.temperature,
                'beta': self.beta,
            }
        }, path)
        
    def load(self, path: str):
        """Load the model state."""
        checkpoint = torch.load(path)
        self.gr00t_model.load_state_dict(checkpoint['gr00t_state_dict'])
        
        # Update config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.num_rollouts_per_update = config.get('num_rollouts_per_update', self.num_rollouts_per_update)
            self.temperature = config.get('temperature', self.temperature)
            self.beta = config.get('beta', self.beta)