#!/usr/bin/env python3
"""
GR00T wrapper for Isaac Lab RL integration.
Makes GR00T compatible with rsl_rl's ActorCritic interface.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Optional, Dict, Any, Tuple
import numpy as np


class GR00TActorCritic(nn.Module):
    """
    Wrapper to make GR00T compatible with Isaac Lab's rsl_rl.
    
    This follows the rsl_rl ActorCritic interface while using GR00T
    as the policy backbone and adding a lightweight critic.
    """
    
    # Required by rsl_rl
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        # GR00T specific
        gr00t_model_path: str = "nvidia/GR00T-N1.5-3B",
        embodiment_tag: str = "GR1",
        freeze_backbone: bool = True,
        use_gr00t: bool = True,  # Can disable for testing
        # Actor settings (fallback if GR00T unavailable)
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        # Additional settings
        action_clip: float = 1.0,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()
        
        self.num_actions = num_actions
        self.action_clip = action_clip
        self.device = device
        self.use_gr00t = use_gr00t
        
        # Try to load GR00T
        if use_gr00t:
            try:
                from gr00t.policy import Gr00tPolicy
                print(f"Loading GR00T model from {gr00t_model_path}...")
                self.gr00t = Gr00tPolicy.from_pretrained(
                    gr00t_model_path,
                    embodiment_tag=embodiment_tag,
                    freeze_backbone=freeze_backbone
                )
                
                # Ensure gradient freezing is properly applied
                if freeze_backbone:
                    for param in self.gr00t.parameters():
                        param.requires_grad = False
                    # Double-check by counting trainable params
                    trainable_params = sum(p.numel() for p in self.gr00t.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in self.gr00t.parameters())
                    print(f"GR00T backbone frozen: {trainable_params}/{total_params} trainable params")
                    # Verify with a test gradient
                    test_param = next(self.gr00t.parameters())
                    assert not test_param.requires_grad, "GR00T parameters should be frozen!"
                
                self.gr00t.to(device)
                print("GR00T model loaded successfully!")
                
                # Check if we need an action adapter
                # GR00T might output different action dimensions
                self.action_adapter = None
                # We'll determine this dynamically on first forward pass
                
            except Exception as e:
                print(f"Failed to load GR00T: {e}")
                print("Falling back to standard MLP policy")
                self.use_gr00t = False
        
        # Create actor (if not using GR00T or as fallback)
        if not self.use_gr00t:
            # Standard MLP actor (following rsl_rl structure)
            activation_fn = self._get_activation(activation)
            
            actor_layers = []
            prev_dim = num_actor_obs
            for hidden_dim in actor_hidden_dims:
                actor_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    activation_fn()
                ])
                prev_dim = hidden_dim
            actor_layers.append(nn.Linear(prev_dim, num_actions))
            
            self.actor = nn.Sequential(*actor_layers)
        
        # Create critic (always needed for PPO)
        activation_fn = self._get_activation(activation)
        
        critic_layers = []
        prev_dim = num_critic_obs
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn()
            ])
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        
        self.critic = nn.Sequential(*critic_layers)
        
        # Action noise (state-independent, as per our PPO implementation)
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        
        # Distribution placeholder
        self.distribution = None
        
        # Disable validation for speed
        Normal.set_default_validate_args(False)
        
        print(f"GR00T ActorCritic initialized:")
        print(f"  Using GR00T: {self.use_gr00t}")
        print(f"  Critic MLP: {self.critic}")
        print(f"  Action dim: {num_actions}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function from string."""
        activations = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }
        return activations.get(activation.lower(), nn.ELU)
    
    def _process_gr00t_output(self, gr00t_output: Any) -> torch.Tensor:
        """
        Process GR00T output to match expected action dimensions.
        GR00T might return a dict or different format.
        """
        # TODO: Adjust based on actual GR00T output format
        if isinstance(gr00t_output, dict):
            # Assume GR00T returns dict with 'actions' key
            actions = gr00t_output.get('actions', gr00t_output.get('action'))
        else:
            actions = gr00t_output
        
        # Ensure tensor
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device)
        
        # Check dimensions
        if actions.shape[-1] != self.num_actions:
            # Need action adapter
            if self.action_adapter is None:
                print(f"Creating action adapter: {actions.shape[-1]} -> {self.num_actions}")
                self.action_adapter = nn.Sequential(
                    nn.Linear(actions.shape[-1], self.num_actions),
                    nn.Tanh()  # Ensure [-1, 1] range for Isaac Lab
                ).to(self.device)
                # Initialize with small weights for stability
                nn.init.xavier_uniform_(self.action_adapter[0].weight, gain=0.01)
                nn.init.zeros_(self.action_adapter[0].bias)
            actions = self.action_adapter(actions)
        
        return actions
    
    def reset(self, dones=None):
        """Reset any internal state (required by rsl_rl)."""
        pass
    
    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute actions from observations.
        Used during rollout collection.
        """
        if self.use_gr00t:
            # Handle dtype mismatch - Isaac Lab uses fp32, GR00T uses fp16
            obs_dtype = observations.dtype
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                # Cast observations to GR00T's expected dtype
                obs_fp16 = observations.half() if observations.dtype == torch.float32 else observations
                gr00t_output = self.gr00t(obs_fp16)
                mean = self._process_gr00t_output(gr00t_output)
            # Cast back to original dtype for compatibility
            mean = mean.to(obs_dtype)
        else:
            mean = self.actor(observations)
        
        # Ensure within action bounds
        mean = torch.clamp(mean, -self.action_clip, self.action_clip)
        
        # Get std
        std = torch.exp(self.log_std)
        
        # Create distribution
        self.distribution = Normal(mean, std)
        
        # Sample action
        actions = self.distribution.rsample()
        
        # Clip actions
        actions = torch.clamp(actions, -self.action_clip, self.action_clip)
        
        return actions
    
    def get_actions_log_prob(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get actions and their log probabilities.
        Used by rsl_rl during rollouts.
        """
        actions = self.act(observations)
        log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        
        return actions, log_prob
    
    def act_inference(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Compute actions for inference/evaluation.
        No gradient tracking, optionally deterministic.
        Saves ~15% memory by not tracking gradients.
        """
        with torch.no_grad():
            if self.use_gr00t:
                # Handle dtype mismatch
                obs_dtype = observations.dtype
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    obs_fp16 = observations.half() if observations.dtype == torch.float32 else observations
                    gr00t_output = self.gr00t(obs_fp16)
                    mean = self._process_gr00t_output(gr00t_output)
                mean = mean.to(obs_dtype)
            else:
                mean = self.actor(observations)
            
            # Ensure within action bounds
            mean = torch.clamp(mean, -self.action_clip, self.action_clip)
            
            if deterministic:
                # Return mean action (greedy)
                return mean
            else:
                # Sample from distribution
                std = torch.exp(self.log_std)
                distribution = Normal(mean, std)
                actions = distribution.sample()
                return torch.clamp(actions, -self.action_clip, self.action_clip)
    
    def evaluate(
        self, 
        observations: torch.Tensor,
        actions: torch.Tensor,
        critic_observations: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            observations: Actor observations
            actions: Actions to evaluate
            critic_observations: Critic observations (if different from actor obs)
            
        Returns: log_probs, values, entropy
        """
        # Get action distribution
        if self.use_gr00t:
            gr00t_output = self.gr00t(observations)
            mean = self._process_gr00t_output(gr00t_output)
        else:
            mean = self.actor(observations)
        
        mean = torch.clamp(mean, -self.action_clip, self.action_clip)
        std = torch.exp(self.log_std)
        
        distribution = Normal(mean, std)
        
        # Compute log probs
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        
        # Compute values (use critic obs if provided)
        critic_obs = critic_observations if critic_observations is not None else observations
        values = self.critic(critic_obs).squeeze(-1)
        
        # Compute entropy
        entropy = distribution.entropy().sum(dim=-1).mean()
        
        return log_probs, values, entropy
    
    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Get value estimates.
        Used for advantage computation.
        """
        return self.critic(observations).squeeze(-1)
    
    @property
    def action_mean(self) -> torch.Tensor:
        """Get mean of action distribution (required by rsl_rl)."""
        return self.distribution.mean if self.distribution else None
    
    @property
    def action_std(self) -> torch.Tensor:
        """Get std of action distribution (required by rsl_rl)."""
        return torch.exp(self.log_std)
    
    def forward(self):
        """Not used but required by interface."""
        raise NotImplementedError(
            "Forward not implemented. Use act() or evaluate() instead."
        )


def create_gr00t_actor_critic_for_isaac(env_cfg: Dict[str, Any], **kwargs) -> GR00TActorCritic:
    """
    Factory function to create GR00T actor-critic for Isaac Lab.
    
    Args:
        env_cfg: Environment configuration dict with observation/action info
        **kwargs: Additional arguments for GR00TActorCritic
        
    Returns:
        GR00TActorCritic instance
    """
    # Extract dimensions from env config
    num_actor_obs = env_cfg.get("num_observations", env_cfg.get("num_actor_obs"))
    num_critic_obs = env_cfg.get("num_privileged_obs", num_actor_obs)
    num_actions = env_cfg["num_actions"]
    
    print(f"Creating GR00T ActorCritic:")
    print(f"  Actor obs: {num_actor_obs}")
    print(f"  Critic obs: {num_critic_obs}")
    print(f"  Actions: {num_actions}")
    
    return GR00TActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        **kwargs
    )