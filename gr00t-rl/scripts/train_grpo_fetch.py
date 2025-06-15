#!/usr/bin/env python3
"""
Train GRPO (Group Relative Policy Optimization) on Fetch environments.
This implements GRPO for robotics tasks with goal-conditioned rewards.
"""

import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import gymnasium_robotics  # This import registers the Fetch environments
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
from environments.fetch_wrapper import FetchGoalWrapper
from utils.logging import get_system_metrics

# Import wandb if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging to tensorboard only")


class GRPOTrainer:
    """
    GRPO trainer for robotics tasks.
    Key differences from language GRPO:
    - Multi-step episodes instead of single generation
    - Continuous action spaces
    - Goal-conditioned rewards
    """
    
    def __init__(
        self,
        model: nn.Module,
        env_id: str,
        num_rollouts: int = 8,
        learning_rate: float = 3e-4,
        beta: float = 0.0,  # KL penalty (0 for pure GRPO)
        epsilon: float = 0.2,  # Clipping threshold
        device: str = "cuda",
        observation_mode: str = "observation_goal",
        reward_mode: str = "sparse",
        normalize_advantages: bool = True,
        use_baseline_model: bool = False,
    ):
        self.model = model
        self.env_id = env_id
        self.num_rollouts = num_rollouts
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.device = device
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.normalize_advantages = normalize_advantages
        self.use_baseline_model = use_baseline_model
        
        # Create environments for parallel rollouts
        self.envs = []
        for _ in range(num_rollouts):
            env = gym.make(env_id)
            env = FetchGoalWrapper(
                env,
                observation_mode=observation_mode,
                reward_mode=reward_mode,
                device="cpu"
            )
            self.envs.append(env)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
        
        # Optional baseline model for KL computation
        if use_baseline_model:
            self.baseline_model = PPOGr00tActorCriticV2(
                observation_space=self.envs[0].observation_space,
                action_dim=self.envs[0].action_space.shape[0],
                hidden_dims=(256, 256),
                use_multimodal_encoder=False,
                device=device
            ).to(device)
            self.baseline_model.load_state_dict(model.state_dict())
            self.baseline_model.eval()
        
        # Tracking
        self.episode_count = 0
        self.global_step = 0
        
    def collect_rollouts(self, initial_seed: int = None):
        """
        Collect multiple rollouts from the same initial state.
        Returns trajectories and rewards for GRPO update.
        """
        trajectories = []
        total_rewards = []
        episode_lengths = []
        success_flags = []
        
        # Set same initial state for all rollouts if seed provided
        if initial_seed is not None:
            for env in self.envs:
                env.reset(seed=initial_seed)
        
        # Collect rollouts in parallel
        for i in range(self.num_rollouts):
            trajectory = {
                'observations': [],
                'actions': [],
                'log_probs': [],
                'rewards': [],
                'dones': [],
                'infos': []
            }
            
            obs, info = self.envs[i].reset()
            done = False
            episode_length = 0
            total_reward = 0
            
            while not done and episode_length < 50:  # Fetch episodes are typically 50 steps
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, log_prob, _, _ = self.model.get_action_and_value(obs_tensor)
                
                action_np = action.cpu().numpy().squeeze()
                next_obs, reward, terminated, truncated, info = self.envs[i].step(action_np)
                done = terminated or truncated
                
                # Store trajectory data
                trajectory['observations'].append(obs)
                trajectory['actions'].append(action_np)
                trajectory['log_probs'].append(log_prob.item())
                trajectory['rewards'].append(reward)
                trajectory['dones'].append(done)
                trajectory['infos'].append(info)
                
                obs = next_obs
                total_reward += reward
                episode_length += 1
                self.global_step += 1
            
            trajectories.append(trajectory)
            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            success_flags.append(info.get('is_success', False))
            self.episode_count += 1
        
        return trajectories, total_rewards, episode_lengths, success_flags
    
    def compute_advantages(self, rewards: list):
        """
        Compute GRPO advantages (relative to group mean).
        """
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        if std_reward < 1e-8:
            # All rewards are the same - no learning signal
            return np.zeros_like(rewards)
        
        # GRPO advantage: normalize by group statistics
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)
        
        return advantages
    
    def update(self, trajectories: list, advantages: np.ndarray):
        """
        Update policy using GRPO loss.
        """
        # Prepare batch data
        all_observations = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        
        for i, traj in enumerate(trajectories):
            # Repeat advantage for all steps in trajectory
            traj_advantage = advantages[i]
            
            for j in range(len(traj['observations'])):
                all_observations.append(traj['observations'][j])
                all_actions.append(traj['actions'][j])
                all_old_log_probs.append(traj['log_probs'][j])
                all_advantages.append(traj_advantage)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(all_observations)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(all_actions)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(all_old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(all_advantages).to(self.device)
        
        # Normalize advantages at batch level (optional)
        if self.normalize_advantages:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Forward pass
        _, new_log_probs, entropy, _ = self.model.get_action_and_value(obs_tensor, actions_tensor)
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        
        # GRPO loss (similar to PPO but with group-relative advantages)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Add entropy bonus
        entropy_loss = -entropy.mean()
        
        # Optional KL penalty
        kl_loss = 0
        if self.beta > 0 and self.use_baseline_model:
            with torch.no_grad():
                _, baseline_log_probs, _, _ = self.baseline_model.get_action_and_value(obs_tensor, actions_tensor)
            kl_div = (old_log_probs_tensor - baseline_log_probs).mean()
            kl_loss = self.beta * kl_div
        
        # Total loss
        total_loss = policy_loss + 0.01 * entropy_loss + kl_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        # Return statistics
        stats = {
            'policy_loss': policy_loss.item(),
            'entropy': -entropy_loss.item(),
            'kl_div': kl_loss.item() if self.beta > 0 else 0,
            'mean_ratio': ratio.mean().item(),
            'mean_advantage': advantages_tensor.mean().item(),
            'std_advantage': advantages_tensor.std().item(),
        }
        
        return stats


def train(args):
    """Main training loop for GRPO."""
    run_name = f"{args.env_id}__grpo__{args.seed}__{int(time.time())}"
    
    # Setup logging
    if args.track and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
            tags=["grpo", "fetch", args.env_id, args.reward_mode],
            monitor_gym=True  # Enable automatic Gymnasium video logging
        )
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Create a test environment for model initialization
    test_env = gym.make(args.env_id)
    test_env = FetchGoalWrapper(
        test_env,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        device="cpu"
    )
    
    # Create model
    model = PPOGr00tActorCriticV2(
        observation_space=test_env.observation_space,
        action_dim=test_env.action_space.shape[0],
        hidden_dims=(256, 256),
        use_multimodal_encoder=False,
        device=device
    ).to(device)
    
    test_env.close()
    
    # Create GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        env_id=args.env_id,
        num_rollouts=args.num_rollouts,
        learning_rate=args.learning_rate,
        beta=args.beta,
        epsilon=args.epsilon,
        device=device,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        normalize_advantages=args.normalize_advantages,
        use_baseline_model=args.beta > 0
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    start_time = time.time()
    
    # Training loop
    num_iterations = args.total_timesteps // (args.num_rollouts * 50)  # Approximate
    
    for iteration in range(1, num_iterations + 1):
        # Collect rollouts
        if args.use_same_init:
            # Use same initial state for all rollouts in this iteration
            seed = args.seed + iteration
        else:
            seed = None
            
        trajectories, rewards, lengths, successes = trainer.collect_rollouts(seed)
        
        # Compute advantages
        advantages = trainer.compute_advantages(rewards)
        
        # Skip update if no learning signal
        if np.std(rewards) < 1e-8:
            print(f"Iteration {iteration}: All rewards identical ({rewards[0]:.3f}), skipping update")
            continue
        
        # Update policy
        stats = trainer.update(trajectories, advantages)
        
        # Track metrics
        episode_rewards.extend(rewards)
        episode_lengths.extend(lengths)
        success_rates.extend(successes)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        success_rate = np.mean(successes)
        
        # Logging
        writer.add_scalar("grpo/mean_reward", mean_reward, trainer.global_step)
        writer.add_scalar("grpo/std_reward", std_reward, trainer.global_step)
        writer.add_scalar("grpo/success_rate", success_rate, trainer.global_step)
        writer.add_scalar("grpo/mean_episode_length", np.mean(lengths), trainer.global_step)
        
        for key, value in stats.items():
            writer.add_scalar(f"losses/{key}", value, trainer.global_step)
        
        if args.track and WANDB_AVAILABLE:
            wandb.log({
                "grpo/mean_reward": mean_reward,
                "grpo/std_reward": std_reward,
                "grpo/success_rate": success_rate,
                "grpo/mean_episode_length": np.mean(lengths),
                **{f"losses/{k}": v for k, v in stats.items()},
                "global_step": trainer.global_step,
            })
        
        # Print progress
        if iteration % 10 == 0:
            recent_success_rate = np.mean(success_rates[-100:]) if len(success_rates) > 100 else np.mean(success_rates)
            print(f"\nIteration {iteration}/{num_iterations}")
            print(f"  Global Step: {trainer.global_step}")
            print(f"  Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Recent Success Rate (100 eps): {recent_success_rate:.2%}")
            print(f"  Policy Loss: {stats['policy_loss']:.4f}")
            print(f"  Entropy: {stats['entropy']:.4f}")
            print(f"  Mean Advantage: {stats['mean_advantage']:.4f}")
            print(f"  SPS: {int(trainer.global_step / (time.time() - start_time))}")
    
    # Save final model
    model_path = f"models/{run_name}.pt"
    Path("models").mkdir(exist_ok=True)
    final_success_rate = np.mean(success_rates[-100:]) if len(success_rates) > 100 else np.mean(success_rates)
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'final_success_rate': final_success_rate,
        'total_episodes': len(episode_rewards),
    }, model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Final success rate: {final_success_rate:.2%}")
    
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    # Experiment arguments
    parser.add_argument("--exp-name", type=str, default="grpo_fetch",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=bool, default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=bool, default=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="pippa",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="wild-ai",
        help="the entity (team) of wandb's project")
    
    # Environment arguments
    parser.add_argument("--env-id", type=str, default="FetchReach-v3",
        help="the id of the Fetch environment",
        choices=["FetchReach-v3", "FetchPush-v3", "FetchSlide-v3", "FetchPickAndPlace-v3"])
    parser.add_argument("--observation-mode", type=str, default="observation_goal",
        help="observation mode for the wrapper")
    parser.add_argument("--reward-mode", type=str, default="sparse",
        help="reward mode for the wrapper",
        choices=["sparse", "dense", "distance"])
    
    # GRPO arguments
    parser.add_argument("--num-rollouts", type=int, default=8,
        help="number of rollouts per GRPO update")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--beta", type=float, default=0.0,
        help="KL penalty coefficient (0 for pure GRPO)")
    parser.add_argument("--epsilon", type=float, default=0.2,
        help="clipping threshold for policy ratio")
    parser.add_argument("--normalize-advantages", type=bool, default=True,
        help="normalize advantages at batch level")
    parser.add_argument("--use-same-init", type=bool, default=True,
        help="use same initial state for all rollouts in iteration")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()