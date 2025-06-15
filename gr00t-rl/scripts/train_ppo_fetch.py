#!/usr/bin/env python3
"""
Train PPO on Gymnasium-Robotics Fetch environments.
Supports goal-conditioned tasks with different reward modes.
"""

import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics  # This import registers the Fetch environments
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up offscreen rendering for headless environments
import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Use OSMesa for software rendering on headless servers

import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
# from configs.ppo_config_v2 import PPOConfigV2  # Not used in this script
from environments.fetch_wrapper import FetchGoalWrapper
from environments.vec_isaac_env import make_vec_env, SubprocVecEnv, DummyVecEnv
from utils.buffers import PPORolloutBuffer as RolloutBuffer
from utils.logging import get_system_metrics

# Import wandb if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging to tensorboard only")


def make_fetch_env(env_id: str, idx: int, capture_video: bool, run_name: str, 
                   observation_mode: str = "observation_goal", reward_mode: str = "sparse"):
    """Create a Fetch environment with our wrapper."""
    def thunk():
        env = gym.make(env_id)
        env = FetchGoalWrapper(
            env,
            observation_mode=observation_mode,
            reward_mode=reward_mode,
            goal_in_observation=True,
            normalize_observations=False,
            device="cpu"
        )
        
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}",
                episode_trigger=lambda episode_id: episode_id % 100 == 0
            )
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def train(args):
    """Main training loop for PPO with Fetch environments."""
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Setup logging
    if args.track and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
            tags=["ppo", "fetch", args.env_id, args.reward_mode],
            monitor_gym=True  # Enable automatic Gymnasium video logging
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Create environments
    env_fns = [make_fetch_env(args.env_id, i, args.capture_video, run_name, 
                              args.observation_mode, args.reward_mode) 
               for i in range(args.num_envs)]
    
    if args.num_envs > 1:
        envs = SubprocVecEnv(env_fns)
    else:
        envs = DummyVecEnv(env_fns)
    
    # Create model
    model = PPOGr00tActorCriticV2(
        observation_space=envs.observation_space,
        action_dim=envs.action_space.shape[0],  # Fetch has continuous actions
        hidden_dims=(256, 256),  # Larger network for complex tasks
        use_multimodal_encoder=False,
        device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=args.num_steps,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device.type,  # Pass string device type, not the device object
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_envs=args.num_envs
    )
    
    # Initialize tracking variables
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # Tracking success rate
    success_tracker = []
    episode_tracker = []
    
    # Training loop
    num_updates = args.total_timesteps // args.batch_size
    
    for update in range(1, num_updates + 1):
        # Annealing learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Collect rollout
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            
            # Store current observation for this step
            obs = next_obs
            
            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(obs)
            
            # Execute action in environment
            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            rewards = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)
            
            # Store in buffer (with the observation that led to this action)
            buffer.add(
                obs,
                action,
                rewards,
                next_done,
                value.flatten(),
                logprob
            )
            
            # Track episode statistics
            for idx, info in enumerate(infos):
                if "episode" in info:
                    episode_info = {
                        "return": info["episode"]["r"],
                        "length": info["episode"]["l"],
                        "success": info.get("is_success", False),
                        "final_distance": info.get("distance_to_goal", -1)
                    }
                    episode_tracker.append(episode_info)
                    
                    # Log to tensorboard
                    writer.add_scalar("charts/episodic_return", episode_info["return"], global_step)
                    writer.add_scalar("charts/episodic_length", episode_info["length"], global_step)
                    writer.add_scalar("charts/success_rate", float(episode_info["success"]), global_step)
                    writer.add_scalar("charts/final_distance", episode_info["final_distance"], global_step)
                    
                    if args.track and WANDB_AVAILABLE:
                        wandb.log({
                            "charts/episodic_return": episode_info["return"],
                            "charts/episodic_length": episode_info["length"],
                            "charts/success_rate": float(episode_info["success"]),
                            "charts/final_distance": episode_info["final_distance"],
                            "global_step": global_step,
                        })
        
        # Compute returns and advantages
        with torch.no_grad():
            next_value = model.get_value(next_obs)
            buffer.compute_returns_and_advantages(next_value, next_done)
        
        # Get data from buffer (without batch_size to get all data at once)
        rollout_data = buffer.get(batch_size=None)
        
        # Extract individual components
        b_obs = rollout_data['observations']
        b_actions = rollout_data['actions']
        b_logprobs = rollout_data['log_probs']
        b_advantages = rollout_data['advantages']  # Already normalized by buffer.get()
        b_returns = rollout_data['returns']
        b_values = rollout_data['values']
        
        # Policy and value network update
        clipfracs = []
        for epoch in range(args.update_epochs):
            approx_kl_divs = []
            
            # Mini-batch updates
            b_inds = np.arange(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # Calculate approx_kl
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kl_divs.append(approx_kl.item())
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                
                mb_advantages = b_advantages[mb_inds]
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            
            if args.target_kl is not None and np.mean(approx_kl_divs) > args.target_kl:
                break
        
        # Calculate success rate over recent episodes
        recent_episodes = episode_tracker[-100:] if len(episode_tracker) > 100 else episode_tracker
        if recent_episodes:
            recent_success_rate = np.mean([ep["success"] for ep in recent_episodes])
            avg_final_distance = np.mean([ep["final_distance"] for ep in recent_episodes])
        else:
            recent_success_rate = 0.0
            avg_final_distance = 0.0
        
        # Logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", np.mean(approx_kl_divs), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/recent_success_rate", recent_success_rate, global_step)
        writer.add_scalar("charts/avg_final_distance", avg_final_distance, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        if args.track and WANDB_AVAILABLE:
            wandb.log({
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/approx_kl": np.mean(approx_kl_divs),
                "losses/clipfrac": np.mean(clipfracs),
                "charts/recent_success_rate": recent_success_rate,
                "charts/avg_final_distance": avg_final_distance,
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "global_step": global_step,
            })
        
        # Print progress
        if update % 10 == 0:
            print(f"\nUpdate {update}/{num_updates}, Global Step {global_step}")
            print(f"  SPS: {int(global_step / (time.time() - start_time))}")
            print(f"  Recent Success Rate: {recent_success_rate:.2%}")
            print(f"  Avg Final Distance: {avg_final_distance:.3f}")
            print(f"  Value Loss: {v_loss.item():.4f}")
            print(f"  Policy Loss: {pg_loss.item():.4f}")
            print(f"  Entropy: {entropy_loss.item():.4f}")
            print(f"  Approx KL: {np.mean(approx_kl_divs):.4f}")
    
    # Save final model
    model_path = f"models/{run_name}.pt"
    Path("models").mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        'final_success_rate': recent_success_rate,
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    envs.close()
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    # Experiment arguments
    parser.add_argument("--exp-name", type=str, default="ppo_fetch",
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
    parser.add_argument("--capture-video", type=bool, default=True,
        help="whether to capture videos of the agent performances")
    
    # Algorithm arguments
    parser.add_argument("--env-id", type=str, default="FetchReach-v3",
        help="the id of the Fetch environment",
        choices=["FetchReach-v3", "FetchPush-v3", "FetchSlide-v3", "FetchPickAndPlace-v3"])
    parser.add_argument("--observation-mode", type=str, default="observation_goal",
        help="observation mode for the wrapper",
        choices=["observation", "observation_goal", "full"])
    parser.add_argument("--reward-mode", type=str, default="sparse",
        help="reward mode for the wrapper",
        choices=["sparse", "dense", "distance"])
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=bool, default=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=bool, default=True,
        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    train(args)


if __name__ == "__main__":
    main()