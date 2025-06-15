#!/usr/bin/env python3
"""
Train PPO on Gymnasium-Robotics Fetch environments with video logging to W&B tables.
"""

import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
from collections import deque

# Load environment variables
load_dotenv()

# Set up offscreen rendering for headless environments
import os
os.environ['MUJOCO_GL'] = 'osmesa'

import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
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
        # Create env with render_mode if video capture is enabled
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
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
            # Record every 5 episodes for better visibility during development
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}",
                episode_trigger=lambda episode_id: episode_id % 5 == 0,
                name_prefix=f"episode",
                disable_logger=True
            )
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def train(args):
    """Main training loop for PPO with Fetch environments."""
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Initialize variables for cleanup
    wandb_run = None
    writer = None
    envs = None
    video_table = None
    global_step = 0
    
    try:
        # Setup WandB logging first
        if args.track and WANDB_AVAILABLE:
            wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
            tags=["gr00t-rl", "ppo", "fetch", args.env_id, args.reward_mode],
            monitor_gym=False,  # We'll handle video logging manually
            mode="online"
        )
        # Define custom x-axis
        wandb.define_metric("global_step")
        wandb.define_metric("episode_step")
        wandb.define_metric("*", step_metric="global_step")
        
        # Create video table with INCREMENTAL mode for ongoing updates
        video_table = wandb.Table(
            columns=["global_step", "episode", "video", "episode_return", "episode_length", "success", "final_distance"],
            log_mode="INCREMENTAL"
        )
        
        # Track if we've logged any videos yet
        videos_logged = False
        
        # Setup tensorboard writer
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
        action_dim=envs.action_space.shape[0],
        hidden_dims=(256, 256),
        use_multimodal_encoder=False,
        device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=args.num_steps,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        device=device.type,
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
    
    # Tracking for episodes
    episode_count = 0
    env0_episode_count = 0  # Track episodes specifically for env 0
    recent_episodes = deque(maxlen=100)
    
    # Video tracking
    video_episode_tracker = {}  # Track which episodes have videos
    
    # Training loop
    num_updates = args.total_timesteps // args.batch_size
    
    for update in range(1, num_updates + 1):
        # Annealing learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Lists to collect episode data during this update
        update_episodes = []
        
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
            
            # Store in buffer
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
                    episode_count += 1
                    episode_data = {
                        "episode_return": info["episode"]["r"],
                        "episode_length": info["episode"]["l"],
                        "episode_success": float(info.get("is_success", False)),
                        "episode_final_distance": info.get("distance_to_goal", -1),
                        "episode_step": episode_count,
                    }
                    recent_episodes.append(episode_data)
                    update_episodes.append(episode_data)
                    
                    # Track this episode for video checking
                    if idx == 0:  # Only env 0 records videos
                        # Track episodes for env 0 specifically
                        video_episode_tracker[env0_episode_count] = {
                            "global_step": global_step,
                            "episode_data": episode_data,
                            "actual_episode": episode_count
                        }
                        env0_episode_count += 1
                    
                    # Log individual episode to tensorboard
                    writer.add_scalar("charts/episodic_return", episode_data["episode_return"], global_step)
                    writer.add_scalar("charts/episodic_length", episode_data["episode_length"], global_step)
                    writer.add_scalar("charts/success_rate", episode_data["episode_success"], global_step)
                    writer.add_scalar("charts/final_distance", episode_data["episode_final_distance"], global_step)
                    
                    # Log individual episode to WandB
                    if args.track and WANDB_AVAILABLE:
                        wandb.log({
                            **episode_data,
                            "global_step": global_step,
                        })
        
        # Check for new videos after each rollout
        if args.track and WANDB_AVAILABLE and args.capture_video:
            video_dir = Path(f"videos/{run_name}")
            if video_dir.exists():
                video_files = sorted(video_dir.glob("*.mp4"))
                
                for video_file in video_files:
                    # Extract episode number from filename (e.g., "episode-10.mp4" or "episode-episode-0.mp4" -> 0)
                    try:
                        # Handle both "episode-0" and "episode-episode-0" formats
                        parts = video_file.stem.split('-')
                        episode_num = int(parts[-1])
                        
                        # Debug logging
                        print(f"Found video file: {video_file.name}, extracted episode: {episode_num}")
                        print(f"Video tracker keys: {list(video_episode_tracker.keys())}")
                        
                        # Check if we have data for this episode and haven't logged it yet
                        if episode_num in video_episode_tracker and not video_episode_tracker[episode_num].get("logged", False):
                            ep_info = video_episode_tracker[episode_num]
                            ep_data = ep_info["episode_data"]
                            
                            # Create video object
                            video_obj = wandb.Video(str(video_file), fps=30, format="mp4")
                            
                            # Add row to incremental table
                            video_table.add_data(
                                ep_info["global_step"],
                                episode_num,
                                video_obj,
                                ep_data["episode_return"],
                                ep_data["episode_length"],
                                ep_data["episode_success"],
                                ep_data["episode_final_distance"]
                            )
                            
                            # Mark that we've added videos
                            videos_logged = True
                            
                            # Mark as logged
                            video_episode_tracker[episode_num]["logged"] = True
                            
                            print(f"Added video for episode {episode_num} to table (global_step: {ep_info['global_step']})")
                            
                    except (ValueError, IndexError):
                        # Skip files that don't match expected format
                        continue
        
        # Compute returns and advantages
        with torch.no_grad():
            next_value = model.get_value(next_obs)
            buffer.compute_returns_and_advantages(next_value, next_done)
        
        # Get data from buffer
        buffer_output = buffer.get(batch_size=None)
        if hasattr(buffer_output, '__next__'):
            try:
                rollout_data = next(buffer_output)
            except StopIteration as e:
                # If StopIteration has a value, that's our data
                if hasattr(e, 'value') and e.value is not None:
                    rollout_data = e.value
                else:
                    raise
        else:
            rollout_data = buffer_output
        
        # Reset buffer for next rollout
        buffer.reset()
        
        # Extract individual components
        b_obs = rollout_data['observations']
        b_actions = rollout_data['actions']
        b_logprobs = rollout_data['log_probs']
        b_advantages = rollout_data['advantages']
        b_returns = rollout_data['returns']
        b_values = rollout_data['values']
        
        # Policy and value network update
        clipfracs = []
        pg_losses = []
        v_losses = []
        entropy_losses = []
        approx_kls = []
        
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
                
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
            
            approx_kls.extend(approx_kl_divs)
            if args.target_kl is not None and np.mean(approx_kl_divs) > args.target_kl:
                break
        
        # Calculate statistics
        if recent_episodes:
            recent_success_rate = np.mean([ep["episode_success"] for ep in recent_episodes])
            recent_avg_return = np.mean([ep["episode_return"] for ep in recent_episodes])
            recent_avg_length = np.mean([ep["episode_length"] for ep in recent_episodes])
            recent_avg_distance = np.mean([ep["episode_final_distance"] for ep in recent_episodes])
        else:
            recent_success_rate = 0.0
            recent_avg_return = 0.0
            recent_avg_length = 0.0
            recent_avg_distance = 0.0
        
        # Create comprehensive metrics dict
        metrics = {
            "learning_rate": optimizer.param_groups[0]["lr"],
            "value_loss": np.mean(v_losses),
            "policy_loss": np.mean(pg_losses),
            "entropy": np.mean(entropy_losses),
            "approx_kl": np.mean(approx_kls),
            "clipfrac": np.mean(clipfracs),
            "recent_success_rate": recent_success_rate,
            "recent_avg_return": recent_avg_return,
            "recent_avg_length": recent_avg_length,
            "recent_avg_distance": recent_avg_distance,
            "SPS": int(global_step / (time.time() - start_time)),
            "global_step": global_step,
            "update": update,
            "total_episodes": episode_count,
        }
        
        # Add episode statistics if we had any complete this update
        if update_episodes:
            metrics.update({
                "update_episodes_completed": len(update_episodes),
                "update_mean_return": np.mean([ep["episode_return"] for ep in update_episodes]),
                "update_mean_success": np.mean([ep["episode_success"] for ep in update_episodes]),
            })
        
        # Log all metrics to tensorboard
        for key, value in metrics.items():
            if key not in ["global_step", "update", "total_episodes"]:
                writer.add_scalar(f"metrics/{key}", value, global_step)
        
        # Log all metrics and tables to WandB
        if args.track and WANDB_AVAILABLE:
            # Log the incremental video table only when we have new videos
            if videos_logged:
                wandb.log({"video_table": video_table}, step=global_step)
            
            # Log all other metrics
            wandb.log(metrics, step=global_step)
        
        # Print progress
        if update % 10 == 0:
            print(f"\nUpdate {update}/{num_updates}, Global Step {global_step}")
            print(f"  Episodes: {episode_count}")
            print(f"  SPS: {metrics['SPS']}")
            print(f"  Recent Success Rate: {recent_success_rate:.2%}")
            print(f"  Recent Avg Return: {recent_avg_return:.3f}")
            print(f"  Recent Avg Distance: {recent_avg_distance:.3f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Approx KL: {metrics['approx_kl']:.4f}")
            if update_episodes:
                print(f"  Episodes this update: {len(update_episodes)}")
    
    # Save final model
    model_path = f"models/{run_name}.pt"
    Path("models").mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        'final_success_rate': recent_success_rate,
        'total_episodes': episode_count,
    }, model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Final success rate: {recent_success_rate:.2%}")
    print(f"Total episodes: {episode_count}")
    
    # Final summary to WandB
    if args.track and WANDB_AVAILABLE:
        wandb.run.summary.update({
            "final_success_rate": recent_success_rate,
            "total_episodes": episode_count,
            "final_avg_return": recent_avg_return,
        })
        
        # Final video table logging moved to finally block for safety
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up resources
        print("\nCleaning up...")
        
        # Close environments
        if envs is not None:
            try:
                envs.close()
            except:
                pass
        
        # Close tensorboard writer
        if writer is not None:
            try:
                writer.close()
            except:
                pass
        
        # Finish WandB run - CRITICAL for uploading table data
        if args.track and WANDB_AVAILABLE and wandb_run is not None:
            try:
                # Log any remaining video table data
                if video_table is not None and len(video_table.data) > 0:
                    print(f"\nLogging {len(video_table.data)} videos to final table...")
                    final_video_table = wandb.Table(
                        columns=video_table.columns,
                        data=video_table.data,
                        log_mode="IMMUTABLE"
                    )
                    wandb.log({"final_video_table": final_video_table})
                    print("Logged final video table data")
                wandb.finish()
                print("WandB run finished successfully")
            except Exception as e:
                print(f"Error finishing WandB run: {e}")


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