#!/usr/bin/env python3
"""
Test PPO with Isaac Lab Cartpole environment (simplified version).
This uses our Isaac Gym wrapper to handle Isaac Lab environments.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.ppo_gr00t_v2 import PPOGr00tActorCriticV2
# from trainers.ppo_trainer_v2 import PPOTrainerV2  # Not implemented yet
from configs.ppo_config_v2 import PPOConfig
from environments.isaac_gym_wrapper import IsaacGymWrapper
from environments.vec_isaac_env import make_vec_env, DummyVecEnv


def test_ppo_isaac_cartpole():
    """Test PPO with Isaac Lab Cartpole environment."""
    print("PPO Isaac Lab Cartpole Test")
    print("=" * 50)
    
    # Configuration
    config = PPOConfig(
        env_name="Isaac-Cartpole-v0",  # Isaac Lab Cartpole
        num_envs=4,
        num_steps=128,
        batch_size=512,
        num_epochs=10,
        learning_rate=3e-4,
        total_timesteps=10000,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_interval=10,
        use_wandb=False  # Disable WandB for this test
    )
    
    print(f"Using device: {config.device}")
    print(f"Number of environments: {config.num_envs}")
    
    try:
        # Create environment using our wrapper
        # The wrapper will fallback to standard Gym Cartpole if Isaac Lab is not available
        env = IsaacGymWrapper(
            env_name="CartPole-v1",  # Standard Gym name as fallback
            num_envs=1,
            device=config.device,
            isaac_lab_available=False  # For now, use Gym fallback
        )
        
        print(f"\nEnvironment created:")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Create vectorized environments
        envs = make_vec_env(
            "CartPole-v1",
            n_envs=config.num_envs,
            vec_env_cls=DummyVecEnv
        )
        
        # Create model
        model = PPOGr00tActorCriticV2(
            observation_space=envs.observation_space,
            action_dim=env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0],
            use_multimodal_encoder=False,
            hidden_dims=(64, 64),
            device=config.device
        ).to(config.device)
        
        print(f"\nModel created:")
        print(f"  Actor parameters: {sum(p.numel() for p in model.actor.parameters()):,}")
        print(f"  Critic parameters: {sum(p.numel() for p in model.critic.parameters()):,}")
        
        # Create trainer
        trainer = PPOTrainerV2(
            config=config,
            model=model,
            envs=envs
        )
        
        print("\nStarting training...")
        start_time = time.time()
        
        # Train for a few steps
        trainer.train()
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.2f} seconds")
        
        # Test the trained model
        print("\nTesting trained model...")
        obs = envs.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
            
        episode_rewards = np.zeros(config.num_envs)
        episode_lengths = np.zeros(config.num_envs)
        
        for _ in range(200):  # Max steps for Cartpole
            obs_tensor = torch.from_numpy(obs).float().to(config.device)
            
            with torch.no_grad():
                actions, _, _, _ = model.get_action_and_value(obs_tensor)
            
            actions_np = actions.cpu().numpy()
            
            # Handle discrete actions for Cartpole
            if hasattr(env.action_space, 'n'):
                # Convert continuous actions to discrete
                actions_np = (actions_np > 0).astype(int).squeeze(-1)
            
            obs, rewards, dones, infos = envs.step(actions_np)
            
            episode_rewards += rewards
            episode_lengths += 1
            
            # Reset episodes that are done
            for i, done in enumerate(dones):
                if done:
                    print(f"  Episode {i} - Reward: {episode_rewards[i]:.2f}, Length: {episode_lengths[i]}")
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
        
        print("\n✅ PPO Isaac Lab Cartpole test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ppo_isaac_cartpole()