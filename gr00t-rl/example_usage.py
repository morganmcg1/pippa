#!/usr/bin/env python3
"""
Example usage of GR00T RL implementations.
Shows how to use PPO and GRPO for robot training.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from configs.ppo_config import PPOConfig
from configs.grpo_config import GRPOConfig


def example_ppo_config():
    """Example PPO configuration for pick and place task."""
    config = PPOConfig(
        # Model settings
        model_path="nvidia/GR00T-N1.5-3B",
        embodiment_tag="GR1",
        freeze_backbone=True,  # Start with frozen backbone
        
        # Environment
        env_name="FrankaCubeLift-v0",
        num_envs=4,  # Use 4 parallel environments
        
        # PPO hyperparameters
        learning_rate=3e-4,
        n_steps=512,  # Shorter for faster updates
        batch_size=32,
        n_epochs=5,
        
        # Training
        total_timesteps=100_000,  # Start with shorter training
        
        # Experiment
        exp_name="ppo_gr00t_cubelift_frozen",
        use_wandb=True
    )
    
    print("PPO Configuration:")
    print(f"  Model: {config.model_path}")
    print(f"  Task: {config.env_name}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print()
    
    return config


def example_grpo_config():
    """Example GRPO configuration for push task."""
    config = GRPOConfig(
        # Model settings
        model_path="nvidia/GR00T-N1.5-3B",
        embodiment_tag="GR1",
        freeze_backbone=True,
        
        # Environment
        env_name="FrankaPush-v0",
        
        # GRPO specific
        num_rollouts_per_update=8,
        learning_rate=5e-5,  # Lower than PPO
        temperature=1.0,
        beta=0.0,  # No KL penalty initially
        
        # Training
        total_episodes=1000,
        update_interval=8,
        
        # Experiment
        exp_name="grpo_gr00t_push_frozen",
        use_wandb=True
    )
    
    print("GRPO Configuration:")
    print(f"  Model: {config.model_path}")
    print(f"  Task: {config.env_name}")
    print(f"  Rollouts per update: {config.num_rollouts_per_update}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Total episodes: {config.total_episodes}")
    print()
    
    return config


def progressive_training_example():
    """Example of progressive unfreezing strategy."""
    print("Progressive Training Strategy:")
    print("1. Start with frozen backbone (fastest)")
    print("2. After initial success, unfreeze last transformer layers")
    print("3. Finally, unfreeze entire model for fine-tuning")
    print()
    
    # Stage 1: Frozen backbone
    stage1_config = PPOConfig(
        freeze_backbone=True,
        freeze_action_head=False,
        exp_name="ppo_gr00t_stage1_frozen",
        total_timesteps=50_000
    )
    
    # Stage 2: Partial unfreezing
    stage2_config = PPOConfig(
        freeze_backbone=False,  # Will need custom logic to freeze only early layers
        freeze_action_head=False,
        learning_rate=1e-4,  # Lower LR for fine-tuning
        exp_name="ppo_gr00t_stage2_partial",
        total_timesteps=100_000
    )
    
    # Stage 3: Full fine-tuning
    stage3_config = PPOConfig(
        freeze_backbone=False,
        freeze_action_head=False,
        learning_rate=5e-5,  # Even lower LR
        exp_name="ppo_gr00t_stage3_full",
        total_timesteps=200_000
    )
    
    return stage1_config, stage2_config, stage3_config


def grpo_vs_ppo_comparison():
    """Compare GRPO and PPO for same task."""
    print("GRPO vs PPO Comparison:")
    print()
    
    # PPO config
    ppo_config = PPOConfig(
        env_name="FrankaStack-v0",
        exp_name="comparison_ppo_stack",
        total_timesteps=500_000
    )
    
    # GRPO config (equivalent training)
    # Approximate timesteps: 500k steps / 200 steps per episode = 2500 episodes
    grpo_config = GRPOConfig(
        env_name="FrankaStack-v0",
        exp_name="comparison_grpo_stack",
        total_episodes=2500,
        num_rollouts_per_update=16,  # More rollouts for better baseline
        beta=0.1  # Small KL penalty for stability
    )
    
    print("PPO advantages:")
    print("  - Value function provides better credit assignment")
    print("  - More sample efficient for long episodes")
    print("  - Well-tested in robotics")
    print()
    
    print("GRPO advantages:")
    print("  - No critic network (saves memory)")
    print("  - Better for verifiable rewards")
    print("  - Simpler to implement and debug")
    print()
    
    return ppo_config, grpo_config


if __name__ == "__main__":
    print("GR00T RL Training Examples")
    print("=" * 50)
    print()
    
    # Show different configuration examples
    ppo_config = example_ppo_config()
    grpo_config = example_grpo_config()
    
    # Show progressive training
    stages = progressive_training_example()
    
    # Show comparison
    ppo_comp, grpo_comp = grpo_vs_ppo_comparison()
    
    print("To run training:")
    print("  1. Run setup: ./setup.sh")
    print("  2. Activate environment: source .venv/bin/activate")
    print("  3. Run training: python scripts/train_ppo.py")
    print()
    
    print("Note: The current implementation uses mock Isaac Lab environments.")
    print("For real training, you'll need to:")
    print("  1. Install Isaac Lab")
    print("  2. Update environment wrapper to use real Isaac Lab APIs")
    print("  3. Implement proper action/observation transformations")