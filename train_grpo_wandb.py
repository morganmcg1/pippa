#!/usr/bin/env python3
"""
Dr GRPO Training Script with WandB integration
This is a simple wrapper that ensures WandB tracking is enabled
"""

from train_grpo import main, ExperimentConfig

if __name__ == "__main__":
    # Create config with wandb tracking enabled
    # The main script will handle wandb initialization using .env variables
    config = ExperimentConfig(
        exp_name="grpo_dr_math_qwen",
        model_name="Qwen/Qwen2-0.5B-Instruct",
        track=True,  # This enables WandB tracking
    )
    
    # Run training
    main(config)