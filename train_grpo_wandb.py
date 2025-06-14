#!/usr/bin/env python3
"""
Dr GRPO Training Script with WandB integration
"""

import os
from dotenv import load_dotenv
import wandb

# Load environment variables
load_dotenv()

# Import the main training script
from train_grpo import main, ExperimentConfig

if __name__ == "__main__":
    # Initialize wandb with environment variables
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "grpo-experiments"),
        entity=os.getenv("WANDB_ENTITY", None),
        name="grpo_dr_math_qwen",
        config={
            "model": "Qwen/Qwen2-0.5B-Instruct",
            "dataset_size": 12,
            "num_generations": 4,
            "epochs": 3,
        }
    )
    
    # Create config with wandb tracking enabled
    config = ExperimentConfig(
        exp_name="grpo_dr_math_qwen",
        model_name="Qwen/Qwen2-0.5B-Instruct",
        track=True,
    )
    
    # Run training
    main(config)
    
    # Finish wandb run
    wandb.finish()