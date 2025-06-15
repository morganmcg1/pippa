#!/usr/bin/env python3
"""
Save GR00T SFT checkpoints and final model to WandB
"""

import os
import sys
from pathlib import Path
import wandb
from dotenv import load_dotenv
import argparse
from datetime import datetime


def save_checkpoint_to_wandb(checkpoint_dir: str, run_name: str = None):
    """Save model checkpoint directory to WandB."""
    # Load environment variables
    load_dotenv(os.path.expanduser("~/pippa/.env"))
    
    # Initialize WandB
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"gr00t-sft-upload-{timestamp}"
    
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=run_name,
        tags=["gr00t-sft", "model-upload"],
        job_type="upload",
    )
    
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist!")
        return
    
    # Create artifact for the entire checkpoint directory
    artifact_name = f"gr00t-sft-{checkpoint_path.name}"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=f"GR00T SFT checkpoint from {checkpoint_path}",
    )
    
    # Add all files in the checkpoint directory
    print(f"Adding files from {checkpoint_path}...")
    artifact.add_dir(str(checkpoint_path))
    
    # Log the artifact
    wandb.log_artifact(artifact)
    print(f"Successfully saved {artifact_name} to WandB!")
    
    # Also log individual checkpoints if they exist
    checkpoint_dirs = list(checkpoint_path.glob("checkpoint-*"))
    if checkpoint_dirs:
        print(f"\nFound {len(checkpoint_dirs)} checkpoint directories")
        for ckpt_dir in checkpoint_dirs:
            step = ckpt_dir.name.split("-")[-1]
            ckpt_artifact = wandb.Artifact(
                name=f"gr00t-sft-checkpoint-step-{step}",
                type="model",
                description=f"GR00T SFT checkpoint at step {step}",
            )
            ckpt_artifact.add_dir(str(ckpt_dir))
            wandb.log_artifact(ckpt_artifact)
            print(f"  - Saved checkpoint at step {step}")
    
    # Log final checkpoint if it exists
    final_checkpoint = checkpoint_path / "checkpoint-10000"
    if final_checkpoint.exists():
        final_artifact = wandb.Artifact(
            name="gr00t-sft-final-model",
            type="model",
            description="Final GR00T SFT model after 10k steps",
        )
        final_artifact.add_dir(str(final_checkpoint))
        wandb.log_artifact(final_artifact)
        print("\nSaved final model checkpoint!")
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save GR00T checkpoints to WandB")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./so101-checkpoints",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="WandB run name",
    )
    
    args = parser.parse_args()
    save_checkpoint_to_wandb(args.checkpoint_dir, args.run_name)