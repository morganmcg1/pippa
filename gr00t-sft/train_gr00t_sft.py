#!/usr/bin/env python3
"""
GR00T SFT Training Script with Enhanced WandB Logging
Based on the official gr00t_finetune.py with modifications for:
1. WandB integration with custom tags
2. Model checkpoints saved to WandB
3. Detailed metric logging
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal
from datetime import datetime

import torch
import tyro
from transformers import TrainingArguments
import wandb
from dotenv import load_dotenv

# Add the Isaac-GR00T directory to Python path
sys.path.insert(0, os.path.expanduser("~/pippa/Isaac-GR00T"))

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning with WandB support."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories"""

    output_dir: str = "./gr00t-sft-checkpoints"
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "so100_dualcam"
    """Data configuration name from DATA_CONFIG_MAP"""
    
    # Data sampling
    max_samples: int = -1
    """Maximum number of samples to use from dataset (-1 for all)"""
    
    percentage: float = 1.0
    """Percentage of dataset to use (0.0 to 1.0)"""

    # Training parameters
    max_steps: int = 10000
    """Maximum number of training steps"""

    batch_size: int = 1
    """Per device training batch size"""

    learning_rate: float = 1e-4
    """Learning rate for AdamW optimizer"""

    warmup_ratio: float = 0.05
    """Warmup ratio for learning rate scheduler"""

    weight_decay: float = 0.05
    """Weight decay for AdamW optimizer"""

    save_steps: int = 1000
    """Save checkpoint every N steps"""

    # Model configuration
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Base model path or HuggingFace model ID"""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag for the robot"""

    # Fine-tuning flags
    tune_llm: bool = False
    """Whether to fine-tune the LLM backbone"""

    tune_visual: bool = False
    """Whether to fine-tune the vision encoder"""

    tune_projector: bool = True
    """Whether to fine-tune the action head projector"""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model"""

    # LoRA configuration
    lora_rank: int = 0
    """LoRA rank (0 means no LoRA)"""

    lora_alpha: int = 16
    """LoRA alpha parameter"""

    lora_dropout: float = 0.0
    """LoRA dropout rate"""

    lora_full_model: bool = False
    """Apply LoRA to full model vs action head only"""

    # Data loading
    balance_dataset_weights: bool = True
    """Balance dataset weights for mixture dataset"""

    balance_trajectory_weights: bool = False
    """Balance trajectory weights within datasets"""

    dataloader_num_workers: int = 4
    """Number of dataloader workers"""

    # Multi-GPU training
    num_gpus: int = 1
    """Number of GPUs to use for training"""

    # Logging
    report_to: Literal["wandb", "tensorboard"] = "wandb"
    """Reporting backend"""

    # Resume training
    resume: bool = False
    """Resume from latest checkpoint"""

    # Video backend
    video_backend: Literal["pyav", "torchvision_video", "torchvision_av"] = "torchvision_av"
    """Video decoding backend"""

    # WandB configuration
    wandb_project: str = "pippa"
    """WandB project name"""

    wandb_entity: str = "wild-ai"
    """WandB entity name"""

    wandb_tags: List[str] = None
    """WandB tags for the run"""

    wandb_run_name: str = None
    """Custom WandB run name"""


def setup_wandb(config: ArgsConfig):
    """Setup WandB with custom configuration."""
    # Load environment variables
    load_dotenv(os.path.expanduser("~/pippa/.env"))
    
    # Generate run name if not provided
    if config.wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.wandb_run_name = f"gr00t-sft-{config.data_config}-{timestamp}"
    
    # Set default tags
    if config.wandb_tags is None:
        config.wandb_tags = ["gr00t-sft"]
    else:
        # Ensure gr00t-sft tag is included
        if "gr00t-sft" not in config.wandb_tags:
            config.wandb_tags.append("gr00t-sft")
    
    # Initialize WandB
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        tags=config.wandb_tags,
        config=vars(config),
        save_code=True,
    )
    
    # Log the configuration
    wandb.config.update({
        "model_type": "GR00T-N1.5",
        "task": "supervised_fine_tuning",
        "embodiment": config.embodiment_tag,
        "data_config": config.data_config,
        "training_steps": config.max_steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "lora_enabled": config.lora_rank > 0,
        "tune_vision": config.tune_visual,
        "tune_language": config.tune_llm,
        "tune_diffusion": config.tune_diffusion_model,
    })


def save_checkpoint_to_wandb(checkpoint_dir: str, step: int):
    """Save model checkpoint to WandB."""
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists():
        # Create artifact for the checkpoint
        artifact = wandb.Artifact(
            name=f"gr00t-sft-checkpoint-step-{step}",
            type="model",
            description=f"GR00T SFT checkpoint at step {step}",
        )
        
        # Add all files in the checkpoint directory
        artifact.add_dir(str(checkpoint_path))
        
        # Log the artifact
        wandb.log_artifact(artifact)
        print(f"Saved checkpoint at step {step} to WandB")


def main(config: ArgsConfig):
    """Main training function."""
    # Setup WandB
    if config.report_to == "wandb":
        setup_wandb(config)
    
    # Set video backend
    os.environ["VIDEO_BACKEND"] = config.video_backend

    # Print configuration
    print("\n" + "=" * 50)
    print("GR00T SFT TRAINING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    # Validate GPU configuration
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, f"Requested {config.num_gpus} GPUs but only {available_gpus} available"

    # Get embodiment tag
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    
    # Get modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    
    # Load dataset
    if len(config.dataset_path) == 1:
        # Load single dataset
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
        )
        
        # Note: max_samples feature not yet implemented for LeRobotSingleDataset
        if config.max_samples > 0:
            print(f"Warning: max_samples={config.max_samples} specified but feature not yet implemented")
            print(f"Dataset has {len(train_dataset)} samples total")
        
        print(f"Loaded single dataset from {config.dataset_path[0]} with {len(train_dataset)} samples")
    else:
        single_datasets = []
        for dataset_path in config.dataset_path:
            dataset = LeRobotSingleDataset(
                dataset_path=dataset_path,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            
            # Note: max_samples feature not yet implemented
                
            single_datasets.append(dataset)
        
        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # Equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets with total {sum(len(d) for d in single_datasets)} samples")

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )

    # Set compute dtype
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # Apply LoRA if configured
    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )
        print(f"Applied LoRA with rank {config.lora_rank}")

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.wandb_run_name if config.report_to == "wandb" else None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # Create trainer
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # Add callback to save checkpoints to WandB
    if config.report_to == "wandb":
        class WandBCheckpointCallback:
            def on_save(self, args, state, control, **kwargs):
                if state.global_step % config.save_steps == 0:
                    checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{state.global_step}")
                    save_checkpoint_to_wandb(checkpoint_dir, state.global_step)
        
        # Note: The actual callback integration depends on the TrainRunner implementation
        # This is a placeholder for the callback pattern

    # Train the model
    print("\nStarting training...")
    experiment.train()

    # Save final model to WandB
    if config.report_to == "wandb":
        print("\nSaving final model to WandB...")
        final_checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{config.max_steps}")
        if os.path.exists(final_checkpoint_dir):
            save_checkpoint_to_wandb(final_checkpoint_dir, config.max_steps)
        
        # Also save the entire output directory as an artifact
        final_artifact = wandb.Artifact(
            name="gr00t-sft-final-model",
            type="model",
            description="Final GR00T SFT model and configuration",
        )
        final_artifact.add_dir(config.output_dir)
        wandb.log_artifact(final_artifact)
        
        # Finish WandB run
        wandb.finish()

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    # Parse arguments
    config = tyro.cli(ArgsConfig)
    
    # Check if we're already in a distributed process
    if "LOCAL_RANK" in os.environ:
        # We're already in a distributed process, just run main
        main(config)
    elif config.num_gpus > 1:
        # Use torchrun for multi-GPU training
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={config.num_gpus}",
            "--nnodes=1",
            "--rdzv_backend=c10d",
            __file__,
        ] + sys.argv[1:]
        
        print(f"Launching multi-GPU training with {config.num_gpus} GPUs...")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        # Single GPU training
        main(config)