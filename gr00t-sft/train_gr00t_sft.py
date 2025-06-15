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


class SubsetLeRobotSingleDataset(LeRobotSingleDataset):
    """Wrapper to create a subset of LeRobotSingleDataset while maintaining the same interface."""
    
    def __init__(self, base_dataset: LeRobotSingleDataset, max_samples: int):
        """Initialize subset wrapper without calling parent __init__."""
        # Copy all attributes from base dataset
        self.__dict__.update(base_dataset.__dict__.copy())
        
        # Store original length
        self._original_length = len(base_dataset)
        
        # Create subset indices
        self._subset_indices = list(range(min(max_samples, self._original_length)))
        self._subset_length = len(self._subset_indices)
        
        print(f"Created subset: {self._subset_length} samples from original {self._original_length}")
    
    def __len__(self):
        """Return subset length."""
        return self._subset_length
    
    def __getitem__(self, idx):
        """Get item from subset."""
        if idx >= self._subset_length:
            raise IndexError(f"Index {idx} out of range for subset of size {self._subset_length}")
        
        # Map subset index to original index
        original_idx = self._subset_indices[idx]
        
        # Call parent's getitem with original index
        return super().__getitem__(original_idx)


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


def save_checkpoint_to_wandb(checkpoint_dir: str, step: int, model=None, config=None, include_optimizer=False):
    """Save model checkpoint to WandB with only essential components for inference.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        step: Training step number
        model: Model instance (optional, for saving action head separately)
        config: Training configuration
        include_optimizer: Whether to include optimizer state (default: False)
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint directory {checkpoint_dir} does not exist")
        return
        
    if wandb.run is None:
        print("Warning: No active WandB run")
        return
    
    try:
        # Create a consistent artifact name based on run characteristics
        batch_size = config.batch_size if config else "unknown"
        dataset_name = config.data_config if config else "unknown"
        max_samples = config.max_samples if config and config.max_samples > 0 else "full"
        
        # Create artifact name without step
        artifact_name = f"gr00t-sft-{dataset_name}-bs{batch_size}"
        if max_samples != "full":
            artifact_name += f"-{max_samples}samples"
        
        # Create artifact for the checkpoint
        artifact = wandb.Artifact(
            name=artifact_name,
            type="gr00t-model",
            description=f"GR00T SFT model - batch_size={batch_size}, dataset={dataset_name}, samples={max_samples}, step={step}",
        )
        
        print(f"Creating WandB artifact '{artifact_name}' for step {step}...")
        
        # Add only essential files for inference (skip optimizer.pt to save space)
        essential_files = [
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors", 
            "model.safetensors.index.json",
            "trainer_state.json"
        ]
        
        files_added = 0
        total_size = 0
        
        for filename in essential_files:
            filepath = checkpoint_path / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  Adding {filename} ({size_mb:.1f} MB)")
                artifact.add_file(str(filepath))
                files_added += 1
                total_size += size_mb
        
        # Optionally include optimizer (for resuming training)
        if include_optimizer:
            optimizer_path = checkpoint_path / "optimizer.pt"
            if optimizer_path.exists():
                size_mb = optimizer_path.stat().st_size / (1024 * 1024)
                print(f"  Adding optimizer.pt ({size_mb:.1f} MB)")
                artifact.add_file(str(optimizer_path))
                files_added += 1
                total_size += size_mb
        
        # Add experiment config directory if it exists
        exp_cfg_dir = checkpoint_path / "experiment_cfg"
        if exp_cfg_dir.exists() and exp_cfg_dir.is_dir():
            print(f"  Adding experiment_cfg directory")
            artifact.add_dir(str(exp_cfg_dir))
        
        # Save and add action head weights separately (small file with just trainable params)
        if model is not None:
            action_head_path = checkpoint_path / "action_head_new_embodiment.pt"
            action_head_state = {}
            for name, param in model.named_parameters():
                if param.requires_grad and "action_head" in name:
                    action_head_state[name] = param.data.cpu()  # Move to CPU to save
            
            if action_head_state:
                torch.save(action_head_state, action_head_path)
                size_mb = action_head_path.stat().st_size / (1024 * 1024)
                print(f"  Adding action_head_new_embodiment.pt ({size_mb:.1f} MB) with {len(action_head_state)} parameters")
                artifact.add_file(str(action_head_path))
                files_added += 1
                total_size += size_mb
        
        # Copy and add modality.json
        if config and hasattr(config, 'dataset_path'):
            modality_src = Path(config.dataset_path[0]) / "meta" / "modality.json"
            if modality_src.exists():
                modality_dst = checkpoint_path / "modality.json"
                import shutil
                shutil.copy2(modality_src, modality_dst)
                print(f"  Adding modality.json")
                artifact.add_file(str(modality_dst))
                files_added += 1
        
        print(f"Total: {files_added} files, {total_size:.1f} MB")
        
        # Log the artifact to the current run with step and run ID as aliases
        print(f"Uploading artifact to WandB...")
        aliases = [f"step-{step}", "latest"]
        if wandb.run and wandb.run.id:
            aliases.append(f"run-{wandb.run.id}")
        wandb.run.log_artifact(artifact, aliases=aliases)
        print(f"✓ Successfully uploaded checkpoint '{artifact_name}' at step {step} with aliases: {aliases}")
        
    except Exception as e:
        print(f"Error saving checkpoint to WandB: {e}")
        import traceback
        traceback.print_exc()


def main(config: ArgsConfig):
    """Main training function."""
    wandb_run_active = False
    
    try:
        # Setup WandB
        if config.report_to == "wandb":
            setup_wandb(config)
            wandb_run_active = True
        
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
            
            full_dataset_size = len(train_dataset)
            
            # Apply max_samples if specified
            if config.max_samples > 0 and config.max_samples < full_dataset_size:
                # Use our custom wrapper to maintain type compatibility
                train_dataset = SubsetLeRobotSingleDataset(train_dataset, config.max_samples)
                print(f"Applied max_samples={config.max_samples} - using subset of dataset")
                print(f"Original dataset size: {full_dataset_size}, subset size: {len(train_dataset)}")
            else:
                print(f"Using full dataset with {len(train_dataset)} samples")
            
            print(f"Dataset ready from {config.dataset_path[0]}")
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
                        save_checkpoint_to_wandb(checkpoint_dir, state.global_step, model, config)
            
            # Note: The actual callback integration depends on the TrainRunner implementation
            # This is a placeholder for the callback pattern

        # Train the model
        print("\nStarting training...")
        experiment.train()
        
        # After training, save only the final checkpoint to WandB

        # Save final model to WandB
        if config.report_to == "wandb":
            print("\nSaving final model to WandB...")
            final_checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{config.max_steps}")
            if os.path.exists(final_checkpoint_dir):
                save_checkpoint_to_wandb(final_checkpoint_dir, config.max_steps, model, config)
            else:
                # If exact step checkpoint doesn't exist, use the latest one
                import glob
                checkpoints = glob.glob(os.path.join(config.output_dir, "checkpoint-*"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
                    step = int(latest_checkpoint.split('-')[-1])
                    save_checkpoint_to_wandb(latest_checkpoint, step, model, config)
            
            # Don't upload the entire output directory - it's too large (115GB+)
            # The final checkpoint has already been uploaded above
            
        print("\nTraining completed successfully!")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always finish WandB run if it was started
        if wandb_run_active and wandb.run is not None:
            print("\nFinishing WandB run...")
            try:
                wandb.finish()
                print("WandB run finished successfully")
            except Exception as e:
                print(f"Error finishing WandB run: {e}")


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