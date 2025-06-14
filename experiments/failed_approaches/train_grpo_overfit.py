#!/usr/bin/env python3
"""
Dr GRPO Overfitting Experiment
Tests the training pipeline by overfitting on a small subset of GSM8K
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer
import warnings

# Load environment variables
load_dotenv()

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class OverfitExperimentConfig:
    """Configuration for overfitting experiment"""
    # Experiment tracking
    exp_name: str = "grpo_overfit_gsm8k"
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = torch.cuda.is_available()
    track: bool = True
    wandb_project_name: str = os.getenv("WANDB_PROJECT", "pippa")
    wandb_entity: Optional[str] = os.getenv("WANDB_ENTITY", None)
    
    # Model configuration
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    
    # Dataset configuration
    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "train"
    num_samples: int = 100  # Small subset for overfitting
    
    # Training hyperparameters for overfitting
    total_epochs: int = 20  # More epochs to ensure overfitting
    learning_rate: float = 5e-5  # Higher LR for faster overfitting
    per_device_train_batch_size: int = 8  # Larger batch size for better GPU utilization
    gradient_accumulation_steps: int = 2  # Effective batch size = 16
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0  # No regularization for overfitting
    
    # GRPO specific parameters
    num_generations: int = 4
    temperature: float = 0.7
    max_prompt_length: int = 256  # Longer for GSM8K problems
    max_completion_length: int = 128  # Longer for solutions
    beta: float = 0.0  # Dr GRPO (no KL penalty)
    
    # Reward configuration
    correctness_weight: float = 1.0
    length_penalty_weight: float = 0.1  # Lower penalty for GSM8K
    target_response_length: int = 50
    max_response_length: int = 200
    
    # System configuration
    num_workers: int = 0
    output_dir: str = "./grpo_overfit_output"
    logging_steps: int = 1
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.num_generations >= 1, "Must generate at least 1 completion per prompt"
        assert self.temperature > 0, "Temperature must be positive"
        assert self.per_device_train_batch_size > 0, "Batch size must be positive"


def set_random_seeds(seed: int, torch_deterministic: bool = True):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    
    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)


def load_gsm8k_subset(config: OverfitExperimentConfig):
    """Load a small subset of GSM8K for overfitting"""
    # Load the dataset with 'main' config
    dataset = load_dataset(config.dataset_name, 'main', split=config.dataset_split)
    
    # Convert to our format
    data = []
    for i, example in enumerate(dataset.select(range(min(config.num_samples, len(dataset))))):
        # GSM8K format: question and answer
        prompt = example['question']
        # Extract numerical answer from the solution
        solution = example['answer'].split('####')[-1].strip()
        
        data.append({
            "prompt": prompt,
            "solution": solution,
            "full_answer": example['answer']  # Keep full answer for reference
        })
    
    print(f"Loaded {len(data)} examples from GSM8K")
    print(f"Example prompt: {data[0]['prompt'][:100]}...")
    print(f"Example solution: {data[0]['solution']}")
    
    from datasets import Dataset
    return Dataset.from_list(data)


def extract_number_from_text(text: str) -> Optional[str]:
    """Extract the final numerical answer from model output"""
    import re
    
    # Look for patterns like "#### 42" or "answer is 42" or just numbers at the end
    patterns = [
        r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',  # GSM8K format
        r'answer\s*(?:is|:)?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',  # "answer is X"
        r'=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$',  # "= X" at end
        r'(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$',  # Just number at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Remove commas from numbers
            return match.group(1).replace(',', '')
    
    return None


def gsm8k_correctness_reward(completions: List[str], 
                            prompts: Optional[List[str]] = None, 
                            solutions: Optional[List[str]] = None, 
                            **kwargs) -> List[float]:
    """
    Reward function for GSM8K mathematical correctness
    """
    if solutions is None:
        raise ValueError("Solutions must be provided for correctness reward")
    
    rewards = []
    for completion, solution in zip(completions, solutions):
        # Extract numerical answer from completion
        extracted = extract_number_from_text(completion)
        
        if extracted and extracted == solution:
            rewards.append(1.0)
        else:
            # Partial credit if number is mentioned anywhere
            if solution in completion:
                rewards.append(0.5)
            else:
                rewards.append(-1.0)
    
    return rewards


def length_penalty_reward(completions: List[str], 
                         config: Optional[OverfitExperimentConfig] = None,
                         **kwargs) -> List[float]:
    """
    Soft penalty for response length (lighter for GSM8K)
    """
    if config is None:
        target_length = 50
        max_length = 200
    else:
        target_length = config.target_response_length
        max_length = config.max_response_length
    
    rewards = []
    for completion in completions:
        length = len(completion.strip())
        
        if length <= target_length:
            rewards.append(0.0)
        elif length <= max_length:
            # Very light penalty for GSM8K
            penalty = -0.1 * ((length - target_length) / (max_length - target_length))
            rewards.append(penalty)
        else:
            rewards.append(-0.1)
    
    return rewards


def log_training_info(config: OverfitExperimentConfig, dataset_size: int):
    """Log training configuration"""
    print("=" * 60)
    print("Dr GRPO Overfitting Experiment - GSM8K")
    print("=" * 60)
    print(f"Experiment: {config.exp_name}")
    print(f"Seed: {config.seed}")
    print(f"Device: {'cuda' if config.cuda else 'cpu'}")
    print(f"\nModel: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Dataset size: {dataset_size} samples")
    print(f"\nTraining hyperparameters:")
    print(f"  - Epochs: {config.total_epochs}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Batch size: {config.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"\nGRPO parameters:")
    print(f"  - Generations per prompt: {config.num_generations}")
    print(f"  - Temperature: {config.temperature}")
    print(f"  - Max prompt length: {config.max_prompt_length}")
    print(f"  - Max completion length: {config.max_completion_length}")
    print("=" * 60)


def main(config: Optional[OverfitExperimentConfig] = None):
    """Main training function for overfitting experiment"""
    if config is None:
        config = OverfitExperimentConfig()
    
    # Set random seeds
    set_random_seeds(config.seed, config.torch_deterministic)
    
    # Initialize WandB if tracking
    if config.track:
        import wandb
        
        os.environ["WANDB_PROJECT"] = config.wandb_project_name
        if config.wandb_entity:
            os.environ["WANDB_ENTITY"] = config.wandb_entity
        
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            name=f"{config.exp_name}_e{config.total_epochs}_b{config.per_device_train_batch_size}_seed{config.seed}",
            config={
                "model": config.model_name,
                "dataset": config.dataset_name,
                "dataset_size": config.num_samples,
                "epochs": config.total_epochs,
                "batch_size": config.per_device_train_batch_size,
                "gradient_accumulation": config.gradient_accumulation_steps,
                "effective_batch_size": config.per_device_train_batch_size * config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "num_generations": config.num_generations,
                "temperature": config.temperature,
                "seed": config.seed,
            },
            reinit=True,
        )
    
    # Load dataset
    dataset = load_gsm8k_subset(config)
    
    # Log configuration
    log_training_info(config, len(dataset))
    
    # Start timing
    start_time = time.time()
    
    try:
        # Create TRL GRPO configuration
        training_args = GRPOConfig(
            output_dir=config.output_dir,
            
            # Training parameters
            learning_rate=config.learning_rate,
            num_train_epochs=config.total_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            
            # Generation parameters
            num_generations=config.num_generations,
            max_completion_length=config.max_completion_length,
            max_prompt_length=config.max_prompt_length,
            temperature=config.temperature,
            
            # Dr GRPO specific
            beta=config.beta,
            remove_unused_columns=False,
            
            # Optimization settings
            bf16=config.cuda and torch.cuda.is_bf16_supported(),
            fp16=False,
            dataloader_num_workers=config.num_workers,
            
            # Logging and saving
            logging_steps=config.logging_steps,
            save_strategy=config.save_strategy,
            save_total_limit=config.save_total_limit,
            report_to=["wandb"] if config.track else [],
            
            # Experiment tracking
            run_name=f"{config.exp_name}_seed{config.seed}",
            
            # Reward weights
            reward_weights=[config.correctness_weight, config.length_penalty_weight],
        )
        
        # Create reward functions
        def correctness_reward_wrapped(completions, prompts=None, **kwargs):
            solutions = kwargs.get('solution', None)
            return gsm8k_correctness_reward(completions, prompts=prompts, solutions=solutions)
        
        def length_penalty_wrapped(completions, **kwargs):
            return length_penalty_reward(completions, config=config, **kwargs)
        
        # Initialize trainer
        trainer = GRPOTrainer(
            model=config.model_name,
            reward_funcs=[correctness_reward_wrapped, length_penalty_wrapped],
            args=training_args,
            train_dataset=dataset,
        )
        
        print("\nTrainer initialized successfully!")
        print(f"Total training steps: {len(dataset) * config.total_epochs // (config.per_device_train_batch_size * config.gradient_accumulation_steps)}")
        
        # Start training
        print("\nStarting GRPO overfitting experiment...")
        train_result = trainer.train()
        print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
        
        # Save the final model
        final_model_path = os.path.join(config.output_dir, "final_model")
        trainer.save_model(final_model_path)
        print(f"Model saved to: {final_model_path}")
        
        # Log final metrics
        if train_result is not None:
            print("\nFinal training metrics:")
            for key, value in train_result.metrics.items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    
    finally:
        print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")
        
        if config.track:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    # Run with default configuration
    main()
    
    # Example: Run with custom configuration for more aggressive overfitting
    # custom_config = OverfitExperimentConfig(
    #     num_samples=50,  # Even smaller dataset
    #     total_epochs=50,  # More epochs
    #     learning_rate=1e-4,  # Higher learning rate
    #     per_device_train_batch_size=16,  # Larger batch size
    # )
    # main(custom_config)