#!/usr/bin/env python3
"""
Dr GRPO (Group Relative Policy Optimization Done Right) Training Script
A minimalistic implementation using Hugging Face TRL for testing and verification

Following best practices from cleanRL and stable-baselines3:
- Single-file implementation with all details visible
- Reproducibility through proper seed management
- Clear hyperparameter documentation
- Proper normalization and preprocessing
- Extensive logging and monitoring
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer
import warnings

def evaluate_on_standard_dataset(model, tokenizer, device) -> Dict[str, float]:
    """
    Evaluate model on the standardized arithmetic evaluation dataset.
    This ensures fair comparison across all experiments.
    """
    print("\n" + "="*60)
    print("STANDARDIZED EVALUATION (morgan/arithmetic_eval)")
    print("="*60)
    
    # Load standardized evaluation dataset
    from datasets import load_dataset
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    print(f"Loaded {len(eval_dataset)} evaluation samples")
    
    # Initialize results tracking
    results_by_difficulty = {}
    results_by_operation = {}
    correct_total = 0
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(eval_dataset):
            if i % 50 == 0 and i > 0:
                print(f"Progress: {i}/{len(eval_dataset)}")
            
            prompt = sample['prompt']
            expected = sample['answer']
            difficulty = sample['difficulty']
            operation = sample['metadata']['operation'] if sample['metadata'] else None
            
            # Generate model answer
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer (reuse existing extract_answer function if available)
            completion = response[len(prompt):].strip()
            match = re.match(r'^-?\d+', completion)
            predicted = match.group(0) if match else completion.split()[0] if completion else ""
            
            # Check correctness
            is_correct = predicted == expected
            if is_correct:
                correct_total += 1
            
            # Track by difficulty
            if difficulty not in results_by_difficulty:
                results_by_difficulty[difficulty] = {'correct': 0, 'total': 0}
            results_by_difficulty[difficulty]['total'] += 1
            if is_correct:
                results_by_difficulty[difficulty]['correct'] += 1
            
            # Track by operation
            if operation:
                if operation not in results_by_operation:
                    results_by_operation[operation] = {'correct': 0, 'total': 0}
                results_by_operation[operation]['total'] += 1
                if is_correct:
                    results_by_operation[operation]['correct'] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct_total / len(eval_dataset)
    
    # Display results
    print(f"\nOverall accuracy: {overall_accuracy:.2%} ({correct_total}/{len(eval_dataset)})")
    
    print("\nAccuracy by difficulty:")
    difficulty_accs = {}
    for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        if diff in results_by_difficulty:
            stats = results_by_difficulty[diff]
            acc = stats['correct'] / stats['total']
            difficulty_accs[f"eval_{diff}_accuracy"] = acc
            print(f"  {diff}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nAccuracy by operation:")
    operation_accs = {}
    for op in ['+', '-', '*', '/']:
        if op in results_by_operation:
            stats = results_by_operation[op]
            acc = stats['correct'] / stats['total']
            op_name = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op]
            operation_accs[f"eval_{op_name}_accuracy"] = acc
            print(f"  {op}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("="*60)
    
    # Return all metrics for WandB logging
    return {
        'arithmetic_eval': overall_accuracy,  # Primary metric
        'arithmetic_eval_correct': correct_total,
        'arithmetic_eval_total': len(eval_dataset),
        **difficulty_accs,
        **operation_accs
    }


# Load environment variables
load_dotenv()

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class ExperimentConfig:
    """Configuration for GRPO experiment following cleanRL style"""
    # Experiment tracking
    exp_name: str = "grpo_dr_math"
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = torch.cuda.is_available()
    track: bool = True
    wandb_project_name: str = os.getenv("WANDB_PROJECT", "grpo-experiments")
    wandb_entity: Optional[str] = os.getenv("WANDB_ENTITY", None)
    
    # Model configuration
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"  # 0.5B params for testing
    
    # Training hyperparameters
    total_epochs: int = 3
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    
    # GRPO specific parameters
    num_generations: int = 4  # G in the paper
    temperature: float = 0.7
    max_prompt_length: int = 64
    max_completion_length: int = 50
    beta: float = 0.0  # KL penalty coefficient (0 for Dr GRPO)
    
    # Reward configuration
    correctness_weight: float = 1.0
    length_penalty_weight: float = 0.3
    target_response_length: int = 20
    max_response_length: int = 100
    
    # System configuration
    num_workers: int = 0  # Set to 0 to avoid worker process issues
    output_dir: str = "./grpo_test_output"
    logging_steps: int = 1
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.num_generations >= 1, "Must generate at least 1 completion per prompt"
        assert self.temperature > 0, "Temperature must be positive"
        assert self.per_device_train_batch_size > 0, "Batch size must be positive"


def set_random_seeds(seed: int, torch_deterministic: bool = True):
    """
    Set random seeds for reproducibility following stable-baselines3 practices
    
    Args:
        seed: Random seed to use
        torch_deterministic: Whether to use deterministic algorithms in PyTorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # Transformers seed setting
    
    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)


def create_minimal_dataset():
    """Create a small math reasoning dataset for testing"""
    data = [
        {"prompt": "What is 2 + 2?", "solution": "4"},
        {"prompt": "What is 5 + 3?", "solution": "8"},
        {"prompt": "What is 10 - 4?", "solution": "6"},
        {"prompt": "What is 3 * 4?", "solution": "12"},
        {"prompt": "What is 15 / 3?", "solution": "5"},
        {"prompt": "What is 7 + 8?", "solution": "15"},
        {"prompt": "What is 20 - 12?", "solution": "8"},
        {"prompt": "What is 6 * 7?", "solution": "42"},
        {"prompt": "What is 9 * 9?", "solution": "81"},
        {"prompt": "What is 100 / 10?", "solution": "10"},
        {"prompt": "What is 25 + 25?", "solution": "50"},
        {"prompt": "What is 16 - 9?", "solution": "7"},
    ]
    return Dataset.from_list(data)


def correctness_reward(completions: List[str], 
                      prompts: Optional[List[str]] = None, 
                      solutions: Optional[List[str]] = None, 
                      **kwargs) -> List[float]:
    """
    Binary reward function for mathematical correctness
    
    Following RL best practices:
    - Clear reward signal (binary: correct/incorrect)
    - No implicit biases
    - Robust number extraction
    
    Returns:
        List of rewards: 1.0 for correct, -1.0 for incorrect
    """
    import re
    
    if solutions is None:
        raise ValueError("Solutions must be provided for correctness reward")
    
    rewards = []
    for completion, solution in zip(completions, solutions):
        # Extract all numbers from completion (handles various formats)
        numbers = re.findall(r'-?\d+\.?\d*', completion)
        
        # Check if the exact solution appears
        if solution in numbers:
            rewards.append(1.0)
        else:
            # Also check if solution appears as a substring (for decimal answers)
            if solution in completion:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
    
    return rewards


def length_penalty_reward(completions: List[str], 
                         config: Optional[ExperimentConfig] = None,
                         **kwargs) -> List[float]:
    """
    Soft penalty for response length to encourage conciseness
    
    Note: Dr GRPO specifically addresses length bias issues in standard GRPO.
    This penalty is optional and should be used carefully.
    
    Returns:
        List of penalties: 0.0 (no penalty) to -0.5 (max penalty)
    """
    if config is None:
        target_length = 20
        max_length = 100
    else:
        target_length = config.target_response_length
        max_length = config.max_response_length
    
    rewards = []
    for completion in completions:
        length = len(completion.strip())  # Strip whitespace
        
        if length <= target_length:
            rewards.append(0.0)  # No penalty for concise responses
        elif length <= max_length:
            # Smooth penalty curve
            penalty = -0.5 * ((length - target_length) / (max_length - target_length)) ** 2
            rewards.append(penalty)
        else:
            rewards.append(-0.5)  # Maximum penalty
    
    return rewards


def compute_advantage_dr_grpo(rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute advantages using Dr GRPO method (unbiased version)
    
    Key insight: Removes the standard deviation normalization that
    introduces bias in favor of questions with high variance in correctness
    
    Args:
        rewards: Tensor of shape (batch_size, num_generations)
    
    Returns:
        advantages: Tensor of same shape with zero-mean advantages per prompt
    """
    # Compute mean reward for each prompt (baseline)
    baseline = rewards.mean(dim=1, keepdim=True)
    
    # Advantages are simply rewards minus baseline
    # No std normalization as in standard GRPO
    advantages = rewards - baseline
    
    return advantages


def log_training_info(config: ExperimentConfig, dataset_size: int):
    """Log training configuration following cleanRL practices"""
    print("=" * 60)
    print("Dr GRPO Training Configuration")
    print("=" * 60)
    print(f"Experiment: {config.exp_name}")
    print(f"Seed: {config.seed}")
    print(f"Device: {'cuda' if config.cuda else 'cpu'}")
    print(f"Deterministic: {config.torch_deterministic}")
    print(f"\nModel: {config.model_name}")
    print(f"Dataset size: {dataset_size}")
    print(f"\nTraining hyperparameters:")
    print(f"  - Epochs: {config.total_epochs}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Batch size: {config.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"\nGRPO parameters:")
    print(f"  - Generations per prompt: {config.num_generations}")
    print(f"  - Temperature: {config.temperature}")
    print(f"  - Beta (KL coefficient): {config.beta} {'(Dr GRPO - no KL penalty)' if config.beta == 0 else ''}")
    print(f"\nReward configuration:")
    print(f"  - Correctness weight: {config.correctness_weight}")
    print(f"  - Length penalty weight: {config.length_penalty_weight}")
    print("=" * 60)


def validate_environment():
    """Validate environment setup following stable-baselines3 practices"""
    print("\nEnvironment validation:")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available - will use CPU")
    
    # Check if we can use deterministic algorithms
    if hasattr(torch, 'use_deterministic_algorithms'):
        print("Deterministic algorithms: supported")
    else:
        print("Deterministic algorithms: not supported (older PyTorch version)")
    
    print()


def main(config: Optional[ExperimentConfig] = None):
    """
    Main training function with proper initialization and error handling
    
    Args:
        config: Experiment configuration (creates default if None)
    """
    # Initialize configuration
    if config is None:
        config = ExperimentConfig()
    
    # Set random seeds for reproducibility
    set_random_seeds(config.seed, config.torch_deterministic)
    
    # Initialize WandB if tracking is enabled
    if config.track:
        import wandb
        
        # Set environment variables for wandb
        os.environ["WANDB_PROJECT"] = config.wandb_project_name
        if config.wandb_entity:
            os.environ["WANDB_ENTITY"] = config.wandb_entity
        
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            name=f"{config.exp_name}_seed{config.seed}",
            config={
                "model": config.model_name,
                "dataset_size": 12,
                "epochs": config.total_epochs,
                "batch_size": config.per_device_train_batch_size,
                "gradient_accumulation": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "num_generations": config.num_generations,
                "temperature": config.temperature,
                "beta": config.beta,
                "correctness_weight": config.correctness_weight,
                "length_penalty_weight": config.length_penalty_weight,
                "seed": config.seed,
            },
            reinit=True,
        )
    
    # Validate environment
    validate_environment()
    
    # Create dataset
    dataset = create_minimal_dataset()
    
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
            beta=config.beta,  # 0.0 for Dr GRPO (no KL penalty)
            remove_unused_columns=False,  # Keep solution column
            
            # Optimization settings
            bf16=config.cuda and torch.cuda.is_bf16_supported(),
            fp16=False,  # Prefer bf16 over fp16
            dataloader_num_workers=config.num_workers,
            
            # Logging and saving
            logging_steps=config.logging_steps,
            save_strategy=config.save_strategy,
            save_total_limit=config.save_total_limit,
            report_to=["wandb"] if config.track else [],
            
            # Experiment tracking
            run_name=f"{config.exp_name}_seed{config.seed}",
            
            # Reward weights for multiple reward functions
            reward_weights=[config.correctness_weight, config.length_penalty_weight],
        )
        
        # Create reward functions with config
        def correctness_reward_wrapped(completions, prompts=None, **kwargs):
            # Extract solutions - it's passed directly in kwargs
            solutions = kwargs.get('solution', None)
            return correctness_reward(completions, prompts=prompts, solutions=solutions)
        
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
        print("\nStarting GRPO training...")
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
        # Cleanup
        print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")
        
        # Finish WandB run if tracking
        if config.track:
            import wandb
            

    # Evaluate on standardized dataset
    print("\nEvaluating on standardized arithmetic dataset...")
    standard_eval_metrics = evaluate_on_standard_dataset(trainer.model, trainer.tokenizer, trainer.model.device)
    
    # Log standardized metrics
    if 'wandb' in globals() and wandb.run is not None:
        wandb.log(standard_eval_metrics)
        wandb.log({
            "arithmetic_eval": standard_eval_metrics['arithmetic_eval']  # Primary metric
        })
    
    print(f"\n🎯 Standardized evaluation accuracy: {standard_eval_metrics['arithmetic_eval']:.2%}")

wandb.finish()


if __name__ == "__main__":
    # Run with default configuration
    main()
    
    # Example: Run with custom configuration
    # custom_config = ExperimentConfig(
    #     exp_name="grpo_dr_math_custom",
    #     seed=123,
    #     num_generations=8,
    #     total_epochs=5,
    # )
    # main(custom_config)