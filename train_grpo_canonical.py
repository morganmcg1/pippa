#!/usr/bin/env python3
"""
Canonical GRPO Training Script for Arithmetic Tasks
This is the reference implementation incorporating all learnings from our experiments.

Key features:
- Proper batch_indices handling for reward function
- Support for large batch sizes with proper num_generations
- Fixed completion logging
- Comprehensive configuration options
- Built-in dataset creation functions
"""

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
from dotenv import load_dotenv
import os
import wandb
import re
from typing import Dict, Any, List, Optional, Callable
import argparse
from datetime import datetime

# Load environment variables
load_dotenv()

# Custom GRPOTrainer to fix log_completions issue
class GRPOTrainerFixed(GRPOTrainer):
    """GRPOTrainer with fixed completion logging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_completions = []
    
    def _generate_completions(self, prompts, **generation_kwargs):
        completions = super()._generate_completions(prompts, **generation_kwargs)
        self._last_completions = completions
        return completions
    
    def _print_completions_simple(self):
        if hasattr(self, '_last_completions') and self._last_completions:
            print("\n=== Sample Completions ===")
            for i, (prompt, completion) in enumerate(zip(self.state.current_batch['prompt'][:3], 
                                                         self._last_completions[:3])):
                print(f"\nPrompt {i+1}: {prompt}")
                print(f"Completion {i+1}: {completion}")
            print("=" * 50 + "\n")
    
    def log(self, logs: Dict[str, Any], start_time: float = None) -> None:
        try:
            super().log(logs, start_time)
        except AttributeError as e:
            if "add_section" in str(e):
                self._print_completions_simple()
                if hasattr(self, '_wandb'):
                    self._wandb.log(logs, step=self.state.global_step)
            else:
                raise


# Dataset creation functions
def create_arithmetic_dataset(n_samples: int = 100, min_num: int = 0, max_num: int = 20) -> Dataset:
    """Create a dataset of arithmetic problems."""
    prompts = []
    answers = []
    
    operations = ['+', '-', '*']
    
    # Generate unique problems to maximize diversity
    seen_problems = set()
    attempts = 0
    max_attempts = n_samples * 10
    
    while len(prompts) < n_samples and attempts < max_attempts:
        attempts += 1
        a = random.randint(min_num, max_num)
        b = random.randint(min_num, max_num)
        op = random.choice(operations)
        
        # Skip if we've seen this problem
        problem_key = (a, op, b)
        if problem_key in seen_problems:
            continue
        seen_problems.add(problem_key)
        
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:  # *
            result = a * b
        
        prompt = f"Calculate: {a} {op} {b} = "
        prompts.append(prompt)
        answers.append(str(result))
    
    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers
    })


def create_mixed_dataset(n_samples: int = 150) -> Dataset:
    """Create a mixed dataset with arithmetic, counting, and comparison tasks."""
    prompts = []
    answers = []
    task_types = []
    
    # Task distribution: 50% arithmetic, 25% counting, 25% comparison
    n_arithmetic = n_samples // 2
    n_counting = n_samples // 4
    n_comparison = n_samples - n_arithmetic - n_counting
    
    # Arithmetic tasks
    operations = ['+', '-', '*']
    for _ in range(n_arithmetic):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        op = random.choice(operations)
        
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:  # *
            result = a * b
        
        prompt = f"Calculate: {a} {op} {b} = "
        prompts.append(prompt)
        answers.append(str(result))
        task_types.append("arithmetic")
    
    # Counting tasks (easier)
    for _ in range(n_counting):
        count = random.randint(1, 10)
        word = random.choice(["apple", "cat", "dog", "star", "book"])
        words = " ".join([word] * count)
        
        prompt = f"Count the words: {words}. Answer: "
        prompts.append(prompt)
        answers.append(str(count))
        task_types.append("counting")
    
    # Comparison tasks (easier)
    for _ in range(n_comparison):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        
        if a > b:
            answer = "yes"
        else:
            answer = "no"
        
        prompt = f"Is {a} greater than {b}? Answer yes or no: "
        prompts.append(prompt)
        answers.append(answer)
        task_types.append("comparison")
    
    # Shuffle to mix task types
    combined = list(zip(prompts, answers, task_types))
    random.shuffle(combined)
    prompts, answers, task_types = zip(*combined)
    
    return Dataset.from_dict({
        "prompt": list(prompts),
        "answer": list(answers),
        "task_type": list(task_types)
    })


# Answer extraction functions
def extract_arithmetic_answer(text: str, prompt: str) -> str:
    """Extract numerical answer from model output."""
    completion = text[len(prompt):].strip()
    
    # Try to extract first number (including negative)
    match = re.match(r'^-?\d+', completion)
    if match:
        return match.group(0)
    
    # Fallback: take first token
    tokens = completion.split()
    if tokens:
        return tokens[0]
    
    return completion


def extract_mixed_answer(text: str, prompt: str, task_type: str) -> str:
    """Extract answer based on task type."""
    completion = text[len(prompt):].strip()
    
    if task_type == "comparison":
        # Look for yes/no
        completion_lower = completion.lower()
        if "yes" in completion_lower:
            return "yes"
        elif "no" in completion_lower:
            return "no"
    
    # For arithmetic and counting, extract number
    return extract_arithmetic_answer(text, prompt)


# Reward functions
def arithmetic_reward_function(samples: List[str], prompts: List[str], answers: List[str], **kwargs) -> List[float]:
    """Simple binary reward: +1 if correct, -1 if incorrect."""
    rewards = []
    
    for sample, prompt, expected in zip(samples, prompts, answers):
        extracted = extract_arithmetic_answer(sample, prompt)
        
        if extracted == expected:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    
    return rewards


def mixed_reward_function(samples: List[str], prompts: List[str], answers: List[str], 
                         task_types: List[str], **kwargs) -> List[float]:
    """Binary reward with task-specific extraction."""
    rewards = []
    
    for sample, prompt, expected, task_type in zip(samples, prompts, answers, task_types):
        extracted = extract_mixed_answer(sample, prompt, task_type)
        
        if extracted == expected:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    
    return rewards


def create_reward_wrapper(dataset: Dataset, reward_fn: Callable, use_mixed: bool = False) -> Callable:
    """Create a reward wrapper that properly handles batch_indices from GRPO."""
    
    def reward_wrapper(completions, prompts=None, **kwargs):
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        # Get indices from kwargs - this is how GRPO passes the batch indices
        batch_indices = kwargs.get('batch_indices', None)
        
        if batch_indices is not None:
            # Use the provided indices
            answers = [dataset[idx]["answer"] for idx in batch_indices]
            if use_mixed:
                task_types = [dataset[idx]["task_type"] for idx in batch_indices]
                return reward_fn(completions, prompts, answers, task_types, **kwargs)
            else:
                return reward_fn(completions, prompts, answers, **kwargs)
        else:
            # Fallback: try to match prompts to dataset
            if use_mixed:
                prompt_to_data = {d["prompt"]: (d["answer"], d["task_type"]) for d in dataset}
                answers = []
                task_types = []
                for p in prompts:
                    if p in prompt_to_data:
                        ans, tt = prompt_to_data[p]
                        answers.append(ans)
                        task_types.append(tt)
                    else:
                        answers.append("")
                        task_types.append("arithmetic")
                return reward_fn(completions, prompts, answers, task_types, **kwargs)
            else:
                prompt_to_answer = {d["prompt"]: d["answer"] for d in dataset}
                answers = [prompt_to_answer.get(p, "") for p in prompts]
                return reward_fn(completions, prompts, answers, **kwargs)
    
    return reward_wrapper


def main():
    parser = argparse.ArgumentParser(description="Canonical GRPO Training Script")
    
    # Dataset arguments
    parser.add_argument("--dataset_type", type=str, default="arithmetic", 
                       choices=["arithmetic", "mixed", "small_numbers"],
                       help="Type of dataset to use")
    parser.add_argument("--n_samples", type=int, default=100,
                       help="Number of samples in dataset")
    parser.add_argument("--min_num", type=int, default=0,
                       help="Minimum number for arithmetic problems")
    parser.add_argument("--max_num", type=int, default=20,
                       help="Maximum number for arithmetic problems")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size (should be divisible by num_generations)")
    parser.add_argument("--num_generations", type=int, default=16,
                       help="Number of generations per prompt")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                       help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="KL penalty coefficient (critical for arithmetic)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct",
                       help="Model to train")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--exp_name", type=str, default=None,
                       help="Experiment name for WandB")
    parser.add_argument("--tags", type=str, nargs="+", default=["grpo", "canonical"],
                       help="Tags for WandB run")
    
    args = parser.parse_args()
    
    # Validate batch size
    if args.batch_size % args.num_generations != 0:
        raise ValueError(f"Batch size ({args.batch_size}) must be divisible by num_generations ({args.num_generations})")
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Create dataset
    print("Creating dataset...")
    if args.dataset_type == "arithmetic":
        dataset = create_arithmetic_dataset(args.n_samples, args.min_num, args.max_num)
        use_mixed = False
        reward_fn = arithmetic_reward_function
    elif args.dataset_type == "small_numbers":
        dataset = create_arithmetic_dataset(args.n_samples, 0, 10)
        use_mixed = False
        reward_fn = arithmetic_reward_function
    else:  # mixed
        dataset = create_mixed_dataset(args.n_samples)
        use_mixed = True
        reward_fn = mixed_reward_function
    
    # Print dataset info
    print(f"\nDataset type: {args.dataset_type}")
    print(f"Dataset size: {len(dataset)} samples")
    print("\nSample problems:")
    for i in range(min(10, len(dataset))):
        if use_mixed:
            print(f"[{i}] [{dataset['task_type'][i]}] {dataset['prompt'][i]} → {dataset['answer'][i]}")
        else:
            print(f"[{i}] {dataset['prompt'][i]} → {dataset['answer'][i]}")
    
    # Prepare dataset
    def prepare_dataset(sample):
        result = {
            "prompt": sample["prompt"],
            "answer": sample["answer"]
        }
        if "task_type" in sample:
            result["task_type"] = sample["task_type"]
        return result
    
    dataset = dataset.map(prepare_dataset)
    
    # Create reward wrapper
    reward_wrapper = create_reward_wrapper(dataset, reward_fn, use_mixed)
    
    # Initialize WandB
    if args.exp_name is None:
        args.exp_name = f"grpo_{args.dataset_type}_b{args.batch_size}_g{args.num_generations}"
    
    wandb_config = {
        "model": args.model_name,
        "dataset_type": args.dataset_type,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "epochs": args.epochs,
        "seed": args.seed,
        "beta": args.beta,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H:%M"),
    }
    
    if args.dataset_type == "arithmetic" or args.dataset_type == "small_numbers":
        wandb_config["min_num"] = args.min_num
        wandb_config["max_num"] = args.max_num
    
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=args.exp_name,
        tags=args.tags,
        config=wandb_config
    )
    
    # GRPO configuration
    print("\nConfiguring GRPO trainer...")
    config = GRPOConfig(
        output_dir=f"./grpo_output/{args.exp_name}",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="no",
        report_to="wandb",
        remove_unused_columns=False,
        log_completions=True,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=16,
        max_prompt_length=128,
        seed=args.seed,
        # GRPO specific
        beta=args.beta,
        loss_type="grpo",
        # Additional optimizations
        dataloader_num_workers=0,
        wandb_log_unique_prompts=True,
        bf16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",
        scale_rewards=True,
    )
    
    print(f"\nGRPO Configuration:")
    print(f"Batch size: {args.batch_size}")
    print(f"Num generations: {args.num_generations}")
    print(f"Effective batch size per update: {args.batch_size}")
    print(f"Beta (KL penalty): {args.beta}")
    print(f"Temperature: {args.temperature}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    
    # Initialize trainer
    trainer = GRPOTrainerFixed(
        model=args.model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper],
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate final performance
    print("\nEvaluating final model...")
    correct = 0
    total = len(dataset)
    
    # Get model and tokenizer from trainer
    model = trainer.model
    tokenizer = trainer.tokenizer
    
    model.eval()
    
    # Track performance by category
    if use_mixed:
        results_by_type = {}
        task_types = set(dataset["task_type"])
    else:
        errors_by_op = {'+': 0, '-': 0, '*': 0}
        total_by_op = {'+': 0, '-': 0, '*': 0}
    
    with torch.no_grad():
        for i in range(total):
            prompt = dataset[i]["prompt"]
            expected = dataset[i]["answer"]
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if use_mixed:
                task_type = dataset[i]["task_type"]
                extracted = extract_mixed_answer(response, prompt, task_type)
                
                if task_type not in results_by_type:
                    results_by_type[task_type] = {"correct": 0, "total": 0}
                
                results_by_type[task_type]["total"] += 1
                if extracted == expected:
                    correct += 1
                    results_by_type[task_type]["correct"] += 1
            else:
                extracted = extract_arithmetic_answer(response, prompt)
                
                # Determine operation for arithmetic
                for op in ['+', '-', '*']:
                    if f" {op} " in prompt:
                        total_by_op[op] += 1
                        break
                
                if extracted == expected:
                    correct += 1
                else:
                    # Track which operation failed
                    for op in ['+', '-', '*']:
                        if f" {op} " in prompt:
                            errors_by_op[op] += 1
                            break
    
    accuracy = correct / total
    print(f"\nFinal accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Detailed breakdown
    if use_mixed:
        print("\nAccuracy by task type:")
        for task_type, results in results_by_type.items():
            task_acc = results["correct"] / results["total"] if results["total"] > 0 else 0
            print(f"  {task_type}: {task_acc:.2%} ({results['correct']}/{results['total']})")
            wandb.log({f"final_accuracy_{task_type}": task_acc})
    else:
        print("\nAccuracy by operation:")
        for op in ['+', '-', '*']:
            if total_by_op[op] > 0:
                op_accuracy = (total_by_op[op] - errors_by_op[op]) / total_by_op[op]
                print(f"  {op}: {op_accuracy:.2%} ({total_by_op[op] - errors_by_op[op]}/{total_by_op[op]})")
                wandb.log({f"final_accuracy_{op}": op_accuracy})
    
    # Log final metrics
    wandb.log({
        "final_accuracy": accuracy,
        "final_correct": correct,
        "final_total": total
    })
    
    wandb.finish()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()