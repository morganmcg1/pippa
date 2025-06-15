#!/usr/bin/env python3
"""
GRPO Ultra-High Diversity Experiment - 64 Generations
Implements mixed dataset (arithmetic + counting + comparison) with maximum generation diversity
Target: Break 60% final accuracy barrier
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import torch
from datasets import Dataset
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer
import wandb
from datetime import datetime
import numpy as np
import random
import re

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "pippa"
os.environ["WANDB_ENTITY"] = "wild-ai"

@dataclass
class ScriptArguments:
    """Script arguments for ultra-high diversity GRPO training"""
    model_name: str = field(
        default="Qwen/Qwen2-0.5B-Instruct",
        metadata={"help": "Model to train"}
    )
    dataset_size: int = field(
        default=150,
        metadata={"help": "Total number of samples in mixed dataset"}
    )
    batch_size: int = field(
        default=960,  # 64 generations × 15 prompts (fallback from 1024 if OOM)
        metadata={"help": "Batch size for training"}
    )
    num_generations: int = field(
        default=64,  # Ultra-high diversity!
        metadata={"help": "Number of generations per prompt"}
    )
    learning_rate: float = field(
        default=5e-6,
        metadata={"help": "Learning rate"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Temperature for generation"}
    )
    epochs: int = field(
        default=50,
        metadata={"help": "Number of training epochs"}
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "KL penalty coefficient"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )

def create_mixed_dataset(n_samples: int = 150) -> Dataset:
    """Create a mixed dataset with arithmetic, counting, and comparison tasks"""
    
    # Calculate samples per task type
    n_arithmetic = n_samples // 2  # 50%
    n_counting = n_samples // 4    # 25%
    n_comparison = n_samples // 4  # 25%
    
    samples = []
    
    # 1. Arithmetic tasks (50%)
    for _ in range(n_arithmetic):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        op = random.choice(['+', '-', '*'])
        
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else:  # *
            answer = a * b
            
        prompt = f"Calculate: {a} {op} {b} = "
        samples.append({
            "prompt": prompt,
            "answer": str(answer),
            "task_type": "arithmetic"
        })
    
    # 2. Counting tasks (25%)
    for _ in range(n_counting):
        # Count items in a list
        num_items = random.randint(1, 10)
        items = random.choice(['apple', 'star', 'ball', 'cat', 'dog'])
        item_list = ' '.join([items] * num_items)
        
        prompt = f"Count the {items}s: {item_list}. Answer: "
        samples.append({
            "prompt": prompt,
            "answer": str(num_items),
            "task_type": "counting"
        })
    
    # 3. Comparison tasks (25%)
    for _ in range(n_comparison):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        
        prompt = f"Is {a} greater than {b}? Answer yes or no: "
        answer = "yes" if a > b else "no"
        
        samples.append({
            "prompt": prompt,
            "answer": answer,
            "task_type": "comparison"
        })
    
    # Shuffle to mix task types
    random.shuffle(samples)
    
    return Dataset.from_list(samples)

def extract_answer(completion: str, prompt: str, task_type: str) -> str:
    """Extract answer from model completion based on task type"""
    # Remove the prompt from completion
    answer_part = completion[len(prompt):].strip()
    
    if task_type == "comparison":
        # Look for yes/no
        answer_lower = answer_part.lower()
        if "yes" in answer_lower:
            return "yes"
        elif "no" in answer_lower:
            return "no"
        else:
            # Default to first word
            return answer_part.split()[0] if answer_part else "unknown"
    else:
        # For arithmetic and counting, extract number
        match = re.match(r'^-?\d+', answer_part)
        if match:
            return match.group()
        
        # Fallback: first token
        tokens = answer_part.split()
        return tokens[0] if tokens else "0"

class MixedTaskRewardWrapper:
    """Reward wrapper for mixed task dataset"""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        
    def __call__(self, completions: List[str], prompts: List[str], 
                 outputs: List[str], **kwargs) -> List[float]:
        rewards = []
        
        # Get batch indices to map back to dataset
        batch_indices = kwargs.get('batch_indices', list(range(len(prompts))))
        
        for i, (completion, prompt) in enumerate(zip(completions, prompts)):
            # Get the dataset index
            dataset_idx = batch_indices[i] % len(self.dataset)
            expected = self.dataset[dataset_idx]['answer']
            task_type = self.dataset[dataset_idx]['task_type']
            
            # Extract answer based on task type
            extracted = extract_answer(completion, prompt, task_type)
            
            # Calculate reward
            if extracted == expected:
                reward = 1.0
            else:
                reward = -1.0
                
            rewards.append(reward)
            
        return rewards

class GRPOTrainerFixed(GRPOTrainer):
    """Fixed GRPO trainer that handles log_completions properly"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_completions = []
        
    def _generate_completions(self, prompts, **kwargs):
        """Override to capture completions"""
        result = super()._generate_completions(prompts, **kwargs)
        # Store for logging
        self.last_completions = result['completions'] if isinstance(result, dict) else []
        return result
        
    def log(self, logs: Dict[str, Any], start_time: float = None) -> None:
        """Override log to handle completion logging"""
        try:
            super().log(logs, start_time)
        except AttributeError as e:
            if "add_section" in str(e):
                # Print a sample of completions
                if self.last_completions and self.args.log_completions:
                    print("\n=== Sample Completions ===")
                    for i, comp in enumerate(self.last_completions[:3]):
                        print(f"[{i}] {comp[:100]}...")
                    print("=" * 50 + "\n")
                
                # Still log metrics to WandB
                if self.accelerator.is_main_process:
                    wandb.log(logs, step=self.state.global_step)
            else:
                raise

def evaluate_model(model, tokenizer, dataset: Dataset, reward_wrapper) -> Dict[str, float]:
    """Evaluate model on mixed dataset"""
    correct_by_type = {"arithmetic": 0, "counting": 0, "comparison": 0}
    total_by_type = {"arithmetic": 0, "counting": 0, "comparison": 0}
    
    model.eval()
    with torch.no_grad():
        for item in dataset:
            prompt = item['prompt']
            expected = item['answer']
            task_type = item['task_type']
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            extracted = extract_answer(completion, prompt, task_type)
            
            total_by_type[task_type] += 1
            if extracted == expected:
                correct_by_type[task_type] += 1
    
    # Calculate accuracies
    results = {
        "final_correct": sum(correct_by_type.values()),
        "final_total": sum(total_by_type.values()),
        "final_accuracy": sum(correct_by_type.values()) / sum(total_by_type.values())
    }
    
    # Add per-task accuracies
    for task_type in ["arithmetic", "counting", "comparison"]:
        if total_by_type[task_type] > 0:
            accuracy = correct_by_type[task_type] / total_by_type[task_type]
            results[f"final_accuracy_{task_type}"] = accuracy
    
    return results

def main():
    args = ScriptArguments()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize wandb
    run_name = f"grpo_ultra_diversity_g{args.num_generations}_b{args.batch_size}"
    wandb.init(
        project="pippa",
        name=run_name,
        tags=["grpo-setup", "arithmetic", "ultra-diversity", "mixed-tasks", "64-generations", "break-60"],
        config=vars(args)
    )
    
    print(f"\n{'='*60}")
    print(f"GRPO Ultra-High Diversity Experiment")
    print(f"{'='*60}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dataset: Mixed (50% arithmetic, 25% counting, 25% comparison)")
    print(f"Target: Break 60% final accuracy")
    print(f"{'='*60}\n")
    
    # Create mixed dataset
    print("Creating mixed dataset...")
    dataset = create_mixed_dataset(args.dataset_size)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample distribution:")
    task_counts = {}
    for item in dataset:
        task_type = item['task_type']
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create reward wrapper
    reward_wrapper = MixedTaskRewardWrapper(dataset)
    
    # Training configuration
    config = GRPOConfig(
        output_dir=f"./grpo_ultra_diversity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        beta=args.beta,
        temperature=args.temperature,
        num_generations=args.num_generations,
        logging_steps=1,
        save_steps=100,
        report_to=["wandb"],
        remove_unused_columns=False,
        num_workers=0,
        seed=args.seed,
        max_completion_length=16,
        loss_type="grpo",
        scale_rewards=True,
        wandb_log_unique_prompts=True,
        log_completions=True,
        mask_truncated_completions=True,
    )
    
    # Initialize trainer
    print(f"\nInitializing GRPO trainer with {args.num_generations} generations...")
    trainer = GRPOTrainerFixed(
        model=args.model_name,
        reward_funcs=[reward_wrapper],
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    print(f"Effective batch size: {config.per_device_train_batch_size}")
    print(f"Prompts per batch: {config.per_device_train_batch_size // args.num_generations}")
    print(f"Total training steps: {len(dataset) * args.epochs // (config.per_device_train_batch_size // args.num_generations)}")
    
    # Train
    print("\nStarting ultra-high diversity training...")
    print("This will be ~4x slower per epoch but should achieve much better accuracy!")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating model...")
    eval_results = evaluate_model(trainer.model, tokenizer, dataset, reward_wrapper)
    
    # Log results
    wandb.log(eval_results)
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {eval_results['final_accuracy']:.1%}")
    for task_type in ["arithmetic", "counting", "comparison"]:
        key = f"final_accuracy_{task_type}"
        if key in eval_results:
            print(f"{task_type.capitalize()} Accuracy: {eval_results[key]:.1%}")
    print(f"{'='*60}\n")
    
    wandb.finish()

if __name__ == "__main__":
    main()