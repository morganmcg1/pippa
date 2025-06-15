#!/usr/bin/env python3
"""
GRPO training with rich rewards + mixed tasks (arithmetic, counting, comparison).
Combining the two most successful approaches.
"""

import torch
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
from dotenv import load_dotenv
import os
import wandb
import re
from typing import Dict, Any, List

# Load environment variables
load_dotenv()

# Enable WandB model checkpointing
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def evaluate_on_standard_dataset(model, tokenizer, device) -> Dict[str, float]:
    """Evaluate model on the standardized arithmetic evaluation dataset."""
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    
    results_by_difficulty = {}
    results_by_operation = {}
    correct_total = 0
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(eval_dataset):
            prompt = sample['prompt']
            expected = sample['answer']
            difficulty = sample['difficulty']
            operation = sample['metadata']['operation'] if sample['metadata'] else None
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            completion = response[len(prompt):].strip()
            match = re.match(r'^-?\d+', completion)
            predicted = match.group(0) if match else completion.split()[0] if completion else ""
            
            is_correct = predicted == expected
            if is_correct:
                correct_total += 1
            
            if difficulty not in results_by_difficulty:
                results_by_difficulty[difficulty] = {'correct': 0, 'total': 0}
            results_by_difficulty[difficulty]['total'] += 1
            if is_correct:
                results_by_difficulty[difficulty]['correct'] += 1
            
            if operation:
                if operation not in results_by_operation:
                    results_by_operation[operation] = {'correct': 0, 'total': 0}
                results_by_operation[operation]['total'] += 1
                if is_correct:
                    results_by_operation[operation]['correct'] += 1
    
    overall_accuracy = correct_total / len(eval_dataset)
    
    difficulty_accs = {}
    for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        if diff in results_by_difficulty:
            stats = results_by_difficulty[diff]
            acc = stats['correct'] / stats['total']
            difficulty_accs[f"eval_{diff}_accuracy"] = acc
    
    operation_accs = {}
    for op in ['+', '-', '*', '/']:
        if op in results_by_operation:
            stats = results_by_operation[op]
            acc = stats['correct'] / stats['total']
            op_name = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op]
            operation_accs[f"eval_{op_name}_accuracy"] = acc
    
    return {
        'eval_accuracy': overall_accuracy,
        'eval_correct': correct_total,
        'eval_total': len(eval_dataset),
        **difficulty_accs,
        **operation_accs
    }

# Custom GRPOTrainer to fix log_completions issue
class GRPOTrainerFixed(GRPOTrainer):
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
                if hasattr(self, '_wandb') and self._wandb:
                    self._wandb.log(logs, step=self.state.global_step)
            else:
                raise

def create_mixed_dataset_with_rich_rewards(n_samples=240):
    """Create mixed dataset: 50% arithmetic, 25% counting, 25% comparison (0-10 range)."""
    prompts = []
    answers = []
    task_types = []
    
    n_arithmetic = n_samples // 2
    n_counting = n_samples // 4
    n_comparison = n_samples // 4
    
    # Arithmetic tasks (50%)
    operations = ['+', '-', '*']
    for _ in range(n_arithmetic):
        a = random.randint(0, 10)
        b = random.randint(0, 10)
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
    
    # Counting tasks (25%)
    for _ in range(n_counting):
        count = random.randint(1, 10)
        item = random.choice(["apple", "star", "dot", "number"])
        items_str = " ".join([item] * count)
        prompt = f"Count the {item}s: {items_str}. Answer: "
        prompts.append(prompt)
        answers.append(str(count))
        task_types.append("counting")
    
    # Comparison tasks (25%)
    for _ in range(n_comparison):
        a = random.randint(0, 10)
        b = random.randint(0, 10)
        if a > b:
            answer = "yes"
        else:
            answer = "no"
        prompt = f"Is {a} greater than {b}? Answer yes or no: "
        prompts.append(prompt)
        answers.append(answer)
        task_types.append("comparison")
    
    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "task_type": task_types
    })

def extract_answer(text: str, prompt: str) -> str:
    """Extract the answer from model output"""
    completion = text[len(prompt):].strip()
    
    # For yes/no questions
    if "yes or no" in prompt.lower():
        completion_lower = completion.lower()
        if completion_lower.startswith("yes"):
            return "yes"
        elif completion_lower.startswith("no"):
            return "no"
    
    # For numeric answers
    match = re.match(r'^-?\d+', completion)
    if match:
        return match.group(0)
    
    tokens = completion.split()
    if tokens:
        return tokens[0]
    return completion

def rich_reward_function(samples: List[str], prompts: List[str], answers: List[str], **kwargs) -> List[float]:
    """
    Rich reward function with task-specific partial credit.
    """
    rewards = []
    
    for sample, prompt, expected in zip(samples, prompts, answers):
        extracted = extract_answer(sample, prompt)
        
        # Comparison tasks (yes/no)
        if "yes or no" in prompt.lower():
            if extracted.lower() == expected.lower():
                reward = 1.0
            else:
                reward = -1.0  # Binary for yes/no
        
        # Numeric tasks (arithmetic and counting)
        else:
            try:
                predicted_num = int(extracted)
                expected_num = int(expected)
                
                # Calculate distance
                distance = abs(predicted_num - expected_num)
                
                # Assign reward based on distance
                if distance == 0:
                    reward = 1.0  # Correct
                elif distance == 1:
                    reward = 0.7  # Very close
                elif distance == 2:
                    reward = 0.4  # Close
                elif distance <= 5:
                    reward = 0.1  # Somewhat close
                elif distance <= 10:
                    reward = -0.2  # Far
                else:
                    reward = -0.5  # Very far
                    
            except (ValueError, AttributeError):
                # Non-numeric output for numeric task
                reward = -1.0
        
        rewards.append(reward)
    
    return rewards

def main():
    # Model configuration
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # Training configuration
    n_samples = 240  # Divisible by 16 and supports mixed ratios
    batch_size = 240  # Match dataset size
    num_generations = 16
    learning_rate = 5e-6
    temperature = 0.7
    epochs = 75
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="grpo_rich_mixed_tasks",
        tags=["grpo-setup", "standardized-eval", "rich-rewards", "mixed-tasks", "best-combination"],
        config={
            "model": model_name,
            "n_samples": n_samples,
            "batch_size": batch_size,
            "num_generations": num_generations,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "epochs": epochs,
            "seed": SEED,
            "beta": 0.1,
            "eval_dataset": "morgan/arithmetic_eval",
            "number_range": "0-10",
            "task_distribution": "50% arithmetic, 25% counting, 25% comparison",
            "reward_type": "rich_task_specific"
        }
    )
    
    print("Creating mixed dataset with rich rewards (0-10)...")
    dataset = create_mixed_dataset_with_rich_rewards(n_samples)
    
    # Print dataset statistics
    task_counts = {}
    for task in dataset["task_type"]:
        task_counts[task] = task_counts.get(task, 0) + 1
    print(f"Task distribution: {task_counts}")
    
    # Prepare dataset
    def prepare_dataset(sample):
        return {
            "prompt": sample["prompt"],
            "answer": sample["answer"],
            "task_type": sample["task_type"]
        }
    
    dataset = dataset.map(prepare_dataset)
    
    # Create reward function wrapper
    def reward_wrapper(completions, prompts=None, **kwargs):
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        batch_indices = kwargs.get('batch_indices', None)
        
        if batch_indices is not None:
            answers = [dataset[idx]["answer"] for idx in batch_indices]
        else:
            prompt_to_answer = {d["prompt"]: d["answer"] for d in dataset}
            answers = [prompt_to_answer.get(p, "") for p in prompts]
        
        return rich_reward_function(completions, prompts, answers, **kwargs)
    
    # GRPO configuration
    config = GRPOConfig(
        output_dir="./grpo-rich-mixed",
        run_name="grpo_rich_mixed_tasks",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=3,
        report_to="wandb",
        remove_unused_columns=False,
        log_completions=True,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=16,
        max_prompt_length=128,
        seed=SEED,
        # GRPO specific
        beta=0.1,
        loss_type="grpo",
        # Additional
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
    
    print("\n" + "="*60)
    print("RICH REWARDS + MIXED TASKS EXPERIMENT")
    print("="*60)
    print("Combining the two most successful approaches:")
    print("1. Rich rewards (achieved 75% accuracy)")
    print("2. Mixed tasks (achieved 54.7% before standardized eval)")
    print("\nExpected: 80%+ accuracy by combining strengths")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = GRPOTrainerFixed(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper]
    )
    
    print("Starting training with rich rewards + mixed tasks...")
    trainer.train()
    
    print("\nTraining completed!")
    
    # Final evaluation
    model = trainer.model
    tokenizer = trainer.tokenizer
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    
    print("\nRunning final standardized evaluation...")
    final_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
    
    wandb.log({
        "final/eval_accuracy": final_eval_metrics['eval_accuracy'],
        **{f"final/{k}": v for k, v in final_eval_metrics.items()}
    })
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS - Rich Rewards + Mixed Tasks")
    print("="*60)
    print(f"Final standardized eval accuracy: {final_eval_metrics['eval_accuracy']:.2%}")
    print(f"Rich rewards only: 75.0%")
    print(f"Mixed tasks only: 54.7% (pre-standardized)")
    print(f"Binary rewards best: 45.5%")
    print(f"Expected with combination: 80%+")
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    main()