#!/usr/bin/env python3
"""
GRPO Arithmetic Training - Smaller Numbers Experiment
Uses numbers 0-10 instead of 0-20 to make the task easier
Hypothesis: Smaller number range will be easier to learn and achieve higher accuracy
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
from typing import Dict, Any, List, Optional
import logging

# Load environment variables
load_dotenv()

# Force seed for reproducibility
SEED = 4567
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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
                if hasattr(self, '_wandb'):
                    self._wandb.log(logs, step=self.state.global_step)
            else:
                raise

def create_small_arithmetic_dataset(n_samples=200, max_num=10):
    """Create arithmetic problems with smaller numbers (0-10)"""
    prompts = []
    answers = []
    
    operations = ['+', '-', '*']
    
    # Generate unique problems to maximize diversity
    seen_problems = set()
    
    while len(prompts) < n_samples:
        a = random.randint(0, max_num)
        b = random.randint(0, max_num)
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

def extract_answer(text: str, prompt: str) -> str:
    """Extract the answer from model output"""
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

def reward_function(samples: List[str], prompts: List[str], answers: List[str], **kwargs) -> List[float]:
    """Simple binary reward: +1 if correct, -1 if incorrect"""
    rewards = []
    
    for sample, prompt, expected in zip(samples, prompts, answers):
        extracted = extract_answer(sample, prompt)
        
        if extracted == expected:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    
    return rewards

def main():
    # Model configuration
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # Training configuration - SMALLER NUMBERS
    n_samples = 200  # More samples since range is smaller
    max_num = 10  # Smaller than 20
    batch_size = 64
    num_generations = 4  # Must divide evenly into batch_size
    learning_rate = 5e-6
    temperature = 0.7
    epochs = 50
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="grpo_arithmetic_smaller_numbers",
        tags=["grpo-setup", "arithmetic", "overfit", "break-barrier", "small-numbers"],
        config={
            "model": model_name,
            "n_samples": n_samples,
            "max_number": max_num,
            "batch_size": batch_size,
            "num_generations": num_generations,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "epochs": epochs,
            "seed": SEED,
            "beta": 0.1,
            "experiment": "smaller_numbers"
        }
    )
    
    print("Preparing training configuration...")
    
    print(f"Creating dataset with numbers 0-{max_num}...")
    dataset = create_small_arithmetic_dataset(n_samples, max_num)
    
    # Analyze answer distribution
    answer_range = {}
    for answer in dataset['answer']:
        val = int(answer)
        if val < -10:
            key = "< -10"
        elif val < 0:
            key = "-10 to -1"
        elif val <= 10:
            key = "0 to 10"
        elif val <= 50:
            key = "11 to 50"
        else:
            key = "> 50"
        answer_range[key] = answer_range.get(key, 0) + 1
    
    print(f"\nAnswer distribution:")
    for key, count in sorted(answer_range.items()):
        print(f"  {key}: {count} samples")
    
    print("\nSample problems:")
    for i in range(10):
        print(f"[{i}] {dataset['prompt'][i]} → {dataset['answer'][i]}")
    
    # Prepare dataset
    def prepare_dataset(sample):
        return {
            "prompt": sample["prompt"],
            "answer": sample["answer"]
        }
    
    dataset = dataset.map(prepare_dataset)
    
    # Create reward function wrapper
    def reward_wrapper(samples, prompts, **kwargs):
        answers = [dataset[i]["answer"] for i in range(len(prompts))]
        return reward_function(samples, prompts, answers, **kwargs)
    
    # GRPO configuration
    config = GRPOConfig(
        output_dir="./grpo-arithmetic-small-numbers",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="no",
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
    
    print("\nGRPO Configuration:")
    print(f"Smaller number range (0-{max_num}) for easier learning")
    print(f"Beta (KL penalty): {config.beta}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    
    # Initialize trainer
    trainer = GRPOTrainerFixed(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper],
    )
    
    print("\nStarting training with smaller numbers...")
    print(f"Hypothesis: Numbers 0-{max_num} will be easier than 0-20")
    print("Expected: Higher accuracy, potentially break 87.5% barrier")
    trainer.train()
    
    # Evaluate final performance
    print("\nEvaluating final model...")
    correct = 0
    total = len(dataset)
    
    # Track errors by operation type
    errors_by_op = {'+': 0, '-': 0, '*': 0}
    total_by_op = {'+': 0, '-': 0, '*': 0}
    
    # Get model and tokenizer from trainer
    model = trainer.model
    tokenizer = trainer.tokenizer
    
    model.eval()
    with torch.no_grad():
        for i in range(total):
            prompt = dataset[i]["prompt"]
            expected = dataset[i]["answer"]
            
            # Determine operation
            for op in ['+', '-', '*']:
                if f" {op} " in prompt:
                    total_by_op[op] += 1
                    break
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            extracted = extract_answer(response, prompt)
            
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
    
    # Accuracy by operation
    print("\nAccuracy by operation:")
    for op in ['+', '-', '*']:
        if total_by_op[op] > 0:
            op_accuracy = (total_by_op[op] - errors_by_op[op]) / total_by_op[op]
            print(f"  {op}: {op_accuracy:.2%} ({total_by_op[op] - errors_by_op[op]}/{total_by_op[op]})")
    
    # Log final metrics
    wandb.log({
        "final_accuracy": accuracy,
        "final_correct": correct,
        "final_total": total,
        "accuracy_addition": (total_by_op['+'] - errors_by_op['+']) / total_by_op['+'] if total_by_op['+'] > 0 else 0,
        "accuracy_subtraction": (total_by_op['-'] - errors_by_op['-']) / total_by_op['-'] if total_by_op['-'] > 0 else 0,
        "accuracy_multiplication": (total_by_op['*'] - errors_by_op['*']) / total_by_op['*'] if total_by_op['*'] > 0 else 0,
    })
    
    wandb.finish()
    print("\nTraining completed!")

if __name__ == "__main__":
    main()