#!/usr/bin/env python3
"""
GRPO training with rich reward function and 500 samples.
This uses the EXACT configuration that achieved 75% accuracy,
but with 500 unique training samples instead of 130.
"""

import torch
from datasets import Dataset, load_dataset
from transformers import TrainerCallback, TrainerState, TrainerControl
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

def create_arithmetic_dataset_500(n_samples=500):
    """Create arithmetic dataset with 500 unique problems, 0-10 numbers only."""
    prompts = []
    answers = []
    
    operations = ['+', '-', '*']
    
    # Generate unique problems
    seen_problems = set()
    attempts = 0
    max_attempts = n_samples * 5
    
    while len(prompts) < n_samples and attempts < max_attempts:
        a = random.randint(0, 10)
        b = random.randint(0, 10)
        op = random.choice(operations)
        
        # Create unique identifier
        problem_id = f"{a}{op}{b}"
        
        # Skip if we've seen this exact problem
        if problem_id in seen_problems:
            attempts += 1
            continue
        
        seen_problems.add(problem_id)
        
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:  # *
            result = a * b
        
        prompt = f"Calculate: {a} {op} {b} = "
        prompts.append(prompt)
        answers.append(str(result))
        attempts += 1
    
    print(f"Generated {len(prompts)} unique arithmetic problems (0-10 range)")
    
    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers
    })

def extract_answer(text: str, prompt: str) -> str:
    """Extract the answer from model output"""
    completion = text[len(prompt):].strip()
    match = re.match(r'^-?\d+', completion)
    if match:
        return match.group(0)
    tokens = completion.split()
    if tokens:
        return tokens[0]
    return completion

def rich_reward_function(samples: List[str], prompts: List[str], answers: List[str], **kwargs) -> List[float]:
    """
    Rich reward function with partial credit - EXACT CONFIGURATION FROM 75% SUCCESS.
    Gives graduated rewards based on how close the answer is to correct.
    """
    rewards = []
    
    for sample, prompt, expected in zip(samples, prompts, answers):
        extracted = extract_answer(sample, prompt)
        
        try:
            predicted_num = int(extracted)
            expected_num = int(expected)
            
            distance = abs(predicted_num - expected_num)
            
            # EXACT reward schedule from 75% success
            if distance == 0:
                reward = 1.0
            elif distance == 1:
                reward = 0.7
            elif distance == 2:
                reward = 0.4
            elif distance <= 5:
                reward = 0.1
            elif distance <= 10:
                reward = -0.2
            else:
                reward = -0.5
                
        except (ValueError, AttributeError):
            # Non-numeric output
            reward = -1.0
        
        rewards.append(reward)
    
    return rewards

def main():
    # Model configuration - EXACT SAME AS 75% SUCCESS
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # Training configuration - EXACT SAME except n_samples
    n_samples = 500  # Only change: 500 instead of 130
    batch_size = 256
    num_generations = 16
    learning_rate = 5e-6
    temperature = 0.7
    epochs = 50
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="grpo_rich_rewards_500_optimal",
        tags=["grpo-setup", "standardized-eval", "rich-rewards", "500-samples"],
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
            "reward_type": "rich_stepped"
        }
    )
    
    print("\n" + "="*60)
    print("RICH REWARDS WITH 500 SAMPLES - OPTIMAL CONFIGURATION")
    print("="*60)
    print("Using EXACT configuration that achieved 75% accuracy")
    print("Only change: 500 unique samples instead of 130")
    print("Goal: Push beyond 75% with more training data")
    print("="*60 + "\n")
    
    print("Creating arithmetic dataset with 500 unique samples (0-10)...")
    dataset = create_arithmetic_dataset_500(n_samples)
    
    # Prepare dataset
    def prepare_dataset(sample):
        return {
            "prompt": sample["prompt"],
            "answer": sample["answer"]
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
    
    # GRPO configuration - EXACT SAME AS 75% SUCCESS
    config = GRPOConfig(
        output_dir="./grpo-rich-500-optimal",
        run_name="grpo_rich_rewards_500_optimal",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="wandb",
        remove_unused_columns=False,
        log_completions=True,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=16,
        max_prompt_length=128,
        seed=SEED,
        # GRPO specific - EXACT SAME
        beta=0.1,
        loss_type="grpo",
        # Additional - EXACT SAME
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
    
    # Initialize trainer
    trainer = GRPOTrainerFixed(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper]
    )
    
    print("Starting training with rich rewards + 500 samples...")
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
    print("FINAL RESULTS - Rich Rewards + 500 Samples")
    print("="*60)
    print(f"Final standardized eval accuracy: {final_eval_metrics['eval_accuracy']:.2%}")
    print(f"Previous best (130 samples): 75.0%")
    print(f"Previous 500 samples result: 70.5%")
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    main()