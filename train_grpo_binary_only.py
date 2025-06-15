#!/usr/bin/env python3
"""
GRPO training on ONLY binary arithmetic (0s and 1s).
Hypothesis: Drastically reducing the output space to just {0, 1, 2} might finally achieve high accuracy.
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

def evaluate_on_binary_test(model, tokenizer, device) -> Dict[str, float]:
    """Evaluate specifically on binary arithmetic problems."""
    test_problems = []
    
    # Generate all possible binary arithmetic problems
    for a in [0, 1]:
        for b in [0, 1]:
            test_problems.append((f"Calculate: {a} + {b} = ", str(a + b), '+'))
            test_problems.append((f"Calculate: {a} - {b} = ", str(a - b), '-'))
            test_problems.append((f"Calculate: {a} * {b} = ", str(a * b), '*'))
    
    correct = 0
    results_by_op = {'+': {'correct': 0, 'total': 0}, 
                     '-': {'correct': 0, 'total': 0}, 
                     '*': {'correct': 0, 'total': 0}}
    
    model.eval()
    with torch.no_grad():
        for prompt, expected, op in test_problems:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            completion = response[len(prompt):].strip()
            match = re.match(r'^-?\d+', completion)
            predicted = match.group(0) if match else completion.split()[0] if completion else ""
            
            results_by_op[op]['total'] += 1
            if predicted == expected:
                correct += 1
                results_by_op[op]['correct'] += 1
    
    return {
        'binary_accuracy': correct / len(test_problems),
        'binary_correct': correct,
        'binary_total': len(test_problems),
        'binary_add_accuracy': results_by_op['+']['correct'] / results_by_op['+']['total'],
        'binary_sub_accuracy': results_by_op['-']['correct'] / results_by_op['-']['total'],
        'binary_mul_accuracy': results_by_op['*']['correct'] / results_by_op['*']['total'],
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

def create_binary_dataset(n_samples=100):
    """Create arithmetic dataset with only 0s and 1s."""
    prompts = []
    answers = []
    
    operations = ['+', '-', '*']
    
    # Generate diverse binary problems
    for _ in range(n_samples):
        a = random.randint(0, 1)
        b = random.randint(0, 1)
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

def reward_function(samples: List[str], prompts: List[str], answers: List[str], **kwargs) -> List[float]:
    """Binary reward function"""
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
    
    # Training configuration
    n_samples = 100
    batch_size = 64  # Smaller batch for binary
    num_generations = 16
    learning_rate = 5e-6
    temperature = 0.7
    epochs = 50
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="grpo_binary_only",
        tags=["grpo-setup", "standardized-eval", "binary-arithmetic", "minimal-space"],
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
            "number_range": "0-1 (binary only)",
            "output_space": "{-1, 0, 1, 2}"
        }
    )
    
    print("Creating binary arithmetic dataset (0s and 1s only)...")
    dataset = create_binary_dataset(n_samples)
    
    # Print dataset statistics
    unique_prompts = set(dataset["prompt"])
    unique_answers = set(dataset["answer"])
    print(f"Unique prompts: {len(unique_prompts)}")
    print(f"Unique answers: {unique_answers}")
    print(f"Total output space: {len(unique_answers)} possible answers")
    
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
        
        return reward_function(completions, prompts, answers, **kwargs)
    
    # GRPO configuration
    config = GRPOConfig(
        output_dir="./grpo-binary-only",
        run_name="grpo_binary_only",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="wandb",
        remove_unused_columns=False,
        log_completions=True,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=8,  # Shorter for binary
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
    print("BINARY ARITHMETIC EXPERIMENT")
    print("="*60)
    print("- Numbers: Only 0 and 1")
    print("- Possible answers: {-1, 0, 1, 2}")
    print("- Total unique problems: ~12")
    print("- Hypothesis: Tiny output space → finally achieve 90%+ accuracy")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = GRPOTrainerFixed(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper]
    )
    
    print("Starting training with binary arithmetic...")
    trainer.train()
    
    print("\nTraining completed!")
    
    # Evaluation
    model = trainer.model
    tokenizer = trainer.tokenizer
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    
    # Binary-specific evaluation
    print("\nRunning binary arithmetic evaluation...")
    binary_eval = evaluate_on_binary_test(model, tokenizer, device)
    
    wandb.log({
        **{f"final/{k}": v for k, v in binary_eval.items()}
    })
    
    # Standard evaluation
    print("\nRunning standard evaluation...")
    final_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
    
    wandb.log({
        "final/eval_accuracy": final_eval_metrics['eval_accuracy'],
        **{f"final/{k}": v for k, v in final_eval_metrics.items()}
    })
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS - Binary Arithmetic")
    print("="*60)
    print(f"Binary test accuracy: {binary_eval['binary_accuracy']:.2%}")
    print(f"  Addition (0+0, 0+1, 1+0, 1+1): {binary_eval['binary_add_accuracy']:.2%}")
    print(f"  Subtraction: {binary_eval['binary_sub_accuracy']:.2%}")
    print(f"  Multiplication: {binary_eval['binary_mul_accuracy']:.2%}")
    print(f"\nStandardized eval: {final_eval_metrics['eval_accuracy']:.2%}")
    print(f"Previous best: 45.5%")
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    main()