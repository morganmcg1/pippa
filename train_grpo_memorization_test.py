#!/usr/bin/env python3
"""
GRPO training on just 10 fixed problems - pure memorization test.
If GRPO can't even memorize 10 problems, it confirms fundamental limitations.
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

def evaluate_on_training_set(model, tokenizer, device, dataset) -> Dict[str, float]:
    """Evaluate on the exact training problems to test memorization."""
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            prompt = dataset[i]["prompt"]
            expected = dataset[i]["answer"]
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                temperature=0.1,  # Low temp for memorization test
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            completion = response[len(prompt):].strip()
            match = re.match(r'^-?\d+', completion)
            predicted = match.group(0) if match else completion.split()[0] if completion else ""
            
            if predicted == expected:
                correct += 1
            else:
                print(f"Failed: {prompt}{expected} → {predicted}")
    
    return {
        'memorization_accuracy': correct / len(dataset),
        'memorization_correct': correct,
        'memorization_total': len(dataset)
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

def create_memorization_dataset():
    """Create a fixed set of 10 problems for memorization."""
    # Hand-picked diverse problems that should be easy to memorize
    problems = [
        ("Calculate: 2 + 2 = ", "4"),
        ("Calculate: 5 - 3 = ", "2"),
        ("Calculate: 3 * 3 = ", "9"),
        ("Calculate: 1 + 0 = ", "1"),
        ("Calculate: 4 - 4 = ", "0"),
        ("Calculate: 2 * 5 = ", "10"),
        ("Calculate: 7 + 1 = ", "8"),
        ("Calculate: 6 - 2 = ", "4"),
        ("Calculate: 4 * 2 = ", "8"),
        ("Calculate: 0 + 5 = ", "5"),
    ]
    
    prompts = [p[0] for p in problems]
    answers = [p[1] for p in problems]
    
    # Repeat to create larger dataset (but same 10 unique problems)
    prompts = prompts * 10  # 100 samples total
    answers = answers * 10
    
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
    batch_size = 80  # Must be divisible by 16 and <= 100
    num_generations = 16
    learning_rate = 1e-5  # Higher LR for memorization
    temperature = 0.5  # Lower temp for consistency
    epochs = 100  # Many epochs for memorization
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="grpo_memorization_test",
        tags=["grpo-setup", "standardized-eval", "memorization-test", "10-problems"],
        config={
            "model": model_name,
            "unique_problems": 10,
            "total_samples": 100,
            "batch_size": batch_size,
            "num_generations": num_generations,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "epochs": epochs,
            "seed": SEED,
            "beta": 0.1,
            "eval_dataset": "morgan/arithmetic_eval"
        }
    )
    
    print("Creating memorization dataset (10 fixed problems)...")
    dataset = create_memorization_dataset()
    
    # Print the 10 unique problems
    print("\nThe 10 problems to memorize:")
    seen = set()
    for i in range(len(dataset)):
        problem = f"{dataset[i]['prompt']}{dataset[i]['answer']}"
        if problem not in seen:
            print(f"  {problem}")
            seen.add(problem)
    
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
        output_dir="./grpo-memorization",
        run_name="grpo_memorization_test",
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
        max_completion_length=8,
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
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",
        scale_rewards=True,
    )
    
    print("\n" + "="*60)
    print("MEMORIZATION TEST")
    print("="*60)
    print("- Only 10 unique problems")
    print("- 100 epochs of training")
    print("- If GRPO can't memorize 10 problems, it's fundamentally broken")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = GRPOTrainerFixed(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper]
    )
    
    print("Starting memorization training...")
    trainer.train()
    
    print("\nTraining completed!")
    
    # Evaluation
    model = trainer.model
    tokenizer = trainer.tokenizer
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    
    # Test memorization on training set
    print("\nTesting memorization on training problems...")
    memo_eval = evaluate_on_training_set(model, tokenizer, device, dataset[:10])
    
    wandb.log({
        **{f"final/{k}": v for k, v in memo_eval.items()}
    })
    
    # Standard evaluation
    print("\nRunning standard evaluation...")
    final_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
    
    wandb.log({
        "final/eval_accuracy": final_eval_metrics['eval_accuracy'],
        **{f"final/{k}": v for k, v in final_eval_metrics.items()}
    })
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS - Memorization Test")
    print("="*60)
    print(f"Memorization accuracy (10 problems): {memo_eval['memorization_accuracy']:.2%}")
    print(f"Standardized eval: {final_eval_metrics['eval_accuracy']:.2%}")
    print(f"Previous best: 45.5%")
    print("\nIf memorization < 90%, GRPO is fundamentally broken for arithmetic")
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    main()