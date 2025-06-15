#!/usr/bin/env python3
"""
Retrain Ultra-Diversity model with standardized evaluation.
Based on run icel5tvz which achieved 50% with 64 generations per prompt.
Now we'll evaluate on the standardized morgan/arithmetic_eval dataset.
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

def evaluate_on_standard_dataset(model, tokenizer, device) -> Dict[str, float]:
    """
    Evaluate model on the standardized arithmetic evaluation dataset.
    This ensures fair comparison across all experiments.
    """
    print("\n" + "="*60)
    print("STANDARDIZED EVALUATION (morgan/arithmetic_eval)")
    print("="*60)
    
    # Load standardized evaluation dataset
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

def create_mixed_dataset(n_samples=150):
    """Create mixed dataset with 0-20 range."""
    prompts = []
    answers = []
    task_types = []
    
    # Define task distribution
    n_arithmetic = n_samples // 2  # 50%
    n_counting = n_samples // 4    # 25%
    n_comparison = n_samples - n_arithmetic - n_counting  # 25%
    
    # 1. Arithmetic tasks (0-20)
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
    
    # 2. Counting tasks
    for _ in range(n_counting):
        # Generate a list of words
        num_words = random.randint(1, 15)
        words = random.sample(['cat', 'dog', 'bird', 'fish', 'tree', 'car', 
                             'book', 'pen', 'chair', 'table', 'phone', 
                             'computer', 'window', 'door', 'light', 'apple',
                             'orange', 'banana', 'grape', 'berry'], num_words)
        sentence = ' '.join(words)
        
        prompt = f"Count the words in: '{sentence}'. Answer: "
        prompts.append(prompt)
        answers.append(str(num_words))
        task_types.append("counting")
    
    # 3. Comparison tasks
    for _ in range(n_comparison):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        prompt = f"Is {a} greater than {b}? Answer yes or no: "
        answer = "yes" if a > b else "no"
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
    
    # Fallback
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
    
    # Training configuration - ultra-diversity with 64 generations
    n_samples = 150
    batch_size = 960  # 64 generations × 15 prompts
    num_generations = 64  # Ultra-high diversity
    learning_rate = 5e-6
    temperature = 0.7
    epochs = 75  # Extended training
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="retrain_ultra_diversity_standard",
        tags=["grpo-setup", "standardized-eval", "ultra-diversity", "64-generations", "retrain"],
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
            "eval_dataset_size": 200,
            "task_distribution": "50% arithmetic (0-20), 25% counting, 25% comparison",
            "based_on_run": "icel5tvz",
            "diversity": "ultra-high (64 generations)"
        }
    )
    
    print("Creating mixed dataset...")
    dataset = create_mixed_dataset(n_samples)
    
    # Show dataset distribution
    print(f"\nDataset created with {len(dataset)} samples:")
    print(f"- Arithmetic: {sum(1 for t in dataset['task_type'] if t == 'arithmetic')} samples")
    print(f"- Counting: {sum(1 for t in dataset['task_type'] if t == 'counting')} samples")
    print(f"- Comparison: {sum(1 for t in dataset['task_type'] if t == 'comparison')} samples")
    
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
            # Fallback
            prompt_to_answer = {d["prompt"]: d["answer"] for d in dataset}
            answers = [prompt_to_answer.get(p, "") for p in prompts]
        
        return reward_function(completions, prompts, answers, **kwargs)
    
    # GRPO configuration
    config = GRPOConfig(
        output_dir="./grpo-ultra-diversity-standard",
        run_name="retrain_ultra_diversity_standard",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="wandb",
        remove_unused_columns=False,
        log_completions=True,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=16,
        max_prompt_length=128,
        seed=SEED,
        # GRPO specific
        beta=0.1,  # KL penalty - important for arithmetic
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
    print(f"Beta (KL penalty): {config.beta}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Generations per prompt: {num_generations} (ULTRA-HIGH DIVERSITY)")
    print(f"Epochs: {epochs}")
    
    # Initialize trainer
    trainer = GRPOTrainerFixed(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper],
    )
    
    print("\nStarting training with ultra-high diversity (64 generations)...")
    trainer.train()
    
    print("\nTraining completed!")
    
    # Get model and tokenizer for evaluation
    model = trainer.model
    tokenizer = trainer.tokenizer
    device = model.device
    
    # Evaluate on standardized dataset
    print("\nEvaluating on standardized arithmetic dataset...")
    standard_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
    
    # Log all metrics
    wandb.log(standard_eval_metrics)
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Standardized eval accuracy: {standard_eval_metrics['arithmetic_eval']:.2%}")
    print(f"Base model baseline: ~38%")
    print(f"Improvement over baseline: {standard_eval_metrics['arithmetic_eval'] - 0.38:.2%}")
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    main()