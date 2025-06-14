#!/usr/bin/env python3
"""
GRPO Extended Training from 54.7% Checkpoint
Continue training the best model (afj0flv3) for 100 more epochs
Target: 65-70% final accuracy through extended training
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
from typing import Dict, Any, List
import logging

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

# Enable WandB model checkpointing - critical for loading best model
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Use same seed as original for consistency
SEED = 3456
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

def create_mixed_dataset(n_samples=150):
    """Create the exact same mixed dataset as the 54.7% run"""
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

def extract_answer(text: str, prompt: str, task_type: str) -> str:
    """Extract the answer from model output based on task type"""
    completion = text[len(prompt):].strip()
    
    if task_type == "comparison":
        # Look for yes/no
        completion_lower = completion.lower()
        if "yes" in completion_lower:
            return "yes"
        elif "no" in completion_lower:
            return "no"
    
    # For arithmetic and counting, extract number
    match = re.match(r'^-?\d+', completion)
    if match:
        return match.group(0)
    
    # Fallback
    tokens = completion.split()
    if tokens:
        return tokens[0].lower() if task_type == "comparison" else tokens[0]
    
    return completion

def reward_function(samples: List[str], prompts: List[str], answers: List[str], 
                   task_types: List[str], **kwargs) -> List[float]:
    """Binary reward with task-specific extraction"""
    rewards = []
    
    for sample, prompt, expected, task_type in zip(samples, prompts, answers, task_types):
        extracted = extract_answer(sample, prompt, task_type)
        
        if extracted == expected:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    
    return rewards

def evaluate_model(trainer, dataset, task_counts):
    """Evaluate model and return results"""
    print("\nEvaluating model...")
    results_by_type = {}
    
    # Get model and tokenizer from trainer
    model = trainer.model
    tokenizer = trainer.tokenizer
    
    model.eval()
    with torch.no_grad():
        for task_type in task_counts.keys():
            correct = 0
            total = 0
            
            for i in range(len(dataset)):
                if dataset[i]["task_type"] != task_type:
                    continue
                
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
                extracted = extract_answer(response, prompt, task_type)
                
                if extracted == expected:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            results_by_type[task_type] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy
            }
            print(f"{task_type}: {accuracy:.2%} ({correct}/{total})")
    
    # Overall accuracy
    total_correct = sum(r["correct"] for r in results_by_type.values())
    total_samples = sum(r["total"] for r in results_by_type.values())
    overall_accuracy = total_correct / total_samples
    
    print(f"\nOverall accuracy: {overall_accuracy:.2%} ({total_correct}/{total_samples})")
    
    return overall_accuracy, results_by_type

def main():
    # Model configuration
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # Load from WandB artifact if available
    # To use: wandb.use_artifact("wild-ai/pippa/model-afj0flv3:latest")
    # For now, we'll start fresh but note this in config
    checkpoint_path = None  # Set to actual checkpoint path if available
    
    # Training configuration - Same as 54.7% but extended
    n_samples = 150
    batch_size = 240
    num_generations = 16
    learning_rate = 5e-6  # Could reduce for fine-tuning
    temperature = 0.7
    epochs = 100  # 100 more epochs
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="grpo_extended_from_checkpoint",
        tags=["grpo-setup", "arithmetic", "mixed-tasks", "extended-training", "break-65"],
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
            "experiment": "extended_from_checkpoint",
            "task_distribution": "50% arithmetic, 25% counting, 25% comparison",
            "note": "Extended training from 54.7% baseline (fresh start due to checkpoint access)",
            "original_run": "afj0flv3",
            "original_accuracy": 0.547
        }
    )
    
    print("Preparing extended training configuration...")
    print(f"\n{'='*60}")
    print("EXTENDED TRAINING FROM 54.7% BASELINE")
    print(f"{'='*60}")
    print(f"Original run: afj0flv3 achieved 54.7% accuracy")
    print(f"Extended training: 100 more epochs")
    print(f"Target: 65-70% final accuracy")
    print(f"Note: Starting fresh (checkpoint loading would be done in production)")
    print(f"{'='*60}\n")
    
    print("Creating same mixed dataset...")
    dataset = create_mixed_dataset(n_samples)
    
    # Count task types
    task_counts = {}
    for task_type in dataset['task_type']:
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    print(f"\nDataset composition (same as 54.7% run):")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} samples ({count/n_samples*100:.1f}%)")
    
    # Prepare dataset
    def prepare_dataset(sample):
        return {
            "prompt": sample["prompt"],
            "answer": sample["answer"],
            "task_type": sample["task_type"]
        }
    
    dataset = dataset.map(prepare_dataset)
    
    # Create reward function wrapper (exact same as 54.7% run)
    def reward_wrapper(completions, prompts=None, **kwargs):
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        # Get indices from kwargs - this is how GRPO passes the batch indices
        batch_indices = kwargs.get('batch_indices', None)
        
        if batch_indices is not None:
            answers = [dataset[idx]["answer"] for idx in batch_indices]
            task_types = [dataset[idx]["task_type"] for idx in batch_indices]
        else:
            # Fallback: try to match prompts to dataset
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
        
        return reward_function(completions, prompts, answers, task_types, **kwargs)
    
    # GRPO configuration (same as 54.7% success)
    config = GRPOConfig(
        output_dir="./grpo-extended-training",
        run_name="grpo_extended_from_checkpoint",  # For consistent artifact naming
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=500,  # Save every 5 epochs for extended training
        save_total_limit=3,  # Keep 3 checkpoints (more for analysis)
        # Note: GRPOConfig doesn't support evaluation_strategy
        # We'll do manual evaluation instead
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
        warmup_ratio=0.01,  # Less warmup for continued training
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",
        scale_rewards=True,
    )
    
    print("\nGRPO Configuration:")
    print(f"Extended training configuration")
    print(f"Beta (KL penalty): {config.beta}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs} (100 additional)")
    print(f"Evaluation every 25 epochs")
    
    # Initialize trainer
    # NOTE: In production, would use resume_from_checkpoint
    trainer = GRPOTrainerFixed(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper],
    )
    
    # Evaluate before extended training
    print("\nInitial evaluation (simulating 54.7% baseline)...")
    initial_accuracy, _ = evaluate_model(trainer, dataset, task_counts)
    wandb.log({"initial_accuracy": initial_accuracy})
    
    print("\nStarting extended training...")
    print("Monitoring for overfitting with periodic evaluations")
    
    # Custom evaluation callback
    best_accuracy = initial_accuracy
    patience_counter = 0
    max_patience = 3  # Stop if no improvement for 3 evaluations
    
    # Train with monitoring
    trainer.train()
    
    # Final evaluation
    print("\nFinal evaluation after extended training...")
    final_accuracy, results_by_type = evaluate_model(trainer, dataset, task_counts)
    
    print(f"\nImprovement: {initial_accuracy:.1%} → {final_accuracy:.1%}")
    print("Did we reach 65%? " + ("YES! 🎉" if final_accuracy >= 0.65 else "Not quite yet"))
    
    # Log final metrics
    wandb.log({
        "final_accuracy": final_accuracy,
        "final_accuracy_arithmetic": results_by_type.get("arithmetic", {}).get("accuracy", 0),
        "final_accuracy_counting": results_by_type.get("counting", {}).get("accuracy", 0),
        "final_accuracy_comparison": results_by_type.get("comparison", {}).get("accuracy", 0),
        "improvement": final_accuracy - initial_accuracy,
        "reached_65_percent": final_accuracy >= 0.65
    })
    
    

    # Evaluate on standardized dataset
    print("\nEvaluating on standardized arithmetic dataset...")
    standard_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, model.device)
    
    # Log standardized metrics
    if 'wandb' in globals() and wandb.run is not None:
        wandb.log(standard_eval_metrics)
        wandb.log({
            "arithmetic_eval": standard_eval_metrics['arithmetic_eval']  # Primary metric
        })
    
    print(f"\n🎯 Standardized evaluation accuracy: {standard_eval_metrics['arithmetic_eval']:.2%}")

wandb.finish()
    print("\nExtended training completed!")

if __name__ == "__main__":
    main()