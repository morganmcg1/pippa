#!/usr/bin/env python3
"""GRPO arithmetic training with higher KL penalty (beta=0.2)."""

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import wandb
import os
from dotenv import load_dotenv
import random
import re
from typing import Dict, Any

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

class GRPOTrainerFixed(GRPOTrainer):
    """GRPOTrainer with fixed completion logging."""
    
    def log(self, logs: Dict[str, Any], start_time: float = None) -> None:
        """Override log method to fix completion printing."""
        # Call parent log method but catch the error
        try:
            super().log(logs, start_time)
        except AttributeError as e:
            if "add_section" in str(e):
                # If it's the table error, print completions manually
                if self.args.log_completions and self.state.global_step % self.args.logging_steps == 0:
                    self._print_completions_simple()
                # Still log the metrics
                if hasattr(self, '_wandb'):
                    self._wandb.log(logs, step=self.state.global_step)
            else:
                raise
    
    def _print_completions_simple(self):
        """Simple completion printing without rich tables."""
        if not hasattr(self, '_recent_completions'):
            return
            
        print("\n" + "="*60)
        print(f"Step {self.state.global_step} - Sample Completions")
        print("="*60)
        
        # Show up to 3 samples
        num_samples = min(3, len(self._recent_completions))
        for i in range(num_samples):
            if i < len(self._recent_completions):
                sample = self._recent_completions[i]
                print(f"\nSample {i+1}:")
                print(f"Prompt: {sample.get('prompt', 'N/A')}")
                print(f"Completion: {sample.get('completion', 'N/A')}")
                print(f"Reward: {sample.get('reward', 'N/A')}")
        
        print("="*60 + "\n")
        
        # Clear stored completions
        self._recent_completions = []
    
    def _generate_completions(self, prompts, **generation_kwargs):
        """Override to capture completions for logging."""
        completions = super()._generate_completions(prompts, **generation_kwargs)
        
        # Store completions for our simple logging
        if not hasattr(self, '_recent_completions'):
            self._recent_completions = []
            
        # Decode and store a few samples
        if hasattr(self, 'tokenizer'):
            for i in range(min(3, len(completions))):
                if i < len(prompts):
                    completion_text = self.tokenizer.decode(completions[i], skip_special_tokens=True)
                    self._recent_completions.append({
                        "prompt": prompts[i],
                        "completion": completion_text,
                        "reward": None  # Will be updated later
                    })
        
        return completions

def create_simple_arithmetic_dataset(n_samples: int = 100):
    """Simple arithmetic problems with verifiable answers."""
    prompts = []
    for _ in range(n_samples):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        op = random.choice(['+', '-', '*'])
        
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else:  # multiplication
            answer = a * b
            
        prompts.append({
            "prompt": f"Calculate: {a} {op} {b} = ",
            "expected": str(answer)
        })
    return prompts

def main():
    # Configuration with higher beta
    n_samples = 100
    batch_size = 64
    num_generations = 16
    lr = 5e-6
    temperature = 0.7
    epochs = 50  # Reasonable epochs
    seed = 321
    beta = 0.2  # Higher KL penalty
    
    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Initialize wandb
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"grpo_arithmetic_beta{beta}_b{batch_size}_g{num_generations}",
        config={
            "task": "arithmetic_higher_beta",
            "n_samples": n_samples,
            "batch_size": batch_size,
            "num_generations": num_generations,
            "learning_rate": lr,
            "temperature": temperature,
            "epochs": epochs,
            "seed": seed,
            "beta": beta,
            "model": "Qwen/Qwen2-0.5B-Instruct",
            "log_completions": True,
            "wandb_log_unique_prompts": True
        },
        tags=["grpo-setup", "overfit", "arithmetic", "higher-beta"]
    )
    
    # Create dataset
    data = create_simple_arithmetic_dataset(n_samples)
    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in data])
    expected_answers = {d["prompt"]: d["expected"] for d in data}
    
    # Print dataset info
    print(f"\n{'='*60}")
    print(f"ARITHMETIC GRPO WITH HIGHER BETA")
    print(f"{'='*60}")
    print(f"Dataset size: {len(dataset)} problems")
    print(f"Batch size: {batch_size}")
    print(f"Generations per prompt: {num_generations}")
    print(f"Learning rate: {lr}")
    print(f"Temperature: {temperature}")
    print(f"Beta (KL penalty): {beta} (2x higher)")
    print(f"Epochs: {epochs}")
    print(f"\nSample problems:")
    for i in range(min(5, len(data))):
        print(f"  [{i}] {data[i]['prompt']} → {data[i]['expected']}")
    print(f"{'='*60}\n")
    
    # Create reward function
    def reward_function(completions, prompts=None, **kwargs):
        """Binary reward for exact match."""
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            # Extract answer
            answer = completion[len(prompt):].strip()
            # Match first number (including negative)
            match = re.search(r'^-?\d+', answer)
            if match:
                extracted = match.group()
            else:
                extracted = answer.split()[0] if answer else ""
            
            expected = expected_answers.get(prompt, "")
            
            # Binary reward
            if extracted == expected:
                rewards.append(1.0)  # Correct
            else:
                rewards.append(-1.0)  # Incorrect
                
        return rewards
    
    # GRPO config with higher beta
    config = GRPOConfig(
        output_dir=f"./grpo_arithmetic_beta_{seed}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=16,
        max_prompt_length=128,
        beta=beta,  # Higher KL penalty
        loss_type="grpo",
        push_to_hub=False,
        report_to=["wandb"],
        logging_steps=1,  # Log every step
        save_steps=500,
        seed=seed,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        log_completions=True,
        wandb_log_unique_prompts=True,
        # Optimization parameters
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",
        scale_rewards=True,
    )
    
    # Create trainer with fixed logging
    trainer = GRPOTrainerFixed(
        model="Qwen/Qwen2-0.5B-Instruct",
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_function],
    )
    
    # Train
    print(f"Starting training with beta={beta}...")
    trainer.train()
    
    print("\nTraining completed!")
    
    # Test final performance
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = trainer.model
    tokenizer = trainer.tokenizer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    correct = 0
    test_samples = 20
    eval_table = wandb.Table(columns=["prompt", "expected", "generated", "extracted", "is_correct"])
    
    with torch.no_grad():
        for i in range(min(test_samples, len(data))):
            prompt = data[i]['prompt']
            expected = data[i]['expected']
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = completion[len(prompt):].strip()
            match = re.search(r'^-?\d+', answer)
            extracted = match.group() if match else answer.split()[0] if answer else ""
            
            is_correct = extracted == expected
            if is_correct:
                correct += 1
            
            eval_table.add_data(prompt, expected, answer, extracted, is_correct)
                
            print(f"[{i}] {prompt} → {expected}")
            print(f"     Generated: {answer}")
            print(f"     Extracted: {extracted} {'✓' if is_correct else '✗'}")
    
    accuracy = 100 * correct / test_samples
    print(f"\nFinal Accuracy: {correct}/{test_samples} ({accuracy:.1f}%)")
    print(f"{'='*60}")
    
    wandb.log({
        "final_accuracy": accuracy,
        "final_evaluation": eval_table
    })
    
    

    # Evaluate on standardized dataset
    print("\nEvaluating on standardized arithmetic dataset...")
    standard_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
    
    # Log standardized metrics
    if 'wandb' in globals() and wandb.run is not None:
        wandb.log(standard_eval_metrics)
        wandb.log({
            "arithmetic_eval": standard_eval_metrics['arithmetic_eval']  # Primary metric
        })
    
    print(f"\n🎯 Standardized evaluation accuracy: {standard_eval_metrics['arithmetic_eval']:.2%}")

wandb.finish()

if __name__ == "__main__":
    main()