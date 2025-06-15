#!/usr/bin/env python3
"""
GRPO training with manual periodic evaluation that actually works.
This version manually triggers evaluation at specific intervals.
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
from typing import Dict, Any, List, Optional

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
    """
    Evaluate model on the standardized arithmetic evaluation dataset.
    """
    # Load standardized evaluation dataset
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    
    # Initialize results tracking
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
            
            # Extract answer
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
    
    # Calculate difficulty accuracies
    difficulty_accs = {}
    for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        if diff in results_by_difficulty:
            stats = results_by_difficulty[diff]
            acc = stats['correct'] / stats['total']
            difficulty_accs[f"eval_{diff}_accuracy"] = acc
    
    # Calculate operation accuracies
    operation_accs = {}
    for op in ['+', '-', '*', '/']:
        if op in results_by_operation:
            stats = results_by_operation[op]
            acc = stats['correct'] / stats['total']
            op_name = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op]
            operation_accs[f"eval_{op_name}_accuracy"] = acc
    
    # Return all metrics
    return {
        'eval_accuracy': overall_accuracy,
        'eval_correct': correct_total,
        'eval_total': len(eval_dataset),
        **difficulty_accs,
        **operation_accs
    }

class ManualEvalCallback(TrainerCallback):
    """Callback that manually runs evaluation at specific epochs."""
    
    def __init__(self, eval_every_n_epochs: int = 5):
        self.eval_every_n_epochs = eval_every_n_epochs
        self.last_eval_epoch = -1
        self.trainer = None
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Store reference to trainer at the beginning of training."""
        # The trainer is passed in kwargs
        self.trainer = kwargs.get('model', None)
        if self.trainer is None:
            # Try to get it from the first positional argument
            for key, value in kwargs.items():
                if hasattr(value, 'model') and hasattr(value, 'tokenizer'):
                    self.trainer = value
                    break
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Run evaluation at the end of specific epochs."""
        current_epoch = state.epoch if state.epoch else 0
        
        # Check if we should evaluate
        if int(current_epoch) % self.eval_every_n_epochs == 0 and int(current_epoch) > self.last_eval_epoch:
            self.last_eval_epoch = int(current_epoch)
            
            # Get trainer from stored reference or kwargs
            trainer = self.trainer or kwargs.get('model', None)
            
            if trainer and hasattr(trainer, 'model') and hasattr(trainer, 'tokenizer'):
                print(f"\n{'='*60}")
                print(f"Running manual evaluation at epoch {current_epoch}")
                print(f"{'='*60}")
                
                model = trainer.model
                tokenizer = trainer.tokenizer
                device = model.device if hasattr(model, 'device') else next(model.parameters()).device
                
                # Run evaluation
                eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
                
                # Log metrics with eval prefix
                logged_metrics = {
                    **eval_metrics,
                    "eval_epoch": current_epoch,
                    f"eval_at_epoch_{int(current_epoch)}/accuracy": eval_metrics['eval_accuracy'],
                }
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log(logged_metrics, step=state.global_step)
                
                print(f"Evaluation complete - Accuracy: {eval_metrics['eval_accuracy']:.2%}")
                print(f"{'='*60}\n")

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

def create_training_dataset(n_samples=150):
    """Create arithmetic dataset for training."""
    prompts = []
    answers = []
    
    operations = ['+', '-', '*']
    for _ in range(n_samples):
        a = random.randint(0, 10)  # Smaller numbers for better performance
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
    n_samples = 150
    batch_size = 240
    num_generations = 16
    learning_rate = 5e-6
    temperature = 0.7
    epochs = 50
    eval_every_n_epochs = 5
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="grpo_manual_eval",
        tags=["grpo-setup", "standardized-eval", "periodic-eval", "arithmetic", "manual-eval"],
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
            "eval_every_n_epochs": eval_every_n_epochs
        }
    )
    
    print("Creating training dataset...")
    dataset = create_training_dataset(n_samples)
    
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
            # Fallback
            prompt_to_answer = {d["prompt"]: d["answer"] for d in dataset}
            answers = [prompt_to_answer.get(p, "") for p in prompts]
        
        return reward_function(completions, prompts, answers, **kwargs)
    
    # GRPO configuration
    config = GRPOConfig(
        output_dir="./grpo-manual-eval",
        run_name="grpo_manual_eval",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
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
    print(f"Beta (KL penalty): {config.beta}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Generations per prompt: {num_generations}")
    print(f"Epochs: {epochs}")
    print(f"Manual eval every: {eval_every_n_epochs} epochs")
    
    # Initialize trainer with callback
    trainer = GRPOTrainerFixed(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper],
        callbacks=[ManualEvalCallback(eval_every_n_epochs=eval_every_n_epochs)]
    )
    
    print("\nStarting training with manual periodic evaluation...")
    trainer.train()
    
    print("\nTraining completed!")
    
    # Get model and tokenizer for final evaluation
    model = trainer.model
    tokenizer = trainer.tokenizer
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    
    # Final evaluation
    print("\nRunning final standardized evaluation...")
    final_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
    
    # Log final metrics
    wandb.log({
        "final/eval_accuracy": final_eval_metrics['eval_accuracy'],
        **{f"final/{k}": v for k, v in final_eval_metrics.items()}
    })
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Final standardized eval accuracy: {final_eval_metrics['eval_accuracy']:.2%}")
    print(f"Base model baseline: ~38%")
    print(f"Improvement over baseline: {final_eval_metrics['eval_accuracy'] - 0.38:.2%}")
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    main()