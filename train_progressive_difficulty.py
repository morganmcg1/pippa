#!/usr/bin/env python3
"""
Progressive difficulty training - start with very easy problems (0-5), 
then gradually increase difficulty as model improves.
Builds on the smaller numbers success by starting even simpler.
"""

import torch
from datasets import Dataset, load_dataset
from transformers import TrainerCallback
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

class ProgressiveDifficultyCallback(TrainerCallback):
    """Callback to increase difficulty and run periodic evaluation."""
    
    def __init__(self, eval_every_n_epochs=5):
        self.eval_every_n_epochs = eval_every_n_epochs
        self.last_eval_epoch = -1
        self.current_max_number = 5  # Start with 0-5
        self.difficulty_schedule = {
            0: 5,    # Epochs 0-14: numbers 0-5
            15: 10,  # Epochs 15-29: numbers 0-10
            30: 15,  # Epochs 30-44: numbers 0-15
            45: 20,  # Epochs 45+: numbers 0-20
        }
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Check if we should increase difficulty."""
        current_epoch = int(state.epoch) if state.epoch else 0
        
        # Update difficulty based on schedule
        for epoch_threshold, max_num in sorted(self.difficulty_schedule.items(), reverse=True):
            if current_epoch >= epoch_threshold:
                if self.current_max_number != max_num:
                    self.current_max_number = max_num
                    print(f"\n{'='*60}")
                    print(f"Increasing difficulty at epoch {current_epoch}")
                    print(f"Now using numbers 0-{self.current_max_number}")
                    print(f"{'='*60}\n")
                    
                    # Regenerate dataset with new difficulty
                    trainer = kwargs.get("model")
                    if hasattr(trainer, "train_dataset"):
                        new_dataset = create_progressive_dataset(
                            len(trainer.train_dataset), 
                            self.current_max_number
                        )
                        trainer.train_dataset = new_dataset
                break
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Run periodic evaluation."""
        if logs is None:
            return
        
        current_epoch = logs.get("epoch", 0)
        
        # Check if we should evaluate
        if current_epoch >= self.last_eval_epoch + self.eval_every_n_epochs:
            self.last_eval_epoch = int(current_epoch)
            
            print(f"\n{'='*60}")
            print(f"Running periodic evaluation at epoch {current_epoch:.1f}")
            print(f"Current difficulty: numbers 0-{self.current_max_number}")
            print(f"{'='*60}")
            
            # Get model and tokenizer from trainer
            trainer = kwargs.get("model")
            if hasattr(trainer, "model") and hasattr(trainer, "tokenizer"):
                model = trainer.model
                tokenizer = trainer.tokenizer
                device = model.device
                
                # Run evaluation
                eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
                
                # Log with epoch prefix
                epoch_metrics = {
                    f"periodic_eval/epoch_{int(current_epoch)}/arithmetic_eval": eval_metrics['standardized_eval_accuracy'],
                    f"periodic_eval/epoch_{int(current_epoch)}/correct": eval_metrics['standardized_eval_correct'],
                    "periodic_eval/current_accuracy": eval_metrics['standardized_eval_accuracy'],
                    "periodic_eval/current_epoch": current_epoch,
                    "periodic_eval/current_max_number": self.current_max_number
                }
                
                # Add difficulty breakdowns
                for key, value in eval_metrics.items():
                    if key.startswith("eval_"):
                        epoch_metrics[f"periodic_eval/epoch_{int(current_epoch)}/{key}"] = value
                
                wandb.log(epoch_metrics)
                
                print(f"Epoch {current_epoch:.1f} - Standardized eval: {eval_metrics['standardized_eval_accuracy']:.2%}")

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
    """Evaluate on standardized dataset."""
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
        'standardized_eval_accuracy': overall_accuracy,
        'standardized_eval_correct': correct_total,
        'standardized_eval_total': len(eval_dataset),
        **difficulty_accs,
        **operation_accs
    }

def create_progressive_dataset(n_samples=150, max_number=5):
    """Create dataset with current difficulty level."""
    prompts = []
    answers = []
    
    # Start with just addition for very easy, then add more operations
    if max_number <= 5:
        operations = ['+']  # Very easy - addition only
    elif max_number <= 10:
        operations = ['+', '-']  # Easy - add subtraction
    else:
        operations = ['+', '-', '*']  # Medium+ - all operations
    
    for _ in range(n_samples):
        a = random.randint(0, max_number)
        b = random.randint(0, max_number)
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
    batch_size = 240  # Slightly smaller for stability
    num_generations = 16
    learning_rate = 5e-6
    temperature = 0.7
    epochs = 60
    eval_every_n_epochs = 5
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="progressive_difficulty",
        tags=["grpo-setup", "standardized-eval", "periodic-eval", "progressive", "curriculum"],
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
            "eval_every_n_epochs": eval_every_n_epochs,
            "difficulty_schedule": "0-5 → 0-10 → 0-15 → 0-20"
        }
    )
    
    print("Creating initial dataset (0-5, addition only)...")
    dataset = create_progressive_dataset(n_samples, max_number=5)
    
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
        output_dir="./grpo-progressive-difficulty",
        run_name="progressive_difficulty",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=10,
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
    
    print("\nProgressive Difficulty Training:")
    print("- Epochs 0-14: Numbers 0-5, addition only")
    print("- Epochs 15-29: Numbers 0-10, addition & subtraction")
    print("- Epochs 30-44: Numbers 0-15, all operations")
    print("- Epochs 45-60: Numbers 0-20, all operations")
    print("\nThis curriculum should help the model build confidence gradually.")
    
    # Initialize trainer with callback
    trainer = GRPOTrainerFixed(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper],
        callbacks=[ProgressiveDifficultyCallback(eval_every_n_epochs=eval_every_n_epochs)]
    )
    
    print("\nStarting progressive difficulty training...")
    trainer.train()
    
    print("\nTraining completed!")
    
    # Get model and tokenizer for final evaluation
    model = trainer.model
    tokenizer = trainer.tokenizer
    device = model.device
    
    # Final evaluation
    print("\nRunning final standardized evaluation...")
    final_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
    
    # Log final metrics
    wandb.log({
        "final/arithmetic_eval": final_eval_metrics['standardized_eval_accuracy'],
        **{f"final/{k}": v for k, v in final_eval_metrics.items()}
    })
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Final standardized eval accuracy: {final_eval_metrics['standardized_eval_accuracy']:.2%}")
    print(f"Base model baseline: ~38%")
    print(f"Previous best (smaller numbers): 45.5%")
    print(f"Improvement over baseline: {final_eval_metrics['standardized_eval_accuracy'] - 0.38:.2%}")
    print(f"{'='*60}")
    
    wandb.finish()

if __name__ == "__main__":
    main()