#!/usr/bin/env python3
"""
GRPO training with proper periodic evaluation using HuggingFace eval_dataset.
This version properly integrates evaluation into the training loop.
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

class EvaluationCallback(TrainerCallback):
    """Callback to log evaluation metrics with proper naming."""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after an evaluation phase."""
        if metrics:
            # Log evaluation metrics with eval/ prefix
            eval_metrics = {}
            for key, value in metrics.items():
                if key.startswith("eval_"):
                    # Keep the eval_ prefix
                    eval_metrics[key] = value
                else:
                    # Add eval_ prefix if not present
                    eval_metrics[f"eval_{key}"] = value
            
            # Also log current epoch
            if state.epoch:
                eval_metrics["eval_epoch"] = state.epoch
                
            wandb.log(eval_metrics, step=state.global_step)
            
            # Print summary
            if "eval_accuracy" in eval_metrics:
                print(f"\nEpoch {state.epoch:.1f} - Eval accuracy: {eval_metrics['eval_accuracy']:.2%}")

# Custom GRPOTrainer with evaluation support
class GRPOTrainerWithEval(GRPOTrainer):
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
    
    def compute_metrics(self, eval_preds):
        """Compute evaluation metrics for the standardized dataset."""
        # For GRPO, we need to handle the evaluation differently
        # since it's not a standard supervised task
        if hasattr(self, 'eval_dataset') and self.eval_dataset is not None:
            # Run evaluation on the eval dataset
            model = self.model
            tokenizer = self.tokenizer
            device = model.device
            
            correct = 0
            total = 0
            
            model.eval()
            with torch.no_grad():
                for sample in self.eval_dataset:
                    prompt = sample['prompt']
                    expected = sample['answer']
                    
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
                    
                    if predicted == expected:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            return {
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }
        
        return {}

def create_training_dataset(n_samples=150):
    """Create arithmetic dataset for training."""
    prompts = []
    answers = []
    
    operations = ['+', '-', '*']
    for _ in range(n_samples):
        a = random.randint(0, 10)  # Using smaller numbers for better performance
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

def create_eval_dataset():
    """Load the standardized evaluation dataset."""
    # Load the standardized dataset
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    
    # Convert to the format expected by our evaluation
    def format_sample(sample):
        return {
            "prompt": sample["prompt"],
            "answer": sample["answer"]
        }
    
    return eval_dataset.map(format_sample)

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
    eval_steps = 250  # Evaluate every 250 steps
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="grpo_periodic_eval_fixed",
        tags=["grpo-setup", "standardized-eval", "periodic-eval", "arithmetic", "fixed-eval"],
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
            "eval_steps": eval_steps
        }
    )
    
    print("Creating training dataset...")
    train_dataset = create_training_dataset(n_samples)
    
    print("Loading evaluation dataset...")
    eval_dataset = create_eval_dataset()
    print(f"Evaluation dataset size: {len(eval_dataset)} samples")
    
    # Prepare datasets
    def prepare_dataset(sample):
        return {
            "prompt": sample["prompt"],
            "answer": sample["answer"]
        }
    
    train_dataset = train_dataset.map(prepare_dataset)
    
    # Create reward function wrapper
    def reward_wrapper(completions, prompts=None, **kwargs):
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        batch_indices = kwargs.get('batch_indices', None)
        
        if batch_indices is not None:
            answers = [train_dataset[idx]["answer"] for idx in batch_indices]
        else:
            # Fallback
            prompt_to_answer = {d["prompt"]: d["answer"] for d in train_dataset}
            answers = [prompt_to_answer.get(p, "") for p in prompts]
        
        return reward_function(completions, prompts, answers, **kwargs)
    
    # GRPO configuration with evaluation
    config = GRPOConfig(
        output_dir="./grpo-periodic-eval-fixed",
        run_name="grpo_periodic_eval_fixed",
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
        # Evaluation settings
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        per_device_eval_batch_size=64,
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
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
    )
    
    print("\nGRPO Configuration:")
    print(f"Beta (KL penalty): {config.beta}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Generations per prompt: {num_generations}")
    print(f"Epochs: {epochs}")
    print(f"Evaluation every: {eval_steps} steps")
    
    # Initialize trainer with eval dataset
    trainer = GRPOTrainerWithEval(
        model=model_name,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[reward_wrapper],
        callbacks=[EvaluationCallback()]
    )
    
    print("\nStarting training with periodic evaluation...")
    trainer.train()
    
    print("\nTraining completed!")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_metrics = trainer.evaluate()
    
    # Log final metrics
    final_logged = {f"final/{k}": v for k, v in final_metrics.items()}
    wandb.log(final_logged)
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    if "eval_accuracy" in final_metrics:
        print(f"Final eval accuracy: {final_metrics['eval_accuracy']:.2%}")
        print(f"Base model baseline: ~38%")
        print(f"Improvement over baseline: {final_metrics['eval_accuracy'] - 0.38:.2%}")
    else:
        print("Final metrics:", final_metrics)
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    main()