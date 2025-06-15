#!/usr/bin/env python3
"""
GRPO Arithmetic Training - Higher Temperature with FULL Diversity
Uses temperature 1.0 AND maintains 16 generations per prompt with batch_size=256
This restores the proper generation diversity for effective GRPO training
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
SEED = 2025
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

def create_simple_arithmetic_dataset(n_samples=100):
    """Create a dataset of simple arithmetic problems"""
    prompts = []
    answers = []
    
    operations = ['+', '-', '*']
    
    for _ in range(n_samples):
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
    
    # Training configuration - RESTORED DIVERSITY
    n_samples = 100
    batch_size = 256  # INCREASED to maintain 16 generations
    num_generations = 16  # RESTORED to baseline value
    learning_rate = 5e-6
    temperature = 1.0  # Higher temperature for diversity
    epochs = 50
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="grpo_arithmetic_higher_temp_full_diversity",
        tags=["grpo-setup", "arithmetic", "overfit", "break-barrier", "high-temp", "full-diversity"],
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
            "experiment": "higher_temperature_full_diversity",
            "effective_batch_size": batch_size,
            "note": "Restored 16 generations with larger batch size"
        }
    )
    
    print("Preparing training configuration...")
    print(f"\n{'='*60}")
    print("FULL DIVERSITY RESTORATION")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size} (increased from 64)")
    print(f"Generations: {num_generations} (restored from 4 to 16)")
    print(f"Temperature: {temperature} (high for diversity)")
    print(f"Effective batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    print("Creating dataset...")
    dataset = create_simple_arithmetic_dataset(n_samples)
    
    print("\nSample problems:")
    for i in range(5):
        print(f"[{i}] {dataset['prompt'][i]} → {dataset['answer'][i]}")
    
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
        answers = [dataset[i]["answer"] for i in range(len(prompts))]
        return reward_function(completions, prompts, answers, **kwargs)
    
    # GRPO configuration with FULL DIVERSITY
    config = GRPOConfig(
        output_dir="./grpo-arithmetic-high-temp-full-div",
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
    print(f"Temperature: {temperature} (high for diversity)")
    print(f"Beta (KL penalty): {config.beta}")
    print(f"Batch size: {batch_size} (supports 16 generations)")
    print(f"Num generations: {num_generations} (full diversity restored)")
    
    # Initialize trainer
    trainer = GRPOTrainerFixed(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper],
    )
    
    print("\nStarting training with FULL generation diversity...")
    print("Expected: Much better learning signal than 4-generation experiments")
    trainer.train()
    
    # Evaluate final performance
    print("\nEvaluating final model...")
    correct = 0
    total = len(dataset)
    
    # Get model and tokenizer from trainer
    model = trainer.model
    tokenizer = trainer.tokenizer
    
    model.eval()
    with torch.no_grad():
        for i in range(total):
            prompt = dataset[i]["prompt"]
            expected = dataset[i]["answer"]
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,  # Use lower temp for eval
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            extracted = extract_answer(response, prompt)
            
            if extracted == expected:
                correct += 1
    
    accuracy = correct / total
    print(f"\nFinal accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Log final metrics
    wandb.log({
        "final_accuracy": accuracy,
        "final_correct": correct,
        "final_total": total
    })
    
    wandb.finish()
    print("\nTraining completed!")

if __name__ == "__main__":
    main()