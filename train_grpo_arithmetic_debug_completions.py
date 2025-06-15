#!/usr/bin/env python3
"""GRPO arithmetic training with log_completions debugging."""

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import wandb
import os
from dotenv import load_dotenv
import random
import re

# Monkey patch to fix the table.add_section() issue
import trl.trainer.utils as trl_utils
from rich.console import Console
from rich.table import Table

def patched_print_prompt_completions_sample(
    prompts,
    completions,
    reward_function,
    reward_model,
    model_tokenizer,
    ref_tokenizer,
    log_type="train",
):
    """Fixed version without add_section() call."""
    console = Console()
    
    # Create a simple table without sections
    table = Table(title=f"{log_type.capitalize()} Prompt Completions Sample")
    table.add_column("Prompt", style="cyan", no_wrap=False)
    table.add_column("Completion", style="magenta", no_wrap=False)
    table.add_column("Reward", style="green")
    
    # Sample a few examples to display
    num_samples = min(3, len(prompts))
    indices = random.sample(range(len(prompts)), num_samples)
    
    for idx in indices:
        prompt = prompts[idx]
        completion = completions[idx]
        
        # Calculate reward if function provided
        if reward_function:
            try:
                rewards = reward_function([completion], prompts=[prompt])
                reward = rewards[0] if rewards else "N/A"
            except:
                reward = "Error"
        else:
            reward = "N/A"
            
        table.add_row(prompt, completion, str(reward))
    
    console.print(table)
    print()  # Add spacing

# Apply the monkey patch
trl_utils.print_prompt_completions_sample = patched_print_prompt_completions_sample

# Load environment variables
load_dotenv()

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
    # Configuration for debugging log_completions
    n_samples = 100
    batch_size = 64
    num_generations = 16
    lr = 5e-6
    temperature = 0.7
    epochs = 20  # Shorter for debugging
    seed = 999
    beta = 0.1  # KL penalty
    
    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Initialize wandb
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"grpo_arithmetic_debug_completions_b{batch_size}_g{num_generations}",
        config={
            "task": "arithmetic_debug_completions",
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
        tags=["grpo-setup", "overfit", "arithmetic", "debug-completions"]
    )
    
    # Create dataset
    data = create_simple_arithmetic_dataset(n_samples)
    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in data])
    expected_answers = {d["prompt"]: d["expected"] for d in data}
    
    # Print dataset info
    print(f"\n{'='*60}")
    print(f"ARITHMETIC GRPO WITH COMPLETION LOGGING DEBUG")
    print(f"{'='*60}")
    print(f"Dataset size: {len(dataset)} problems")
    print(f"Batch size: {batch_size}")
    print(f"Generations per prompt: {num_generations}")
    print(f"Learning rate: {lr}")
    print(f"Temperature: {temperature}")
    print(f"Beta (KL penalty): {beta}")
    print(f"Log Completions: ENABLED (with patch)")
    print(f"WandB Tables: ENABLED")
    print(f"\nSample problems:")
    for i in range(min(10, len(data))):
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
    
    # GRPO config with completion logging enabled
    config = GRPOConfig(
        output_dir=f"./grpo_arithmetic_debug_{seed}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=16,
        max_prompt_length=128,
        beta=beta,  # KL penalty for stability
        loss_type="grpo",  # Standard GRPO
        push_to_hub=False,
        report_to=["wandb"],
        logging_steps=1,
        save_steps=100,
        seed=seed,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        log_completions=True,  # ENABLED with patch
        wandb_log_unique_prompts=True,  # Also log to WandB tables
        # Optimization parameters
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",
        scale_rewards=True,
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_function],
    )
    
    # Train
    print(f"Starting training with completion logging enabled...")
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
                
            print(f"[{i}] {prompt} → {expected}")
            print(f"     Generated: {answer}")
            print(f"     Extracted: {extracted} {'✓' if is_correct else '✗'}")
    
    accuracy = 100 * correct / test_samples
    print(f"\nFinal Accuracy: {correct}/{test_samples} ({accuracy:.1f}%)")
    print(f"{'='*60}")
    
    wandb.log({"final_accuracy": accuracy})
    wandb.finish()

if __name__ == "__main__":
    main()