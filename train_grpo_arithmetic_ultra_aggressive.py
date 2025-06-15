#!/usr/bin/env python3
"""Ultra-aggressive GRPO arithmetic overfitting experiment."""

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import wandb
import os
from dotenv import load_dotenv
import random
import re

# Load environment variables
load_dotenv()

def create_ultra_simple_arithmetic_dataset(n_samples: int = 16):
    """Ultra simple addition problems: 0-3 + 0-3 only."""
    prompts = []
    seen = set()
    
    # Generate unique problems
    for a in range(4):  # 0-3
        for b in range(4):  # 0-3
            if len(prompts) >= n_samples:
                break
            problem = f"{a} + {b}"
            if problem not in seen:
                seen.add(problem)
                answer = a + b
                prompts.append({
                    "prompt": f"Calculate: {a} + {b} = ",
                    "expected": str(answer)
                })
    
    # Shuffle to avoid patterns
    random.shuffle(prompts)
    return prompts[:n_samples]

def main():
    # Configuration
    n_samples = 16
    batch_size = 16  # Must be <= n_samples
    num_generations = 16  # Must divide evenly into batch_size
    lr = 5e-5  # More reasonable learning rate
    temperature = 1.0  # High for diversity
    epochs = 100
    seed = 456
    
    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Initialize wandb
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"grpo_arithmetic_ultra_aggressive_b{batch_size}_g{num_generations}",
        config={
            "task": "arithmetic_ultra_simple",
            "n_samples": n_samples,
            "batch_size": batch_size,
            "num_generations": num_generations,
            "learning_rate": lr,
            "temperature": temperature,
            "epochs": epochs,
            "seed": seed,
            "model": "Qwen/Qwen2-0.5B-Instruct"
        },
        tags=["grpo-setup", "overfit", "arithmetic", "ultra-aggressive"]
    )
    
    # Create dataset
    data = create_ultra_simple_arithmetic_dataset(n_samples)
    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in data])
    expected_answers = {d["prompt"]: d["expected"] for d in data}
    
    # Print dataset
    print(f"\n{'='*60}")
    print(f"ULTRA AGGRESSIVE ARITHMETIC OVERFITTING")
    print(f"{'='*60}")
    print(f"Dataset size: {len(dataset)} unique problems")
    print(f"Batch size: {batch_size}")
    print(f"Generations per prompt: {num_generations}")
    print(f"Learning rate: {lr}")
    print(f"Temperature: {temperature}")
    print(f"\nAll problems:")
    for i, d in enumerate(data):
        print(f"  [{i}] {d['prompt']} → {d['expected']}")
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
            # Match first number
            match = re.search(r'^(\d+)', answer)
            if match:
                extracted = match.group(1)
            else:
                extracted = answer.split()[0] if answer else ""
            
            expected = expected_answers.get(prompt, "")
            
            # Binary reward
            if extracted == expected:
                rewards.append(2.0)  # Higher reward for correct
            else:
                rewards.append(-1.0)
                
        return rewards
    
    # GRPO config
    config = GRPOConfig(
        output_dir=f"./grpo_arithmetic_ultra_{seed}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=8,  # Short for simple answers
        max_prompt_length=64,
        beta=0.0,  # No KL penalty
        loss_type="dr_grpo",  # Bias-free
        epsilon=0.5,  # Higher clipping for aggressive updates
        num_iterations=3,  # Multiple policy updates per batch
        push_to_hub=False,
        report_to=["wandb"],
        logging_steps=1,
        save_steps=500,
        seed=seed,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        scale_rewards=False,  # Don't normalize rewards
        # Aggressive optimization
        warmup_ratio=0.0,  # No warmup
        weight_decay=0.0,  # No regularization
        max_grad_norm=10.0,  # Higher gradient clipping
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.99,  # Less momentum for faster adaptation
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_function],
    )
    
    # Train
    print(f"Starting ultra-aggressive training for {epochs} epochs...")
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
    with torch.no_grad():
        for i, d in enumerate(data):
            prompt = d['prompt']
            expected = d['expected']
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = completion[len(prompt):].strip()
            match = re.search(r'^(\d+)', answer)
            extracted = match.group(1) if match else answer.split()[0] if answer else ""
            
            is_correct = extracted == expected
            if is_correct:
                correct += 1
                
            print(f"[{i}] {prompt} → {expected}")
            print(f"     Generated: {answer}")
            print(f"     Extracted: {extracted} {'✓' if is_correct else '✗'}")
    
    accuracy = 100 * correct / len(data)
    print(f"\nFinal Accuracy: {correct}/{len(data)} ({accuracy:.1f}%)")
    print(f"{'='*60}")
    
    wandb.log({"final_accuracy": accuracy})
    wandb.finish()

if __name__ == "__main__":
    main()