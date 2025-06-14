#!/usr/bin/env python3
"""
GRPO overfitting test with simple arithmetic
Using the same reward function approach as our main GRPO training
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import wandb
import os
from dotenv import load_dotenv
import re
from typing import List

# Load environment variables
load_dotenv()

# Initialize WandB
wandb.init(
    project=os.getenv("WANDB_PROJECT", "pippa"),
    entity=os.getenv("WANDB_ENTITY", "wild-ai"),
    name="gr00t-grpo-overfit-arithmetic",
    tags=["gr00t-overfit", "grpo", "arithmetic"],
    config={
        "model": "Qwen/Qwen2-0.5B-Instruct",
        "batch_size": 1,
        "num_generations": 4,
        "learning_rate": 5e-5,
        "num_epochs": 50,
        "task": "simple_arithmetic"
    }
)

# Model name
model_name = "Qwen/Qwen2-0.5B-Instruct"
print(f"Model: {model_name}")

# Create arithmetic examples for overfitting (need 4 for batch size)
train_data = [
    {"prompt": "Calculate: 2 + 3 = ", "expected": "5"},
    {"prompt": "Calculate: 2 + 3 = ", "expected": "5"},
    {"prompt": "Calculate: 2 + 3 = ", "expected": "5"},
    {"prompt": "Calculate: 2 + 3 = ", "expected": "5"},
]

# Create dataset
dataset = Dataset.from_list([{"prompt": item["prompt"]} for item in train_data])

# Reward function (same as in train_grpo_verifiable.py)
def arithmetic_reward_fn(samples: List[str], prompts: List[str], outputs: List[str]) -> List[float]:
    """Reward function for arithmetic problems."""
    rewards = []
    
    for prompt, output in zip(prompts, outputs):
        # Extract the expected answer from training data
        expected = None
        for item in train_data:
            if item["prompt"] in prompt:
                expected = item["expected"]
                break
        
        if expected is None:
            rewards.append(0.0)
            continue
            
        # Clean the output
        output_clean = output.strip()
        
        # Look for the answer in the output
        numbers = re.findall(r'-?\d+', output_clean)
        
        if numbers and numbers[0] == expected:
            # Correct answer
            reward = 2.0
            # Bonus for clean format
            if output_clean == expected or output_clean == f"{expected}.":
                reward = 3.0
        else:
            reward = 0.0
            
        rewards.append(reward)
    
    return rewards

# GRPO configuration for overfitting
config = GRPOConfig(
    output_dir="./grpo_overfit_arithmetic",
    num_train_epochs=50,
    per_device_train_batch_size=4,  # Must be divisible by num_generations
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    num_generations=4,
    temperature=0.7,
    max_completion_length=20,
    max_prompt_length=128,
    beta=0.0,  # No KL penalty for rule-based rewards
    logging_steps=1,
    save_strategy="no",
    report_to="wandb",
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model_name,
    args=config,
    train_dataset=dataset,
    reward_funcs=[arithmetic_reward_fn],
)

# Train
print("\nStarting GRPO overfitting training...")
print(f"Target: Learn to answer '2 + 3 = 5'")
print("-" * 50)

trainer.train()

# Test the model
print("\nTesting the model...")
# Load the trained model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(trainer.model, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

test_prompts = [
    "Calculate: 2 + 3 = ",  # Training example
    "Calculate: 3 + 2 = ",  # Commutative test
    "Calculate: 1 + 4 = ",  # Different numbers
]

final_reward = 0.0
for test_prompt in test_prompts:
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated[len(test_prompt):].strip()
    
    # Compute reward
    reward = arithmetic_reward_fn(
        samples=[generated],
        prompts=[test_prompt],
        outputs=[response]
    )[0]
    
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")
    print(f"Reward: {reward}")
    print()
    
    # Log test results
    if test_prompt == "Calculate: 2 + 3 = ":
        final_reward = reward
        
wandb.log({
    "train/overfit_success": 1 if final_reward >= 2.0 else -1,
    "test/final_reward": final_reward
})

wandb.finish()
print("Training complete!")