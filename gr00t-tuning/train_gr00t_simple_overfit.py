#!/usr/bin/env python3
"""
Simple GR00T overfitting test with easier reward function
Designed to actually achieve overfitting with reward > 1.0
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from dotenv import load_dotenv
import wandb

# Load environment variables
load_dotenv()

# Initialize WandB
wandb.init(
    project=os.getenv("WANDB_PROJECT", "pippa"),
    entity=os.getenv("WANDB_ENTITY", "wild-ai"),
    name="gr00t-simple-overfit",
    tags=["gr00t-overfit", "simple-test"],
    config={
        "model": "Qwen/Qwen2-0.5B-Instruct",
        "batch_size": 1,
        "learning_rate": 1e-3,  # Higher LR for faster overfitting
        "num_epochs": 100,
        "task": "simple_command_generation"
    }
)

# Initialize model and tokenizer
model_name = "Qwen/Qwen2-0.5B-Instruct"
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Single training example for overfitting
train_prompt = "Generate robot command:"
train_response = "MOVE"

# Create full training text
full_text = f"{train_prompt} {train_response}"
inputs = tokenizer(full_text, return_tensors="pt").to(device)

# Simple reward function
def compute_reward(generated_text: str) -> float:
    """Simple reward: 2.0 if output contains 'MOVE', 0.0 otherwise"""
    if "MOVE" in generated_text:
        return 2.0
    return 0.0

# Optimizer with high learning rate
optimizer = AdamW(model.parameters(), lr=1e-3)

# Training loop
model.train()
print("\nStarting overfitting training...")
print(f"Target: Generate 'MOVE' when prompted with '{train_prompt}'")
print("-" * 50)

for epoch in range(100):
    # Forward pass
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Generate and check reward every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            prompt_inputs = tokenizer(train_prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **prompt_inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False
            )
            generated_text = tokenizer.decode(generated[0][prompt_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            reward = compute_reward(generated_text)
        model.train()
        
        # Log metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/epoch": epoch,
            "rewards/value": reward,
            "train/learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch)
        
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Generated: '{generated_text.strip()}', Reward = {reward}")
        
        # Early stopping if we achieve perfect reward
        if reward >= 2.0:
            print(f"\nSuccess! Achieved reward of {reward} at epoch {epoch}")
            break
    else:
        # Log just loss for other epochs
        wandb.log({
            "train/loss": loss.item(),
            "train/epoch": epoch,
        }, step=epoch)

# Final test
print("\nFinal test:")
model.eval()
with torch.no_grad():
    test_prompts = [
        "Generate robot command:",
        "Robot command:",
        "Command:"
    ]
    
    for test_prompt in test_prompts:
        test_inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **test_inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False
        )
        generated = tokenizer.decode(outputs[0][test_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        reward = compute_reward(generated)
        
        print(f"Prompt: '{test_prompt}' -> Generated: '{generated.strip()}' (Reward: {reward})")
        
        wandb.log({
            "test/reward": reward,
            "test/example": wandb.Table(
                columns=["prompt", "generated", "reward"],
                data=[[test_prompt, generated.strip(), reward]]
            )
        })

wandb.log({
    "train/final_loss": loss.item(),
    "train/final_epoch": epoch,
    "train/overfit_success": 1 if reward >= 2.0 else -1
})

wandb.finish()
print("\nTraining complete!")