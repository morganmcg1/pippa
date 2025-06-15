#!/usr/bin/env python3
"""
Minimal GR00T-style training with proper WandB logging
This simulates GR00T training without requiring the full Isaac-GR00T setup
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
import wandb
import time

# Load environment variables
load_dotenv()

# Verify WandB credentials
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "wild-ai")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "pippa")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if not WANDB_API_KEY:
    print("ERROR: WANDB_API_KEY not found in .env file")
    exit(1)

print(f"=== GR00T Minimal Training ===")
print(f"WandB Entity: {WANDB_ENTITY}")
print(f"WandB Project: {WANDB_PROJECT}")
print(f"WandB API Key: {'SET' if WANDB_API_KEY else 'NOT SET'}")

# Initialize WandB with proper tags
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name="gr00t-minimal-training",
    tags=["gr00t-overfit", "gr00t-training", "minimal"],
    config={
        "model": "Qwen/Qwen2-0.5B-Instruct",
        "task": "robot_command_generation",
        "batch_size": 2,
        "learning_rate": 1e-4,
        "epochs": 20,
        "dataset_size": 4
    }
)

# Simple robot command dataset
class RobotCommandDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Simple robot commands in SO-101 style
        self.data = [
            {"input": "Pick up the red cube", "output": "MOVE_ARM(target='red_cube', action='pick')"},
            {"input": "Place the cube on table", "output": "MOVE_ARM(target='table', action='place')"},
            {"input": "Rotate to face door", "output": "ROTATE_BASE(direction='door', angle=90)"},
            {"input": "Walk forward 2 steps", "output": "WALK(direction='forward', steps=2)"},
        ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Format as instruction-following
        text = f"Human: {item['input']}\nRobot: {item['output']}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        encoding = {key: val.squeeze() for key, val in encoding.items()}
        encoding['labels'] = encoding['input_ids'].clone()
        return encoding

# Initialize model and tokenizer
model_name = "Qwen/Qwen2-0.5B-Instruct"
print(f"\nLoading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Create dataset and dataloader
dataset = RobotCommandDataset(tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
print("\nStarting training...")
print("=" * 60)

model.train()
global_step = 0

for epoch in range(20):
    epoch_loss = 0
    epoch_start = time.time()
    
    for batch in dataloader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        global_step += 1
        
        # Log every step
        wandb.log({
            "train/loss": loss.item(),
            "train/step": global_step,
            "train/learning_rate": optimizer.param_groups[0]['lr']
        }, step=global_step)
    
    # Epoch metrics
    avg_loss = epoch_loss / len(dataloader)
    epoch_time = time.time() - epoch_start
    
    # Test generation every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        test_results = []
        
        with torch.no_grad():
            for test_input in ["Pick up the blue box", "Walk backward 3 steps"]:
                prompt = f"Human: {test_input}\nRobot:"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated[len(prompt):].strip()
                
                test_results.append({
                    "input": test_input,
                    "output": response
                })
                
                print(f"Epoch {epoch} - Input: {test_input}")
                print(f"           Output: {response}")
        
        # Log test results to WandB
        test_table = wandb.Table(columns=["epoch", "input", "generated_output"])
        for result in test_results:
            test_table.add_data(epoch, result["input"], result["output"])
        
        wandb.log({
            "test/samples": test_table,
            "test/epoch": epoch
        }, step=global_step)
        
        model.train()
    
    # Log epoch metrics
    wandb.log({
        "train/epoch": epoch,
        "train/epoch_loss": avg_loss,
        "train/epoch_time": epoch_time,
        "train/samples_per_second": len(dataset) / epoch_time
    }, step=global_step)
    
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")

# Final evaluation
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

model.eval()
final_results = []

with torch.no_grad():
    # Test on training examples
    for item in dataset.data:
        prompt = f"Human: {item['input']}\nRobot:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()
        
        # Check if output matches expected format
        is_correct_format = "(" in response and ")" in response
        
        final_results.append({
            "input": item["input"],
            "expected": item["output"],
            "generated": response,
            "correct_format": is_correct_format
        })
        
        print(f"Input: {item['input']}")
        print(f"Expected: {item['output']}")
        print(f"Generated: {response}")
        print(f"Format OK: {is_correct_format}")
        print()

# Log final results
final_table = wandb.Table(columns=["input", "expected", "generated", "correct_format"])
correct_count = 0

for result in final_results:
    final_table.add_data(
        result["input"],
        result["expected"],
        result["generated"],
        result["correct_format"]
    )
    if result["correct_format"]:
        correct_count += 1

accuracy = correct_count / len(final_results)

wandb.log({
    "final/evaluation_table": final_table,
    "final/format_accuracy": accuracy,
    "train/overfit_success": 1 if accuracy >= 0.75 else -1
})

print(f"Final Format Accuracy: {correct_count}/{len(final_results)} ({accuracy:.1%})")

wandb.finish()
print("\nTraining complete!")