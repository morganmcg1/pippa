#!/usr/bin/env python3
"""
Simple overfitting test without WandB
Tests if we can overfit on a single batch
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

# Initialize model and tokenizer
model_name = "Qwen/Qwen2-0.5B-Instruct"
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
model = model.to(device)

# Create a simple robot command dataset (1 sample for overfitting)
train_data = [
    {
        "input": "Move the robot arm to position (10, 20, 30)",
        "output": "MOVE_ARM(x=10, y=20, z=30)"
    }
]

# Prepare the training example
example = train_data[0]
input_text = f"Human: {example['input']}\nAssistant: {example['output']}"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
print("\nStarting overfitting test on 1 batch...")
print(f"Batch size: 1")
print(f"Learning rate: 1e-4")
print(f"Max epochs: 100")
print("-" * 50)

model.train()

for epoch in range(100):
    # Forward pass
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Check if we've overfit (loss < 0.01)
    if loss.item() < 0.01:
        print(f"\nSuccessfully overfit! Final loss: {loss.item():.6f}")
        break

# Test the model
print("\nTesting the model...")
model.eval()
test_input = "Human: Move the robot arm to position (10, 20, 30)\nAssistant:"
test_inputs = tokenizer(test_input, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **test_inputs,
        max_new_tokens=50,
        temperature=0.1,
        do_sample=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")

print(f"\nFinal loss: {loss.item():.6f}")
print(f"Final epoch: {epoch}")
print(f"Overfit success: {loss.item() < 0.01}")
print("\nTest complete!")