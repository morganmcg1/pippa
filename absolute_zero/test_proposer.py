#!/usr/bin/env python3
"""Test the proposer's problem generation capability."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

def test_proposer_generation():
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    # Test different prompts
    test_prompts = [
        "Generate a simple arithmetic problem in the format 'Calculate: X + Y = ': ",
        "Calculate: ",
        "5 + 3 = ",
        "Create an arithmetic problem: ",
        "Write a math problem: Calculate: ",
    ]
    
    print("\nTesting proposer generation with different prompts:\n")
    
    for prompt in test_prompts:
        print(f"Prompt: '{prompt}'")
        
        # Generate multiple samples
        for i in range(3):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated[len(prompt):].strip()
            
            print(f"  Sample {i+1}: '{completion}'")
        
        print()

if __name__ == "__main__":
    test_proposer_generation()