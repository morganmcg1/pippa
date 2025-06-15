#!/usr/bin/env python3
"""
Quick evaluation of base model on a subset of standardized dataset.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re

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

def main():
    print("Quick Base Model Evaluation (20 samples)")
    print("="*60)
    
    # Model configuration
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("\nLoading evaluation dataset...")
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    
    # Evaluate first 20 samples
    correct = 0
    model.eval()
    with torch.no_grad():
        for i in range(min(20, len(eval_dataset))):
            sample = eval_dataset[i]
            prompt = sample['prompt']
            expected = sample['answer']
            difficulty = sample['difficulty']
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted = extract_answer(response, prompt)
            
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            
            print(f"[{difficulty}] {prompt} Expected: {expected}, Got: {predicted} {'✓' if is_correct else '✗'}")
    
    accuracy = correct / 20
    print(f"\nAccuracy on 20 samples: {accuracy:.2%} ({correct}/20)")
    print("\nThis gives us a rough baseline for comparison.")

if __name__ == "__main__":
    main()