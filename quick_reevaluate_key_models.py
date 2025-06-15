#!/usr/bin/env python3
"""
Quick re-evaluation of key models on standardized dataset.
Evaluates only 50 samples for speed.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
import re
from dotenv import load_dotenv
import os
from typing import Dict, Any

load_dotenv()

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

def quick_evaluate(model, tokenizer, device, n_samples=50) -> Dict[str, float]:
    """Quick evaluation on subset of standardized dataset."""
    print(f"\nQuick evaluation on {n_samples} samples from morgan/arithmetic_eval...")
    
    # Load standardized evaluation dataset
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    print(f"Total dataset size: {len(eval_dataset)} samples")
    
    # Take subset
    eval_subset = eval_dataset.select(range(n_samples))
    
    correct_total = 0
    model.eval()
    
    with torch.no_grad():
        for i, sample in enumerate(eval_subset):
            prompt = sample['prompt']
            expected = sample['answer']
            
            # Generate model answer
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer
            predicted = extract_answer(response, prompt)
            
            # Check correctness
            is_correct = predicted == expected
            if is_correct:
                correct_total += 1
            
            if i < 5:  # Show first 5 examples
                print(f"  {prompt} → Model: {predicted}, Expected: {expected} {'✓' if is_correct else '✗'}")
    
    accuracy = correct_total / n_samples
    print(f"\nAccuracy: {accuracy:.2%} ({correct_total}/{n_samples})")
    
    return {
        'arithmetic_eval_quick': accuracy,
        'arithmetic_eval_samples': n_samples
    }

def main():
    print("Quick Re-evaluation of Key Models")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Key models to evaluate
    models_to_test = [
        ("Base Model", "Qwen/Qwen2-0.5B-Instruct", None),
        # Add trained models here if we can access their artifacts
    ]
    
    results = {}
    
    for name, model_path, run_id in models_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        
        try:
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Quick evaluate
            eval_results = quick_evaluate(model, tokenizer, device, n_samples=50)
            results[name] = eval_results['arithmetic_eval_quick']
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = None
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - Quick Evaluation on 50 Samples")
    print("="*60)
    for name, accuracy in results.items():
        if accuracy is not None:
            print(f"{name}: {accuracy:.2%}")
        else:
            print(f"{name}: Failed")
    
    print("\nNote: These are quick evaluations on 50 samples only.")
    print("Full evaluation uses all 200 samples from morgan/arithmetic_eval")

if __name__ == "__main__":
    main()