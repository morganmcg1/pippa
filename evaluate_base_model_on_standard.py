#!/usr/bin/env python3
"""
Quick evaluation of the base Qwen model on standardized arithmetic dataset.
This establishes a baseline for comparison.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import wandb
from dotenv import load_dotenv
import os

load_dotenv()

def extract_answer(text: str, prompt: str) -> str:
    """Extract the answer from model output"""
    completion = text[len(prompt):].strip()
    
    # For arithmetic, extract number
    match = re.match(r'^-?\d+', completion)
    if match:
        return match.group(0)
    
    # Fallback
    tokens = completion.split()
    if tokens:
        return tokens[0]
    
    return completion

def main():
    print("Evaluating Base Qwen Model on Standardized Arithmetic Dataset")
    print("="*60)
    
    # Model configuration
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="qwen_base_standard_eval",
        tags=["evaluation", "baseline", "standardized-eval"],
        config={
            "model": model_name,
            "eval_dataset": "morgan/arithmetic_eval",
            "eval_dataset_size": 200
        }
    )
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load standardized evaluation dataset
    print("\nLoading standardized evaluation dataset...")
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    print(f"Loaded {len(eval_dataset)} evaluation samples")
    
    # Initialize results tracking
    results_by_difficulty = {}
    results_by_operation = {}
    correct_total = 0
    
    # Evaluate
    print("\nEvaluating...")
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(eval_dataset):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(eval_dataset)}")
            
            prompt = sample['prompt']
            expected = sample['answer']
            difficulty = sample['difficulty']
            operation = sample['metadata']['operation'] if sample['metadata'] else None
            
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
            
            # Extract predicted answer
            predicted = extract_answer(response, prompt)
            
            # Check correctness
            is_correct = predicted == expected
            
            if is_correct:
                correct_total += 1
            
            # Track by difficulty
            if difficulty not in results_by_difficulty:
                results_by_difficulty[difficulty] = {'correct': 0, 'total': 0}
            results_by_difficulty[difficulty]['total'] += 1
            if is_correct:
                results_by_difficulty[difficulty]['correct'] += 1
            
            # Track by operation
            if operation:
                if operation not in results_by_operation:
                    results_by_operation[operation] = {'correct': 0, 'total': 0}
                results_by_operation[operation]['total'] += 1
                if is_correct:
                    results_by_operation[operation]['correct'] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct_total / len(eval_dataset)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"BASE MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"\nOverall accuracy: {overall_accuracy:.2%} ({correct_total}/{len(eval_dataset)})")
    
    print("\nAccuracy by difficulty:")
    difficulty_metrics = {}
    for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        if diff in results_by_difficulty:
            stats = results_by_difficulty[diff]
            acc = stats['correct'] / stats['total']
            difficulty_metrics[f"eval_{diff}_accuracy"] = acc
            print(f"  {diff}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nAccuracy by operation:")
    operation_metrics = {}
    for op in ['+', '-', '*', '/']:
        if op in results_by_operation:
            stats = results_by_operation[op]
            acc = stats['correct'] / stats['total']
            op_name = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op]
            operation_metrics[f"eval_{op_name}_accuracy"] = acc
            print(f"  {op}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print(f"{'='*60}\n")
    
    # Log to WandB
    wandb.log({
        'standardized_eval_accuracy': overall_accuracy,
        'standardized_eval_correct': correct_total,
        'standardized_eval_total': len(eval_dataset),
        **difficulty_metrics,
        **operation_metrics
    })
    
    wandb.finish()
    
    print(f"Base model achieves {overall_accuracy:.2%} on standardized evaluation.")
    print("This is the baseline to beat with GRPO training.")

if __name__ == "__main__":
    main()