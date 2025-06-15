#!/usr/bin/env python3
"""
Evaluate trained GRPO models on the standardized arithmetic evaluation dataset.
This provides fair comparison across all experiments.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
import wandb
from typing import Dict, Any

def extract_answer(text: str, prompt: str, operation: str = None) -> str:
    """Extract the answer from model output"""
    completion = text[len(prompt):].strip()
    
    # For comparison tasks, look for yes/no
    if "greater than" in prompt.lower():
        completion_lower = completion.lower()
        if "yes" in completion_lower:
            return "yes"
        elif "no" in completion_lower:
            return "no"
    
    # For arithmetic, extract number
    match = re.match(r'^-?\d+', completion)
    if match:
        return match.group(0)
    
    # Fallback
    tokens = completion.split()
    if tokens:
        return tokens[0]
    
    return completion

def evaluate_model_on_standard_dataset(model_name_or_path: str, run_name: str = None) -> Dict[str, Any]:
    """Evaluate a model on the standardized arithmetic dataset"""
    
    print(f"\nEvaluating model: {model_name_or_path}")
    if run_name:
        print(f"Run name: {run_name}")
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Ensure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load standardized evaluation dataset
    print("\nLoading standardized evaluation dataset...")
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    print(f"Loaded {len(eval_dataset)} evaluation samples")
    
    # Initialize results tracking
    results = {
        'overall': {'correct': 0, 'total': 0},
        'by_difficulty': {},
        'by_operation': {}
    }
    
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
            predicted = extract_answer(response, prompt, operation)
            
            # Check correctness
            is_correct = predicted == expected
            
            # Update overall results
            results['overall']['total'] += 1
            if is_correct:
                results['overall']['correct'] += 1
            
            # By difficulty
            if difficulty not in results['by_difficulty']:
                results['by_difficulty'][difficulty] = {'correct': 0, 'total': 0}
            results['by_difficulty'][difficulty]['total'] += 1
            if is_correct:
                results['by_difficulty'][difficulty]['correct'] += 1
            
            # By operation
            if operation:
                if operation not in results['by_operation']:
                    results['by_operation'][operation] = {'correct': 0, 'total': 0}
                results['by_operation'][operation]['total'] += 1
                if is_correct:
                    results['by_operation'][operation]['correct'] += 1
    
    # Calculate and display results
    overall_acc = results['overall']['correct'] / results['overall']['total']
    print(f"\n{'='*60}")
    print(f"STANDARDIZED EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name_or_path}")
    if run_name:
        print(f"Run: {run_name}")
    print(f"\nOverall accuracy: {overall_acc:.2%} ({results['overall']['correct']}/{results['overall']['total']})")
    
    print("\nAccuracy by difficulty:")
    for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        if diff in results['by_difficulty']:
            stats = results['by_difficulty'][diff]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {diff}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nAccuracy by operation:")
    for op in ['+', '-', '*', '/']:
        if op in results['by_operation']:
            stats = results['by_operation'][op]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {op}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print(f"{'='*60}\n")
    
    # Return results
    return {
        'model': model_name_or_path,
        'run_name': run_name,
        'overall_accuracy': overall_acc,
        'results': results
    }

def main():
    """Evaluate key models from our experiments"""
    
    print("Standardized Arithmetic Evaluation")
    print("="*60)
    
    # List of models to evaluate
    # NOTE: In production, you would load these from WandB artifacts or saved checkpoints
    models_to_evaluate = [
        {
            'name': 'Qwen/Qwen2-0.5B-Instruct',
            'run_name': 'baseline',
            'description': 'Base model (no training)'
        },
        # Add trained model checkpoints here as they become available
        # Example:
        # {
        #     'name': './grpo-mixed-small-numbers/checkpoint-final',
        #     'run_name': 's8snd2a0',
        #     'description': 'Mixed Dataset + Small Numbers (60.7% on training data)'
        # },
    ]
    
    all_results = []
    
    for model_info in models_to_evaluate:
        try:
            result = evaluate_model_on_standard_dataset(
                model_info['name'],
                model_info.get('run_name')
            )
            result['description'] = model_info.get('description', '')
            all_results.append(result)
        except Exception as e:
            print(f"Error evaluating {model_info['name']}: {e}")
    
    # Summary comparison
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Model':<40} {'Accuracy':>10}")
        print("-"*60)
        for result in sorted(all_results, key=lambda x: x['overall_accuracy'], reverse=True):
            model_display = result['run_name'] or result['model']
            if len(model_display) > 40:
                model_display = model_display[:37] + "..."
            print(f"{model_display:<40} {result['overall_accuracy']:>10.2%}")

if __name__ == "__main__":
    main()