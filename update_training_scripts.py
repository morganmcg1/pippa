#!/usr/bin/env python3
"""
Script to update all GRPO training scripts to use standardized evaluation.
This adds the evaluate_on_standard_dataset function to existing scripts.
"""

import os
import glob

# The evaluation function to add
EVAL_FUNCTION = '''
def evaluate_on_standard_dataset(model, tokenizer, device) -> Dict[str, float]:
    """
    Evaluate model on the standardized arithmetic evaluation dataset.
    This ensures fair comparison across all experiments.
    """
    print("\\n" + "="*60)
    print("STANDARDIZED EVALUATION (morgan/arithmetic_eval)")
    print("="*60)
    
    # Load standardized evaluation dataset
    from datasets import load_dataset
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    print(f"Loaded {len(eval_dataset)} evaluation samples")
    
    # Initialize results tracking
    results_by_difficulty = {}
    results_by_operation = {}
    correct_total = 0
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(eval_dataset):
            if i % 50 == 0 and i > 0:
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
            
            # Extract answer (reuse existing extract_answer function if available)
            completion = response[len(prompt):].strip()
            import re
            match = re.match(r'^-?\\d+', completion)
            predicted = match.group(0) if match else completion.split()[0] if completion else ""
            
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
    print(f"\\nOverall accuracy: {overall_accuracy:.2%} ({correct_total}/{len(eval_dataset)})")
    
    print("\\nAccuracy by difficulty:")
    difficulty_accs = {}
    for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        if diff in results_by_difficulty:
            stats = results_by_difficulty[diff]
            acc = stats['correct'] / stats['total']
            difficulty_accs[f"eval_{diff}_accuracy"] = acc
            print(f"  {diff}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\\nAccuracy by operation:")
    operation_accs = {}
    for op in ['+', '-', '*', '/']:
        if op in results_by_operation:
            stats = results_by_operation[op]
            acc = stats['correct'] / stats['total']
            op_name = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op]
            operation_accs[f"eval_{op_name}_accuracy"] = acc
            print(f"  {op}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("="*60)
    
    # Return all metrics for WandB logging
    return {
        'standardized_eval_accuracy': overall_accuracy,
        'standardized_eval_correct': correct_total,
        'standardized_eval_total': len(eval_dataset),
        **difficulty_accs,
        **operation_accs
    }
'''

# The code to add after training completes
EVAL_CALL = '''
    # Evaluate on standardized dataset
    print("\\nEvaluating on standardized arithmetic dataset...")
    standard_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
    
    # Log standardized metrics
    wandb.log(standard_eval_metrics)
    wandb.log({
        "final_accuracy": standard_eval_metrics['standardized_eval_accuracy']  # Override with standardized accuracy
    })
    
    print(f"\\n🎯 Standardized evaluation accuracy: {standard_eval_metrics['standardized_eval_accuracy']:.2%}")
'''

def main():
    # Find all train_grpo*.py files
    training_scripts = glob.glob("train_grpo*.py")
    
    print(f"Found {len(training_scripts)} training scripts to update")
    
    for script_path in training_scripts:
        print(f"\nProcessing: {script_path}")
        
        # Skip already processed files
        if "standard_eval" in script_path or "template" in script_path:
            print("  Skipping (already has standard eval or is template)")
            continue
        
        # Read the script
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check if it already has standardized evaluation
        if "evaluate_on_standard_dataset" in content:
            print("  Already has standardized evaluation")
            continue
        
        # Check if it's a training script with evaluation
        if "trainer.train()" not in content:
            print("  Not a training script, skipping")
            continue
        
        print("  ✓ Would update this script")
        # In practice, we would insert the evaluation function and call here
        # But for now, let's just identify which scripts need updating
    
    print("\nTo update scripts, modify this script to actually insert the evaluation code.")

if __name__ == "__main__":
    main()