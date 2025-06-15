#!/usr/bin/env python3
"""
Add standardized evaluation to all GRPO training scripts.
This ensures all experiments evaluate on morgan/arithmetic_eval dataset.
"""

import os
import re

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
        'arithmetic_eval': overall_accuracy,  # Primary metric
        'arithmetic_eval_correct': correct_total,
        'arithmetic_eval_total': len(eval_dataset),
        **difficulty_accs,
        **operation_accs
    }
'''

# The code to add after training completes
EVAL_CALL_TEMPLATE = '''
    # Evaluate on standardized dataset
    print("\\nEvaluating on standardized arithmetic dataset...")
    standard_eval_metrics = evaluate_on_standard_dataset({model_var}, {tokenizer_var}, {device_var})
    
    # Log standardized metrics
    if 'wandb' in globals() and wandb.run is not None:
        wandb.log(standard_eval_metrics)
        wandb.log({{
            "arithmetic_eval": standard_eval_metrics['arithmetic_eval']  # Primary metric
        }})
    
    print(f"\\n🎯 Standardized evaluation accuracy: {{standard_eval_metrics['arithmetic_eval']:.2%}}")
'''

def find_model_tokenizer_device_vars(content):
    """Find the variable names used for model, tokenizer, and device."""
    # Common patterns
    model_patterns = [
        r'(\w+)\s*=\s*trainer\.model',
        r'model\s*=\s*trainer\.model',
        r'(\w+)\s*=\s*AutoModelForCausalLM',
        r'model\s*=\s*AutoModelForCausalLM'
    ]
    
    tokenizer_patterns = [
        r'(\w+)\s*=\s*trainer\.tokenizer',
        r'tokenizer\s*=\s*trainer\.tokenizer',
        r'(\w+)\s*=\s*AutoTokenizer',
        r'tokenizer\s*=\s*AutoTokenizer'
    ]
    
    device_patterns = [
        r'(\w+)\s*=\s*torch\.device',
        r'device\s*=\s*torch\.device',
        r'(\w+)\s*=\s*model\.device',
        r'device\s*=\s*model\.device'
    ]
    
    model_var = 'trainer.model'
    tokenizer_var = 'trainer.tokenizer'
    device_var = 'model.device'
    
    # Try to find actual variable names
    for pattern in model_patterns:
        match = re.search(pattern, content)
        if match and match.group(1):
            model_var = match.group(1)
            break
    
    for pattern in tokenizer_patterns:
        match = re.search(pattern, content)
        if match and match.group(1):
            tokenizer_var = match.group(1)
            break
    
    # If we have explicit device assignment, use it
    for pattern in device_patterns:
        match = re.search(pattern, content)
        if match and match.group(1):
            device_var = match.group(1)
            break
    else:
        # Otherwise, use model.device
        device_var = f"{model_var}.device"
    
    return model_var, tokenizer_var, device_var

def update_script(script_path):
    """Update a single training script with standardized evaluation."""
    print(f"\nProcessing: {script_path}")
    
    # Skip certain files
    skip_patterns = ['template', 'standard_eval', 'with_standard_eval']
    if any(pattern in script_path for pattern in skip_patterns):
        print("  Skipping (template or already has standard eval)")
        return False
    
    # Read the script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check if it already has standardized evaluation
    if "evaluate_on_standard_dataset" in content or "morgan/arithmetic_eval" in content:
        print("  Already has standardized evaluation")
        return False
    
    # Check if it's a training script
    if "trainer.train()" not in content:
        print("  Not a training script, skipping")
        return False
    
    # Find variable names
    model_var, tokenizer_var, device_var = find_model_tokenizer_device_vars(content)
    print(f"  Found variables: model={model_var}, tokenizer={tokenizer_var}, device={device_var}")
    
    # Create the evaluation call with proper variable names
    eval_call = EVAL_CALL_TEMPLATE.format(
        model_var=model_var,
        tokenizer_var=tokenizer_var,
        device_var=device_var
    )
    
    # Find where to insert the evaluation function
    # Look for the last import statement
    import_matches = list(re.finditer(r'^(from .* import .*|import .*)$', content, re.MULTILINE))
    if import_matches:
        last_import_pos = import_matches[-1].end()
        # Insert the function after imports
        content = content[:last_import_pos] + "\n" + EVAL_FUNCTION + content[last_import_pos:]
    else:
        print("  WARNING: Could not find imports, adding at beginning")
        content = EVAL_FUNCTION + "\n" + content
    
    # Find where to insert the evaluation call
    # Look for after trainer.train() but before wandb.finish()
    train_match = re.search(r'trainer\.train\(\)', content)
    if not train_match:
        print("  ERROR: Could not find trainer.train()")
        return False
    
    # Find the next wandb.finish() or end of main function
    finish_match = re.search(r'wandb\.finish\(\)', content[train_match.end():])
    if finish_match:
        insert_pos = train_match.end() + finish_match.start()
    else:
        # Look for end of main function
        main_match = re.search(r'if __name__ == "__main__":\s*main\(\)', content)
        if main_match:
            # Find the end of main function
            # This is tricky - we'll look for the last line before if __name__
            lines = content[:main_match.start()].split('\n')
            # Find last non-empty line
            for i in range(len(lines)-1, -1, -1):
                if lines[i].strip():
                    insert_pos = len('\n'.join(lines[:i+1]))
                    break
        else:
            insert_pos = len(content)
    
    # Insert the evaluation call
    content = content[:insert_pos] + "\n" + eval_call + "\n" + content[insert_pos:]
    
    # Create backup
    backup_path = script_path + '.backup'
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # Write updated script
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Updated successfully! Backup saved to {backup_path}")
    return True

def main():
    import glob
    
    # Find all train_grpo*.py files
    training_scripts = glob.glob("train_grpo*.py")
    
    print(f"Found {len(training_scripts)} training scripts to check")
    
    updated_count = 0
    for script_path in training_scripts:
        if update_script(script_path):
            updated_count += 1
    
    print(f"\n{'='*60}")
    print(f"Updated {updated_count} scripts with standardized evaluation")
    print("All future experiments will now evaluate on morgan/arithmetic_eval dataset")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()