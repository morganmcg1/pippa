#!/usr/bin/env python3
"""
Test script to load and examine the arithmetic evaluation dataset.
Shows how to use it for model evaluation.
"""

from datasets import Dataset, load_from_disk
import json

def main():
    print("Loading arithmetic evaluation dataset...")
    
    # Load the dataset from disk
    dataset = load_from_disk("arithmetic_eval_dataset")
    
    print(f"\nDataset loaded with {len(dataset)} samples")
    
    # Show distribution
    difficulty_counts = {}
    operation_counts = {}
    
    for i in range(len(dataset)):
        sample = dataset[i]
        difficulty = sample['difficulty']
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        # Extract operation from metadata
        if sample['metadata'] and 'operation' in sample['metadata']:
            op = sample['metadata']['operation']
            operation_counts[op] = operation_counts.get(op, 0) + 1
    
    print("\nDifficulty distribution:")
    for difficulty in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        count = difficulty_counts.get(difficulty, 0)
        print(f"  {difficulty}: {count} samples ({count/200*100:.0f}%)")
    
    print("\nOperation distribution:")
    for op, count in sorted(operation_counts.items()):
        print(f"  {op}: {count} samples ({count/200*100:.0f}%)")
    
    # Show examples from each difficulty level
    print("\nExample problems by difficulty:")
    for difficulty in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        # Find first example of this difficulty
        for sample in dataset:
            if sample['difficulty'] == difficulty:
                print(f"  [{difficulty}] {sample['prompt']} → {sample['answer']}")
                break
    
    # Show some statistics
    print("\nAnswer value statistics:")
    answers = [int(sample['answer']) for sample in dataset]
    print(f"  Min answer: {min(answers)}")
    print(f"  Max answer: {max(answers)}")
    print(f"  Median answer: {sorted(answers)[len(answers)//2]}")
    
    # Example evaluation function
    print("\n" + "="*60)
    print("Example: How to use this dataset for evaluation")
    print("="*60)
    
    print("""
def evaluate_model_on_arithmetic(model, tokenizer, device):
    dataset = load_from_disk("arithmetic_eval_dataset")
    
    results = {
        'overall': {'correct': 0, 'total': 0},
        'by_difficulty': {},
        'by_operation': {}
    }
    
    for sample in dataset:
        prompt = sample['prompt']
        expected = sample['answer']
        difficulty = sample['difficulty']
        operation = sample['metadata']['operation']
        
        # Generate model answer
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=16, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (simple regex approach)
        import re
        completion = response[len(prompt):].strip()
        match = re.match(r'^-?\\d+', completion)
        predicted = match.group(0) if match else completion.split()[0] if completion else ""
        
        # Check correctness
        is_correct = predicted == expected
        
        # Update results
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
        if operation not in results['by_operation']:
            results['by_operation'][operation] = {'correct': 0, 'total': 0}
        results['by_operation'][operation]['total'] += 1
        if is_correct:
            results['by_operation'][operation]['correct'] += 1
    
    # Calculate accuracies
    overall_acc = results['overall']['correct'] / results['overall']['total']
    print(f"Overall accuracy: {overall_acc:.2%}")
    
    print("\\nAccuracy by difficulty:")
    for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        if diff in results['by_difficulty']:
            stats = results['by_difficulty'][diff]
            acc = stats['correct'] / stats['total']
            print(f"  {diff}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\\nAccuracy by operation:")
    for op in ['+', '-', '*', '/']:
        if op in results['by_operation']:
            stats = results['by_operation'][op]
            acc = stats['correct'] / stats['total']
            print(f"  {op}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    return overall_acc
""")
    
    print("\nDataset is ready for use!")
    print("To use in your experiments, copy 'arithmetic_eval_dataset' folder to your experiment directory.")

if __name__ == "__main__":
    main()