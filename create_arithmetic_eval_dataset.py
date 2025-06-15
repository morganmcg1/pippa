#!/usr/bin/env python3
"""
Create a standardized arithmetic evaluation dataset for fair comparison across GRPO experiments.
This dataset will have 200 samples with varying difficulty levels and will be uploaded to HuggingFace.
"""

import random
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import json

def create_arithmetic_eval_dataset():
    """Create 200 arithmetic problems with varying difficulty"""
    prompts = []
    answers = []
    difficulty_levels = []
    metadata = []
    
    # Define difficulty tiers
    # Tier 1: Very Easy (0-5, addition only) - 30 samples
    for i in range(30):
        a = random.randint(0, 5)
        b = random.randint(0, 5)
        result = a + b
        
        prompt = f"Calculate: {a} + {b} = "
        prompts.append(prompt)
        answers.append(str(result))
        difficulty_levels.append("very_easy")
        metadata.append({
            "a": a,
            "b": b,
            "operation": "+",
            "result": result,
            "difficulty": "very_easy",
            "number_range": "0-5"
        })
    
    # Tier 2: Easy (0-10, all operations) - 40 samples
    operations = ['+', '-', '*']
    for i in range(40):
        a = random.randint(0, 10)
        b = random.randint(0, 10)
        op = random.choice(operations)
        
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:  # *
            result = a * b
        
        prompt = f"Calculate: {a} {op} {b} = "
        prompts.append(prompt)
        answers.append(str(result))
        difficulty_levels.append("easy")
        metadata.append({
            "a": a,
            "b": b,
            "operation": op,
            "result": result,
            "difficulty": "easy",
            "number_range": "0-10"
        })
    
    # Tier 3: Medium (0-20, all operations) - 50 samples
    for i in range(50):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        op = random.choice(operations)
        
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:  # *
            result = a * b
        
        prompt = f"Calculate: {a} {op} {b} = "
        prompts.append(prompt)
        answers.append(str(result))
        difficulty_levels.append("medium")
        metadata.append({
            "a": a,
            "b": b,
            "operation": op,
            "result": result,
            "difficulty": "medium",
            "number_range": "0-20"
        })
    
    # Tier 4: Hard (10-50, all operations) - 40 samples
    for i in range(40):
        a = random.randint(10, 50)
        b = random.randint(10, 50)
        op = random.choice(operations)
        
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:  # *
            result = a * b
        
        prompt = f"Calculate: {a} {op} {b} = "
        prompts.append(prompt)
        answers.append(str(result))
        difficulty_levels.append("hard")
        metadata.append({
            "a": a,
            "b": b,
            "operation": op,
            "result": result,
            "difficulty": "hard",
            "number_range": "10-50"
        })
    
    # Tier 5: Very Hard (20-100, all operations including division) - 40 samples
    operations_hard = ['+', '-', '*', '/']
    for i in range(40):
        a = random.randint(20, 100)
        b = random.randint(20, 100)
        op = random.choice(operations_hard)
        
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        elif op == '*':
            result = a * b
        else:  # /
            # Ensure clean division
            if b == 0:
                b = 1
            # Make it a clean division problem
            result = a // b
            a = result * b  # Adjust a to ensure clean division
        
        prompt = f"Calculate: {a} {op} {b} = "
        prompts.append(prompt)
        answers.append(str(result))
        difficulty_levels.append("very_hard")
        metadata.append({
            "a": a,
            "b": b,
            "operation": op,
            "result": result,
            "difficulty": "very_hard",
            "number_range": "20-100"
        })
    
    # Create the dataset
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "difficulty": difficulty_levels,
        "metadata": metadata
    })
    
    # Shuffle the dataset to mix difficulty levels
    dataset = dataset.shuffle(seed=42)
    
    return dataset

def main():
    print("Creating arithmetic evaluation dataset...")
    
    # Create the dataset
    eval_dataset = create_arithmetic_eval_dataset()
    
    print(f"\nDataset created with {len(eval_dataset)} samples")
    
    # Show distribution of difficulties
    difficulty_counts = {}
    for difficulty in eval_dataset['difficulty']:
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    print("\nDifficulty distribution:")
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"  {difficulty}: {count} samples ({count/200*100:.0f}%)")
    
    # Show some examples
    print("\nExample problems:")
    for i in range(5):
        sample = eval_dataset[i]
        print(f"  [{sample['difficulty']}] {sample['prompt']} → {sample['answer']}")
    
    # Create dataset dict with train split (even though it's for evaluation)
    dataset_dict = DatasetDict({
        "test": eval_dataset
    })
    
    # Upload to HuggingFace Hub
    print("\nUploading to HuggingFace Hub...")
    try:
        # Push to hub
        dataset_dict.push_to_hub(
            "morganmcg1/arithmetic_eval",
            private=False,
            commit_message="Standardized arithmetic evaluation dataset for GRPO experiments"
        )
        print("✅ Successfully uploaded to: https://huggingface.co/datasets/morganmcg1/arithmetic_eval")
        
        # Create a comprehensive dataset card
        dataset_card = """
---
language:
- en
task_categories:
- text-generation
tags:
- arithmetic
- evaluation
- grpo
size_categories:
- n<1K
---

# Arithmetic Evaluation Dataset

This dataset contains 200 arithmetic problems for evaluating language models trained with GRPO (Group Relative Policy Optimization) or other methods.

## Dataset Description

A standardized evaluation set for arithmetic tasks with varying difficulty levels, designed to fairly compare different training approaches.

### Difficulty Levels

- **very_easy** (30 samples, 15%): Addition only, numbers 0-5
- **easy** (40 samples, 20%): All operations (+, -, *), numbers 0-10  
- **medium** (50 samples, 25%): All operations (+, -, *), numbers 0-20
- **hard** (40 samples, 20%): All operations (+, -, *), numbers 10-50
- **very_hard** (40 samples, 20%): All operations (+, -, *, /), numbers 20-100

### Dataset Fields

- `prompt`: The arithmetic problem in format "Calculate: a op b = "
- `answer`: The correct answer as a string
- `difficulty`: The difficulty level (very_easy, easy, medium, hard, very_hard)
- `metadata`: Additional information including operands, operation type, and numeric result

### Usage

```python
from datasets import load_dataset

# Load the evaluation dataset
eval_dataset = load_dataset("morganmcg1/arithmetic_eval", split="test")

# Example: evaluate a model
correct = 0
for sample in eval_dataset:
    model_output = your_model.generate(sample["prompt"])
    extracted_answer = extract_number(model_output)
    if extracted_answer == sample["answer"]:
        correct += 1

accuracy = correct / len(eval_dataset)
print(f"Overall accuracy: {accuracy:.2%}")

# Analyze by difficulty
for difficulty in ["very_easy", "easy", "medium", "hard", "very_hard"]:
    subset = [s for s in eval_dataset if s["difficulty"] == difficulty]
    # ... evaluate on subset
```

### Purpose

This dataset was created to provide a fair comparison benchmark for GRPO arithmetic experiments. Previous experiments evaluated on their own training data with different difficulty ranges, making comparisons invalid. This standardized test set ensures all models are evaluated on the same problems.

### Citation

If you use this dataset, please cite:

```
@dataset{arithmetic_eval_2024,
  author = {Morgan McGuire},
  title = {Arithmetic Evaluation Dataset for GRPO Experiments},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/morganmcg1/arithmetic_eval}
}
```
"""
        
        # Save dataset card locally
        with open("arithmetic_eval_dataset_card.md", "w") as f:
            f.write(dataset_card)
        
        print("\nDataset card saved locally as arithmetic_eval_dataset_card.md")
        
    except Exception as e:
        print(f"❌ Error uploading to HuggingFace Hub: {e}")
        print("\nMake sure you are logged in to HuggingFace:")
        print("  huggingface-cli login")
        print("\nOr set HF_TOKEN environment variable")
        
        # Save locally as backup
        eval_dataset.save_to_disk("arithmetic_eval_dataset")
        print("\nDataset saved locally to ./arithmetic_eval_dataset/")

if __name__ == "__main__":
    main()