
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
eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")

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
  url = {https://huggingface.co/datasets/morgan/arithmetic_eval}
}
```
