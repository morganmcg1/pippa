# Canonical GRPO Training Script

This is the reference implementation for GRPO training on arithmetic and related tasks, incorporating all learnings from our experiments.

## Quick Start

```bash
# Basic arithmetic training (default settings)
python train_grpo_canonical.py

# Small numbers (0-10) with custom settings
python train_grpo_canonical.py --dataset_type small_numbers --batch_size 256 --num_generations 16

# Mixed dataset (arithmetic + counting + comparison)
python train_grpo_canonical.py --dataset_type mixed --n_samples 150 --temperature 0.8

# High temperature experiment
python train_grpo_canonical.py --temperature 1.0 --exp_name high_temp_experiment
```

## Key Features

1. **Proper batch_indices handling** - Correctly handles how GRPO passes batch indices to reward functions
2. **Fixed completion logging** - Custom trainer class that handles TRL's logging issues
3. **Multiple dataset types** - Arithmetic, mixed tasks, or small numbers
4. **Configurable via command line** - All hyperparameters accessible via argparse
5. **Comprehensive evaluation** - Breaks down performance by operation/task type

## Critical Parameters

### Must-Have Settings (from our experiments)
- **`--beta 0.1`** - KL penalty is CRITICAL for arithmetic tasks (default)
- **`--num_generations 16`** - More generations = better reward diversity (default)
- **`--batch_size`** - Must be divisible by `num_generations`

### Recommended Configurations

#### For breaking the 87.5% barrier:
```bash
python train_grpo_canonical.py \
    --batch_size 256 \
    --num_generations 16 \
    --learning_rate 5e-6 \
    --temperature 0.7 \
    --epochs 100 \
    --beta 0.1
```

#### For quick overfitting tests:
```bash
python train_grpo_canonical.py \
    --dataset_type small_numbers \
    --n_samples 50 \
    --batch_size 64 \
    --num_generations 16 \
    --learning_rate 1e-5 \
    --epochs 30
```

## Command Line Arguments

### Dataset Options
- `--dataset_type`: Choose from `arithmetic`, `mixed`, or `small_numbers`
- `--n_samples`: Number of samples in dataset (default: 100)
- `--min_num`: Minimum number for arithmetic (default: 0)
- `--max_num`: Maximum number for arithmetic (default: 20)

### Training Options
- `--batch_size`: Training batch size (default: 256)
- `--num_generations`: Completions per prompt (default: 16)
- `--learning_rate`: Learning rate (default: 5e-6)
- `--temperature`: Generation temperature (default: 0.7)
- `--epochs`: Number of epochs (default: 50)
- `--beta`: KL penalty coefficient (default: 0.1)

### Other Options
- `--model_name`: Model to train (default: "Qwen/Qwen2-0.5B-Instruct")
- `--seed`: Random seed (default: 42)
- `--exp_name`: WandB experiment name
- `--tags`: WandB tags (list)

## Important Notes

1. **Batch Size Rule**: The script will error if `batch_size % num_generations != 0`

2. **Generation Diversity**: Never reduce `num_generations` below 8-16. GRPO needs reward variance:
   ```
   advantage = (reward - mean) / std
   ```
   With too few generations, std → 0 and training fails.

3. **KL Penalty**: Always use `beta > 0` for arithmetic tasks. Without it, models collapse to outputting garbage.

4. **Dataset Diversity**: The script automatically ensures unique problems to maximize learning signal.

## Monitoring Training

Key metrics to watch in WandB:
- `train/reward`: Should increase steadily
- `train/frac_reward_zero_std`: Should stay < 0.5 (lower is better)
- `train/kl`: Should stay in 0.5-1.0 range
- `final_accuracy`: Ultimate performance metric

## Common Issues

1. **"IndexError: Invalid key"** - Fixed in this script with proper batch_indices handling
2. **Zero gradients** - Usually means all rewards are identical (increase temperature or generations)
3. **Reward stuck at -1.0** - Model not learning; check beta > 0 and temperature high enough

## Example Experiments

### Reproduce our best results:
```bash
# 87.5% reward baseline
python train_grpo_canonical.py \
    --n_samples 100 \
    --batch_size 64 \
    --num_generations 16 \
    --epochs 100 \
    --exp_name baseline_87_percent

# Breaking 87.5% with more data
python train_grpo_canonical.py \
    --n_samples 500 \
    --batch_size 256 \
    --num_generations 16 \
    --epochs 100 \
    --exp_name expanded_dataset
```

### Test new ideas:
```bash
# Higher temperature for diversity
python train_grpo_canonical.py \
    --temperature 1.0 \
    --batch_size 256 \
    --num_generations 16 \
    --exp_name high_temp_diversity

# Mixed tasks for bootstrapping
python train_grpo_canonical.py \
    --dataset_type mixed \
    --n_samples 300 \
    --batch_size 240 \
    --num_generations 16 \
    --exp_name mixed_bootstrap
```

## Integration with Research Journal

When running experiments, always log:
1. Experiment name
2. WandB run ID (shown at start)
3. Timestamp (included in WandB config)
4. Hypothesis being tested
5. Results and learnings

Example:
```markdown
### Experiment Name - YYYY-MM-DD_HH:MM - Run ID: xxxxxxxx
**Hypothesis**: Testing if X improves Y
**Command**: `python train_grpo_canonical.py --args`
**Results**: Achieved Z% accuracy, learned that...
```