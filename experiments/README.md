# GRPO Training Experiments

This directory contains various GRPO training experiments and scripts.

## Core Training Scripts

### ✅ Production Ready
- `../train_grpo.py` - Main GRPO implementation with Dr GRPO support
- `../train_grpo_wandb.py` - Simple wrapper that enables WandB tracking
- `../train_grpo_verifiable.py` - GRPO with verifiable rewards (arithmetic, counting, etc.)

### 🧪 Experimental / Failed Approaches
These scripts were used to debug GRPO setup issues. They demonstrate what doesn't work:

- `train_grpo_overfit.py` - Initial overfitting attempts with GSM8K
- `train_grpo_progressive_overfit.py` - Echo/pattern tasks (failed - no verifiable rewards)
- `train_grpo_high_gpu_overfit.py` - High GPU utilization attempts
- `train_grpo_debug_overfit.py` - Debug version with extensive logging
- `train_grpo_overfit_max_gpu.py` - Maximum GPU utilization experiments

## Key Learnings

1. **GRPO requires verifiable rewards** - Tasks like "Say X" don't work because there's no objective correctness
2. **Model must have latent understanding** - GRPO elicits existing capabilities, doesn't teach new skills
3. **Use appropriate loss type** - `dr_grpo` for bias-free training, `beta=0.0` for rule-based rewards

## Running Experiments

### Verifiable Rewards (Recommended)
```bash
python train_grpo_verifiable.py --task arithmetic --batch_size 64 --num_generations 16
```

### Original Math Training
```bash
python train_grpo_wandb.py
```

## Failed Approaches to Avoid
- Echo tasks without clear correctness criteria
- Pattern completion without verifiable answers
- Any task where multiple outputs could be "correct"