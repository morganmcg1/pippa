# GRPO Archive

This directory contains historical GRPO training scripts from our experimental journey. These scripts were instrumental in discovering key insights but have been archived to keep the main directory focused on actively used scripts.

## Archived Scripts

### Arithmetic Experiments
- **train_grpo_arithmetic_kl_penalty.py** - The script that led to discovering KL penalty is required for arithmetic tasks
- **train_grpo_arithmetic_with_tables.py** - Experiment with WandB tables logging (led to log_completions compatibility issue)
- **train_grpo_arithmetic_ultra_aggressive.py** - Failed attempt with very high learning rates
- **train_grpo_arithmetic_debug_completions.py** - Debug attempt to fix completion logging

### Verifiable Rewards Experiments  
- **train_grpo_verifiable_callbacks.py** - Implementation using TRL callbacks
- **train_grpo_verifiable_with_tables.py** - Implementation with WandB tables

## Key Learnings from These Scripts

1. **KL Penalty Discovery** (train_grpo_arithmetic_kl_penalty.py):
   - Arithmetic tasks require beta=0.1 to prevent mode collapse
   - Without KL penalty, rewards stuck at -1.0

2. **Logging Issues** (train_grpo_arithmetic_with_tables.py):
   - log_completions=True causes AttributeError with TRL
   - Led to creating GRPOTrainerFixed class in the working baseline

3. **Failed Approaches**:
   - Ultra-aggressive learning rates (5e-3) don't help
   - Need proper temperature (0.7) for generation diversity

## Current Working Baseline

See `../train_grpo_arithmetic_fixed_completions.py` for the current best practice implementation that incorporates all learnings from these experiments.