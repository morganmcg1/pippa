# GRPO Training Guide

## Quick Start

### For Verifiable Rewards Tasks (Recommended)
```bash
# Run arithmetic task with verifiable rewards
python train_grpo_verifiable.py --task arithmetic --batch_size 64 --num_generations 16

# Run all verifiable experiments
./run_verifiable_experiments.sh
```

### For Original Math Training
```bash
# Run with WandB tracking
python train_grpo_wandb.py

# Run on remote H100
./remote_train.sh
```

## Key Scripts

### Production Ready
1. **`train_grpo.py`** - Core Dr GRPO implementation
   - Implements bias-free advantage computation
   - Supports multiple reward functions
   - Best practices from cleanRL

2. **`train_grpo_wandb.py`** - Simple WandB wrapper
   - Just calls train_grpo.py with track=True
   - Use this for easy WandB integration

3. **`train_grpo_verifiable.py`** - GRPO with verifiable rewards
   - Arithmetic, counting, comparison, binary conversion tasks
   - Clear success/failure criteria
   - Tagged with "grpo-setup" in WandB

### Utilities
- `remote_train.sh` - Easy remote training with tmux
- `run_verifiable_experiments.sh` - Run all verifiable task experiments

## Failed Approaches (in experiments/failed_approaches/)
These demonstrate what doesn't work with GRPO:
- Echo tasks ("Say X") - no verifiable correctness
- Pattern completion without clear answers
- Any task where the model just generates text continuations

## Key Insights
1. **GRPO needs verifiable rewards** - Binary correct/incorrect works best
2. **Model must understand the task** - GRPO elicits capabilities, doesn't teach
3. **Use dr_grpo loss type** - Avoids bias in advantage computation
4. **Set beta=0.0** for rule-based rewards (no KL penalty needed)

## WandB Monitoring
All runs are tracked to `wild-ai/pippa` project with appropriate tags:
- "grpo-setup" - Initial GRPO experiments
- "verifiable-rewards" - Tasks with clear correctness
- Task-specific tags (arithmetic, counting, etc.)