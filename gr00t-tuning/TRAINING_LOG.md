# GR00T Training Log

## Overview
This log tracks experiments for fine-tuning/overfitting tests related to NVIDIA Isaac GR00T N1.5 robot foundation model.

**Goal**: Validate training pipeline and achieve successful overfitting as a first step.

**Update**: The Isaac-GR00T repository is actually available at https://github.com/NVIDIA/Isaac-GR00T (was looking under wrong organization name).

## Experiments

### 1. Simple Overfitting Test ✅
**Run ID**: j3247tru  
**Status**: SUCCESS  
**Date**: 2025-06-14 22:19 UTC  
**Script**: `train_gr00t_simple_overfit.py`

**Setup**:
- Model: Qwen/Qwen2-0.5B-Instruct
- Task: Generate "MOVE" when prompted with "Generate robot command:"
- Learning rate: 1e-3 (high for fast overfitting)
- Batch size: 1
- Max epochs: 100

**Results**:
- Final reward: 2.0 ✅
- Final loss: 9.48
- Overfit success: True (1)
- Achieved in epoch 0 (very fast!)

**Key Insights**:
- Simple reward function (2.0 for "MOVE", 0.0 otherwise) worked well
- High learning rate (1e-3) enabled rapid overfitting
- Model successfully learned the task

### 2. Robot Command RL Training ❌
**Run IDs**: tn30jf7w (finished), viv7uzzv (killed)  
**Status**: PARTIAL SUCCESS (ran but didn't achieve positive rewards)  
**Date**: 2025-06-14 22:10 UTC  
**Script**: `train_gr00t_rl.py`

**Setup**:
- Model: Qwen/Qwen2-0.5B-Instruct
- Task: Generate robot commands (MOVE_ARM, ROTATE_BASE, etc.)
- Learning rate: 5e-6
- Batch size: 4
- Num generations: 8
- Temperature: 0.7

**Results**:
- Best reward: 0.0 (never achieved positive rewards)
- Mean reward: -0.48 to -0.50
- Ran for 49 epochs / 201 steps

**Issues**:
- Reward function too strict - required exact command formats
- Model generated generic text instead of specific commands
- No learning signal (rewards stayed negative)

### 3. GRPO Arithmetic Overfitting ❌
**Run IDs**: yxk5wukm, 43rc9kyu, yjgwde3t, u0pht8iw  
**Status**: FAILED (various API/configuration errors)  
**Date**: 2025-06-14 22:28-22:37 UTC  
**Script**: `train_grpo_overfit_arithmetic.py`

**Issues Encountered**:
1. Missing imports (`List` from typing)
2. GRPOConfig API mismatch (no `model_name` parameter)
3. Batch size constraint (must be divisible by num_generations)
4. GRPOTrainer initialization differences

**Note**: This was mixing GRPO (language model RL) with GR00T (robot model) - conceptual confusion!

## Current Status

✅ **Success**: Simple overfitting test proved the basic training pipeline works
❌ **Challenge**: More complex reward functions (robot commands) didn't achieve learning
🚫 **Mistake**: Mixed up GRPO language model training with GR00T robot model fine-tuning

## Next Steps

1. **Clarify approach**: GR00T is about supervised fine-tuning on robot demonstrations, NOT reinforcement learning with rewards
2. **Find alternative**: Since Isaac-GR00T repo is inaccessible, consider:
   - Using a different robot learning framework
   - Creating synthetic robot demonstration data
   - Focusing on validating the training pipeline with simpler tasks
3. **Fix confusion**: Keep GR00T experiments separate from GRPO language model experiments

## Lessons Learned

1. **Start simple**: The basic "generate MOVE" task worked immediately
2. **Reward design matters**: Complex reward functions need careful tuning
3. **Check assumptions**: GR00T tutorial is about supervised learning, not RL
4. **API compatibility**: TRL/GRPO APIs have specific requirements that differ from standard training