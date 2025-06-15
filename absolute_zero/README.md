# Absolute Zero Implementation

This directory contains the implementation of the Absolute Zero algorithm from the paper "Absolute Zero: Mathematics as a Self-Play Repertoire" (https://arxiv.org/pdf/2505.03335v2).

## Overview

Absolute Zero uses a **single unified model** that alternates between two roles:
- **Proposer**: Generates arithmetic tasks across three types (deduction, abduction, induction)
- **Solver**: Attempts to solve the generated tasks
- The proposer is rewarded for generating problems that help the solver improve (learnability reward)
- Joint training with single RL update per iteration using GRPO

## Quick Start

```bash
# Activate virtual environment
source az_venv/bin/activate

# Run maximum GPU utilization training
./launch_unified_max_gpu_v2.sh
```

## Key Features

1. **Unified Model**: Single model learns both proposer and solver roles simultaneously

2. **Three Task Types**:
   - **Deduction**: "Calculate: 5 + 3 = ?" (standard arithmetic)
   - **Abduction**: "Find: ? + ? = 8" (reverse engineering)
   - **Induction**: "Pattern: (2,3)→5, (4,1)→5..." (rule inference)

3. **TRR++ (Task-Relative REINFORCE++)**: Uses 6 separate baselines for variance reduction:
   - {proposer, solver} × {deduction, abduction, induction}

4. **Seeding Phase**: Initial task buffer population without gradients

5. **Natural Curriculum**: Difficulty emerges automatically as model learns

## Files

- `train_absolute_zero_unified.py` - Main unified implementation
- `launch_unified_max_gpu_v2.sh` - Optimized launch script (94% GPU utilization)
- `../research_journal/absolute_zero.md` - Detailed experiment notes and results

## Optimal Configuration (H100)

Current best configuration achieving 94.4% GPU utilization:
- Model: Qwen/Qwen2-0.5B-Instruct
- Temperature: 0.7
- KL Penalty: beta=0.1
- Learning Rate: 5e-6
- Batch Size: 384
- Number of Generations: 32
- Seed Buffer Size: 256

## Monitoring

Training progress is logged to WandB:
- Project: pippa
- Tags: absolute-zero, arithmetic, self-play

View at: https://wandb.ai/wild-ai/pippa

## Dependencies

All dependencies are installed in the virtual environment:
- torch, transformers, trl
- datasets, accelerate
- wandb, python-dotenv

## Next Experiments

1. **Reward Ablation**: Test components of learnability reward
2. **Baseline Ablation**: Compare 6 vs 2 vs 1 baseline
3. **Scale Test**: Larger batch sizes and models
4. **Diversity Mechanisms**: Explicit diversity rewards

See `../research_journal/absolute_zero.md` for detailed plans.