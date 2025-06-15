# Absolute Zero Implementation

This directory contains the implementation of the Absolute Zero algorithm from the paper "Absolute Zero: Mathematics as a Self-Play Repertoire" (https://arxiv.org/pdf/2505.03335v2).

## Overview

Absolute Zero is a self-play algorithm where:
- A **proposer** model generates arithmetic problems
- A **solver** model attempts to solve them
- The proposer is rewarded for generating problems that help the solver improve (learnability reward)
- Both models are trained simultaneously using GRPO

## Quick Start

```bash
# Activate virtual environment
source az_venv/bin/activate

# Run baseline experiment
python train_absolute_zero_baseline.py
```

## Key Features

1. **TRR++ (Task-Relative REINFORCE++)**: Uses 6 separate baselines for variance reduction:
   - proposer × {easy, medium, hard}
   - solver × {easy, medium, hard}

2. **Learnability Rewards**: Proposer rewarded based on:
   - Solver improvement on generated problems
   - Optimal difficulty distribution
   - Problem diversity

3. **Natural Curriculum**: Difficulty emerges automatically as proposer learns what helps solver

## Files

- `train_absolute_zero_baseline.py` - Main implementation
- `../research_journal/absolute_zero.md` - Detailed experiment notes and results

## Configuration

Based on successful GRPO experiments:
- Model: Qwen/Qwen2-0.5B-Instruct
- Temperature: 0.7 (solver), 1.0 (proposer)
- KL Penalty: beta=0.1
- Learning Rate: 5e-6
- Batch Size: 32
- Evaluation: morgan/arithmetic_eval dataset

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