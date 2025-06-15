# Absolute Zero Research Journal

## Overview
This document contains all research notes, experiment results, and learnings from implementing the Absolute Zero algorithm (https://arxiv.org/pdf/2505.03335v2) using the Qwen2-0.5B model. We're applying learnings from our successful GRPO experiments to this new self-play approach.

## Task Description: Arithmetic Self-Play with Learnability

### Goal
Implement Absolute Zero algorithm which uses:
1. **Proposer Model**: Generates new problems based on current capability
2. **Solver Model**: Solves the generated problems  
3. **Learnability Rewards**: Rewards proposer for problems that help solver improve
4. **TRR++ Algorithm**: Task-Relative REINFORCE++ with 6 separate baselines

### Key Differences from GRPO
- **Self-Generated Problems**: Instead of fixed dataset, proposer creates new problems
- **Dual Models**: Separate proposer and solver (though can be same base model)
- **Learnability Signal**: Proposer rewarded for problems that improve solver
- **Curriculum Emerges**: Natural curriculum as proposer learns what helps solver

### Algorithm Components (From Paper)
1. **Task-Relative REINFORCE++ (TRR++)**:
   - 6 separate baselines: proposer/solver × easy/medium/hard
   - Reduces variance in policy gradient estimates
   - Critical for stable training

2. **Learnability Reward**:
   - R_proposer = improvement in solver's accuracy on generated problems
   - Encourages proposer to generate problems at edge of solver's capability
   - Prevents too easy (no learning) or too hard (frustration)

3. **Diversity Bonus**:
   - Additional reward for novel problem structures
   - Prevents mode collapse to single problem type

### Base Configuration (From GRPO Success)
Based on our 75% GRPO success, we'll use:
- **Model**: Qwen/Qwen2-0.5B-Instruct
- **Temperature**: 0.7 (critical for diversity)
- **KL Penalty**: beta=0.1 (prevents mode collapse)
- **Learning Rate**: 5e-6
- **Batch Size**: 256
- **Number of Generations**: 16 per prompt
- **Evaluation**: morgan/arithmetic_eval dataset (200 problems)

## Implementation Plan

### Phase 1: Basic Infrastructure (Current)
1. ✓ Create folder structure
2. ✓ Create research journal
3. Set up virtual environment
4. Create base training script with proposer/solver structure

### Phase 2: Core Algorithm
1. Implement TRR++ with 6 baselines
2. Add learnability reward calculation  
3. Implement diversity bonus
4. Create evaluation loop

### Phase 3: Experiments
1. **Baseline**: Direct replication with arithmetic
2. **Reward Ablation**: Test components of learnability reward
3. **Baseline Ablation**: Compare 6 vs 2 vs 1 baseline
4. **Scale Test**: Try larger batch sizes

## Experiment Log

### Setup - 2025-06-15_14:45
- Created absolute_zero directory structure
- Initialized research journal
- Next: Set up virtual environment and base script

## Key Metrics to Track
1. **Solver Accuracy**: On standardized eval set
2. **Problem Difficulty**: Average difficulty of generated problems
3. **Problem Diversity**: Unique problem types generated
4. **Learnability Score**: How much solver improves on proposed problems
5. **Proposer Reward**: Average reward received by proposer
6. **Curriculum Progression**: How difficulty changes over time

## Technical Considerations

### Memory Management
- Two models in memory (proposer + solver)
- May need gradient checkpointing
- Consider sharing base model weights

### Evaluation Protocol
1. **Solver Evaluation**: Standard morgan/arithmetic_eval dataset
2. **Proposer Evaluation**: Quality and diversity of generated problems
3. **Curriculum Analysis**: Track difficulty progression

### Expected Challenges
1. **Reward Sparsity**: Early proposer may generate unhelpful problems
2. **Mode Collapse**: Proposer might converge to single problem type
3. **Credit Assignment**: Determining which problems actually helped
4. **Compute Cost**: Training two models simultaneously

## References
- [Absolute Zero Paper](https://arxiv.org/pdf/2505.03335v2)
- [GRPO Baseline Results](./GRPO_baseline.md)
- Best GRPO config: train_grpo_arithmetic_fixed_completions.py

---
*This journal will be updated as experiments progress.*