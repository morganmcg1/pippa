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
- ✓ Created absolute_zero directory structure
- ✓ Initialized research journal
- ✓ Set up virtual environment (az_venv)
- ✓ Installed dependencies (torch, transformers, trl, etc.)
- ✓ Created baseline training script (train_absolute_zero_baseline.py)
- ✓ Updated CLAUDE.md with experiment details
- ✓ Committed to git

### Implementation Details - 2025-06-15_15:00
Created `train_absolute_zero_baseline.py` with:
- **AbsoluteZeroTrainer** class managing proposer and solver models
- **TRRPlusBaselines** class implementing 6 separate baselines
- Problem generation using high-temperature sampling (1.0)
- Learnability reward based on solver improvement
- Difficulty classification (easy/medium/hard)
- Integration with GRPO training loop
- Evaluation on standardized arithmetic dataset

Ready to run first experiment!

### First Experiment Launch - 2025-06-15_19:11 - Run ID: fv3ni9jh
- **Status**: FAILED (multiple attempts)
- **Issue**: Various initialization errors

### Current Run - 2025-06-15_19:21 - Run ID: rz2b20mh
- **Status**: RUNNING on H100 GPU (cuda)
- **WandB**: https://wandb.ai/wild-ai/pippa/runs/rz2b20mh
- **Configuration**: Same as above
- **Progress**: Iteration 5/20, Solver accuracy: 31.5%
- **Issue Found**: Proposer rewards are all 0 - the learnability reward calculation is broken

### Debugging Session - 2025-06-15_19:59
**Issues Identified:**
1. **Proposer rewards all 0**: The learnability reward is computed once for entire batch
2. **Poor problem generation**: Proposer doesn't understand the format
3. **No visibility**: Need logging to see what proposer is generating

**Fixes Applied:**
1. Added comprehensive print logging for proposer outputs
2. Added WandB table logging for both proposer generations and solver results
3. Improved prompt engineering with explicit format examples
4. Added raw generation tracking to debug parsing issues

**Key Changes:**
- Modified `generate_problems()` to return raw generations for debugging
- Added WandB tables: `proposer_generations` and `solver_results`
- Better prompts with explicit format: "Calculate: X op Y = "
- Print logging shows parsing success rate and sample outputs

### Critical Discovery - 2025-06-15_20:19
**Proposer Failure**: Testing iter 6 proposer revealed it's generating complex word problems instead of arithmetic!

Examples of broken output:
- "20% of the vets in a state recommend Puppy Kibble..."
- "36 people attend a party. Each person shakes hands..."
- Python code snippets instead of arithmetic problems

**Root Cause**: 
1. Proposer training with 0 rewards led to mode collapse
2. No few-shot examples to guide generation
3. Model defaulted to its pre-training on general text

### Emergency Fix Session - 2025-06-15_20:20
**Actions Taken:**
1. Stopped the broken run (iteration 8/20)
2. Added few-shot examples to all prompts:
   ```
   "Generate arithmetic: Calculate: 5 + 3 = 
    Generate arithmetic: Calculate: 12 - 7 = 
    Generate arithmetic: "
   ```
3. Fixed per-problem reward calculation
4. Improved parsing to handle multiple formats
5. Launched new run: `absolute_zero_improved`

**New Run Details:**
- Session: `absolute_zero_improved` 
- Start time: 2025-06-15_20:21
- Run ID: iwpb33bn
- Key fixes: Few-shot prompts, individual rewards, better parsing
- **Early Results**: Proposer generating valid arithmetic! Rewards working (avg 1.039)

### Improved Problem Generation - 2025-06-15_20:35
**Enhancements Implemented:**
1. **Oversampling**: Generate 2x problems, use only valid ones
2. **Multi-generation**: Use `num_return_sequences=4` for efficiency  
3. **Validation Pipeline**: Parse → Verify format → Compute answer
4. **Success Rate Tracking**: Monitor proposer's generation quality

**Key Learning**: The proposer was generating ~100 problems per iteration. With oversampling and validation, we can ensure all problems used for training are valid arithmetic.

### Paper Clarification - 2025-06-15_20:40
**Research Finding**: The Absolute Zero paper confirms our approach:
- Proposer generates **tasks only**, not answers
- External verifier (code executor/Python) provides ground truth
- Proposer learns from "learnability rewards" - which problems help solver improve
- Focus is on curriculum learning, not proposer's arithmetic ability

**Implication**: Our implementation is correctly aligned with the paper's design philosophy.

## Key Metrics to Track
1. **Solver Accuracy**: On standardized eval set
2. **Problem Difficulty**: Average difficulty of generated problems
3. **Problem Diversity**: Unique problem types generated
4. **Learnability Score**: How much solver improves on proposed problems
5. **Proposer Reward**: Average reward received by proposer
6. **Curriculum Progression**: How difficulty changes over time

## Technical Considerations

### Critical Design Decision: Should Proposer Generate Answers?

**Current Approach**: Proposer generates "Calculate: 5 + 3 = ", we compute answer in Python
**Alternative**: Proposer generates "Calculate: 5 + 3 = 8"

**Paper Analysis (2025-06-15_20:30)**: After researching the Absolute Zero paper:
- The paper focuses on **code reasoning tasks**, not arithmetic
- Proposer generates tasks/problems, **NOT answers**
- A **verifier** (code executor in paper, Python in our case) determines correctness
- This separation is intentional and correct!

**Key Insight**: The proposer's job is to learn what problems help the solver improve, not to solve them itself. Our current approach aligns with the paper's design.

**Pros of Current Approach:**
- Simpler learning task (pattern generation only)
- Guaranteed correct answers for solver training
- Faster convergence
- **Aligned with paper's approach** ✓

**Decision**: Current approach is correct. Proposer generates problems, Python computes answers.

### Memory Management
- Two models in memory (proposer + solver)
- May need gradient checkpointing
- Consider sharing base model weights

### Qwen2-0.5B Usage Notes (from HuggingFace docs)
**Key insights for proper inference:**
1. **Model Loading**:
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
   ```

2. **Proper Generation Parameters**:
   - Use `do_sample=True` for diverse outputs
   - Temperature controls randomness (higher = more diverse)
   - `top_k` and `top_p` for nucleus sampling
   - Set proper `pad_token_id` to avoid warnings

3. **For Arithmetic Problem Generation**:
   - Clear, specific prompts work better
   - The model understands instruction-following
   - May need few-shot examples in prompt

4. **Tokenizer Setup**:
   ```python
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token
   ```

### Evaluation Protocol
1. **Solver Evaluation**: Standard morgan/arithmetic_eval dataset
2. **Proposer Evaluation**: Quality and diversity of generated problems
3. **Curriculum Analysis**: Track difficulty progression

### Expected Challenges
1. **Reward Sparsity**: Early proposer may generate unhelpful problems
2. **Mode Collapse**: Proposer might converge to single problem type
3. **Credit Assignment**: Determining which problems actually helped
4. **Compute Cost**: Training two models simultaneously

## Current Status (2025-06-15_20:45)

### Running Experiment: `absolute_zero_improved`
- **WandB Run ID**: iwpb33bn
- **Status**: Training in progress (iteration 3/20 as of 20:40)
- **Proposer Performance**: Generating valid arithmetic problems with few-shot prompts
- **Solver Performance**: 18% accuracy (concerning - much lower than 75% GRPO baseline)
- **Key Metrics**:
  - Iteration 1: Solver 21.5% accuracy (43/200 correct)
  - Iteration 2: Solver 18.0% accuracy (36/200 correct) - DECLINING!
  - Proposer rewards: Working but very low (mean 0.10)
  - Problem generation: Successfully parsing arithmetic format
  - Training time: ~11 minutes per iteration

### Concerning Issues
- **Solver accuracy is decreasing** (21.5% → 18.0%)
- **Much worse than GRPO baseline** (18% vs 75%)
- **Proposer rewards very low** (0.10 mean)

### Next Steps
1. Monitor full 20 iterations for curriculum emergence
2. Investigate why solver accuracy is so low
3. Check if proposer is generating appropriate difficulty problems
4. Consider adjusting hyperparameters if no improvement by iteration 5

### Lessons Learned
1. **Few-shot prompts are critical** - Without examples, model reverts to general text
2. **Individual rewards matter** - Per-problem rewards prevent mode collapse
3. **Oversampling helps** - Generate extra to ensure quality
4. **Paper alignment confirmed** - Proposer generates problems, not answers
5. **Self-play may hurt initial performance** - Solver accuracy much lower than supervised baseline (18% vs 75%)
6. **Curriculum learning is slow** - No improvement in first 2 iterations

## References
- [Absolute Zero Paper](https://arxiv.org/pdf/2505.03335v2)
- [GRPO Baseline Results](./GRPO_baseline.md)
- Best GRPO config: train_grpo_arithmetic_fixed_completions.py

---
*This journal will be updated as experiments progress.*