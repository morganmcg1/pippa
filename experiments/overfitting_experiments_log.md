# GRPO Overfitting Experiments Log

## Date: 2025-06-14

### Experiment Goal
Achieve reward of 1.0 (perfect overfitting) to validate our GRPO training pipeline.

## Key Learnings So Far

### 1. Batch Size Must Be <= Dataset Size
- **Failed**: Batch size 512 with 50 samples → DataLoader error
- **Lesson**: GRPO's dataloader cannot handle batch sizes larger than the dataset
- **Solution**: Use batch_size <= n_samples

### 2. Progress Towards Overfitting
| Experiment | Task | Best Reward | Status | Notes |
|------------|------|-------------|---------|-------|
| Comparison (lkxgtkc4) | Yes/No | 0.75 | Completed 20 epochs | Good progress! |
| Arithmetic (hfpy0s4q) | Math | -1.0 | Running | Still failing - needs more epochs |
| Binary (6vrh8qp9) | Binary conversion | 0.34 | Running | Some learning happening |

### 3. Hyperparameter Insights

#### What's NOT Working:
- **Low learning rates** (5e-6): Too slow for overfitting
- **Standard temperatures** (0.7): Too much randomness
- **Large datasets** (100 samples): Harder to overfit

#### What IS Working:
- **Comparison task**: Simpler binary yes/no answers
- **Higher learning rates**: Need 1e-4 to 5e-4 for aggressive overfitting
- **MODERATE-HIGH temperatures**: 0.7-1.0 for generation diversity (NOT low!)
- **Small datasets**: 16-32 samples maximum
- **Many generations**: 128-256 to explore reward space

### 4. Task Difficulty Ranking (Easiest to Hardest)
1. **Comparison (Yes/No)**: Binary output, very clear signal → 0.75 reward achieved
2. **Binary conversion**: Limited output space (0-15 → 4 bits max)
3. **Arithmetic**: Larger output space, more complex
4. **Counting**: Variable difficulty based on sentence complexity

### 5. Critical Requirements for GRPO Success
1. **Verifiable rewards**: Must have binary correct/incorrect signal
2. **Model understanding**: Base model must already "know" the task
3. **Clear patterns**: Tasks with deterministic rules work best
4. **Sufficient generations**: Need many samples per prompt (16-128)

## Current Running Experiments

### 1. Arithmetic Overfitting v2 (hfpy0s4q)
- **Config**: batch=32, gen=64, lr=1e-4, temp=0.1, epochs=200
- **Status**: Reward still at -1.0 (failing completely)
- **Issue**: May need even higher learning rate or simpler problems

### 2. Binary Conversion (6vrh8qp9)
- **Config**: batch=64, gen=16, lr=5e-6, temp=0.7
- **Status**: Reward 0.34 at epoch 4
- **Progress**: Showing some learning, but slow

## Next Steps to Achieve Reward 1.0

1. **Ultra-aggressive comparison experiment**:
   - Only 10 samples (guaranteed overfitting)
   - Learning rate 1e-3
   - Temperature 0.01
   - 500+ epochs if needed

2. **Simplify arithmetic**:
   - Only addition, numbers 0-5
   - Single digit answers only

3. **Consider task-specific tweaks**:
   - Increase max_completion_length for tasks needing longer outputs
   - Use reward shaping (partial credit for close answers)

## Failed Approaches to Avoid
- Echo tasks ("Say X") - no verifiable correctness
- Pattern completion without clear answers
- Any task where the model generates free-form text
- Batch sizes larger than dataset size
- Learning rates below 1e-5 for overfitting

## Critical Temperature Discovery (Added 2025-06-14)

### The Mistake
Initially tried temperature 0.01 thinking deterministic outputs would help overfitting.

### Why This Fails with GRPO
- GRPO advantage formula: `(reward - mean) / std`
- Low temperature → all generations nearly identical
- Same outputs → same rewards → **zero standard deviation**
- Zero std → no learning signal!

### The Fix
- Use temperature 0.7-1.0 for healthy generation diversity
- Need variance in reward distribution for GRPO to work
- More generations (256+) help explore the reward space

## Technical Issues Encountered
1. **DataLoader error with large batch sizes**: UnboundLocalError in accelerate
2. **Zero gradients**: Happens when all rewards are identical (no learning signal)
3. **Reward -1.0 stuck**: Model not understanding task format
4. **Low temperature trap**: Temperature too low → no diversity → zero std → no learning