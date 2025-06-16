#!/bin/bash

# Quick test run to verify logging improvements
# Tests:
# - Global step tracking across iterations
# - Evaluation metrics (eval/arithmetic_eval)
# - WandB tables
# - No overwriting of logs

cd ~/pippa

# Activate virtual environment
source absolute_zero/az_venv/bin/activate

# Ensure dependencies are up to date
cd absolute_zero
uv pip sync

# Launch quick test with small batch size and few iterations
python train_absolute_zero_unified.py \
    --model "Qwen/Qwen2-0.5B-Instruct" \
    --iterations 3 \
    --batch-size 12 \
    --seed-buffer-size 16 \
    --learning-rate 5e-6 \
    --temperature 0.7 \
    --beta 0.1 \
    --eval-steps 5 \
    --name-suffix "test_logging_v1"