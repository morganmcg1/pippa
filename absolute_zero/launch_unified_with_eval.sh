#!/bin/bash

# Launch script for Absolute Zero unified training with evaluation
# This version includes:
# - Global step tracking fix
# - Periodic evaluation on arithmetic_eval dataset
# - Final accuracy reporting

cd ~/pippa

# Activate virtual environment
source absolute_zero/az_venv/bin/activate

# Ensure dependencies are up to date
cd absolute_zero
uv pip sync

# Launch training with evaluation
python train_absolute_zero_unified.py \
    --model "Qwen/Qwen2-0.5B-Instruct" \
    --iterations 20 \
    --batch-size 64 \
    --seed-buffer-size 32 \
    --learning-rate 5e-6 \
    --temperature 0.7 \
    --beta 0.1 \
    --eval-steps 10 \
    --name-suffix "with_eval_batch64"