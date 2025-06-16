#!/bin/bash

# Launch script for Absolute Zero unified training with maximum GPU utilization v3
# This version includes fixes for global step tracking

cd ~/pippa

# Activate virtual environment
source absolute_zero/az_venv/bin/activate

# Ensure dependencies are up to date
cd absolute_zero
uv pip sync

# Launch training with the global step fix
python train_absolute_zero_unified.py \
    --model "Qwen/Qwen2-0.5B-Instruct" \
    --iterations 100 \
    --batch-size 384 \
    --seed-buffer-size 256 \
    --learning-rate 5e-6 \
    --temperature 0.7 \
    --beta 0.1 \
    --name-suffix "max_gpu_v3_batch384_gen32_fixed_global_step"