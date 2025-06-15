#!/bin/bash
# TRULY Maximum GPU utilization for Absolute Zero unified training
# Target: 80-90% GPU memory usage

cd ~/pippa
source az_venv/bin/activate

echo "Launching MAXIMUM GPU Absolute Zero Training v2..."
echo "Configuration:"
echo "- 100 iterations for comprehensive training"
echo "- Batch size 384 (64 samples per task-role combination)"
echo "- num_generations=32 for high parallelism"
echo "- Large seed buffer (256) for diversity"
echo "- No gradient accumulation needed"

# Batch size calculation:
# 384 total samples / 6 (3 tasks × 2 roles) = 64 per combination
# With num_generations=32: 384/32 = 12 effective batch size per generation
# This should use 80-90% of H100 80GB memory

python absolute_zero/train_absolute_zero_unified.py \
    --model "Qwen/Qwen2-0.5B-Instruct" \
    --iterations 100 \
    --batch-size 384 \
    --seed-buffer-size 256 \
    --learning-rate 5e-6 \
    --temperature 0.7 \
    --beta 0.1 \
    --name-suffix "max_gpu_v2_batch384_gen32"