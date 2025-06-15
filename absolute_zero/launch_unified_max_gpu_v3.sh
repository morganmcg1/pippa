#!/bin/bash
# ULTRA Maximum GPU utilization for Absolute Zero unified training
# Target: 90-95% GPU memory usage

cd ~/pippa
source az_venv/bin/activate

echo "Launching ULTRA MAXIMUM GPU Absolute Zero Training v3..."
echo "Configuration:"
echo "- 100 iterations for comprehensive training"
echo "- Batch size 510 (85 samples per task-role combination)"
echo "- num_generations=30 for high parallelism"
echo "- Large seed buffer (512) for maximum diversity"
echo "- Pushing H100 to its limits!"

# Batch size calculation:
# 510 total samples / 6 (3 tasks × 2 roles) = 85 per combination
# With num_generations=30: 510/30 = 17 effective batch size per generation
# This should use 90-95% of H100 80GB memory

python absolute_zero/train_absolute_zero_unified.py \
    --model "Qwen/Qwen2-0.5B-Instruct" \
    --iterations 100 \
    --batch-size 510 \
    --seed-buffer-size 512 \
    --learning-rate 5e-6 \
    --temperature 0.7 \
    --beta 0.1 \
    --name-suffix "ultra_max_gpu_batch510_gen30"