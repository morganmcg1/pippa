#!/bin/bash
# Maximum GPU utilization for Absolute Zero unified training

cd ~/pippa
source az_venv/bin/activate

echo "Launching Maximum GPU Absolute Zero Training..."
echo "Configuration:"
echo "- 100 iterations for comprehensive training"
echo "- Batch size 192 (32 samples per task-role combination)"
echo "- num_generations=16 for maximum GPU utilization"
echo "- Large seed buffer (128) for diversity"
echo "- Gradient accumulation if needed"

# Batch size calculation:
# 192 total samples / 6 (3 tasks × 2 roles) = 32 per combination
# With num_generations=16: 192/16 = 12 effective batch size
# This should maximize H100 80GB memory usage

python absolute_zero/train_absolute_zero_unified.py \
    --model "Qwen/Qwen2-0.5B-Instruct" \
    --iterations 100 \
    --batch-size 192 \
    --seed-buffer-size 128 \
    --learning-rate 5e-6 \
    --temperature 0.7 \
    --beta 0.1 \
    --name-suffix "max_gpu_100iter_batch192"