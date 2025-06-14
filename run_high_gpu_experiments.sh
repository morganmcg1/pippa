#!/bin/bash
# Run high GPU utilization experiments

echo "Running high GPU utilization experiments..."

# Experiment 1: Echo task with high batch size and generations
echo "=== Experiment 1: Echo High GPU ==="
python train_grpo_high_gpu_overfit.py \
    --task echo \
    --batch_size 128 \
    --num_generations 64 \
    --epochs 30 \
    --lr 2e-5 \
    --temperature 0.5 \
    --n_samples 50 \
    --gradient_accumulation_steps 2 \
    --num_iterations 2 \
    --epsilon 0.1

# Experiment 2: Echo task with even higher utilization
echo "=== Experiment 2: Echo Max GPU ==="
python train_grpo_high_gpu_overfit.py \
    --task echo \
    --batch_size 256 \
    --num_generations 128 \
    --epochs 20 \
    --lr 1e-5 \
    --temperature 0.3 \
    --n_samples 50 \
    --gradient_accumulation_steps 4 \
    --num_iterations 3 \
    --epsilon 0.05

# Experiment 3: Pattern task with high GPU
echo "=== Experiment 3: Pattern High GPU ==="
python train_grpo_high_gpu_overfit.py \
    --task pattern \
    --batch_size 128 \
    --num_generations 64 \
    --epochs 40 \
    --lr 3e-5 \
    --temperature 0.6 \
    --n_samples 50 \
    --gradient_accumulation_steps 2 \
    --num_iterations 2 \
    --epsilon 0.1

echo "All experiments completed!"