#!/bin/bash
# Aggressive overfitting experiments to achieve reward 1.0

echo "Running aggressive GRPO overfitting experiments..."
echo "Target: Achieve reward of 1.0 (perfect overfitting)"

# Experiment 1: Aggressive arithmetic overfitting
echo "=== Experiment 1: Aggressive Arithmetic Overfitting ==="
python train_grpo_verifiable.py \
    --task arithmetic \
    --batch_size 512 \
    --num_generations 32 \
    --epochs 100 \
    --lr 5e-5 \
    --temperature 0.3 \
    --beta 0.0 \
    --gradient_accumulation_steps 4 \
    --n_samples 50  # Smaller dataset for easier overfitting

# Experiment 2: Binary comparison (should be easier to overfit)
echo "=== Experiment 2: Aggressive Comparison Overfitting ==="
python train_grpo_verifiable.py \
    --task comparison \
    --batch_size 512 \
    --num_generations 32 \
    --epochs 50 \
    --lr 1e-4 \
    --temperature 0.1 \
    --beta 0.0 \
    --gradient_accumulation_steps 4 \
    --n_samples 25  # Very small dataset for guaranteed overfitting

echo "All aggressive overfitting experiments completed!"