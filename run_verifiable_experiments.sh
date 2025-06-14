#!/bin/bash
# Run GRPO experiments with verifiable rewards

echo "Running GRPO experiments with verifiable rewards..."

# Experiment 1: Arithmetic (easiest)
echo "=== Experiment 1: Arithmetic ==="
python train_grpo_verifiable.py \
    --task arithmetic \
    --batch_size 64 \
    --num_generations 16 \
    --epochs 20 \
    --lr 5e-6 \
    --temperature 0.7 \
    --beta 0.1

# Experiment 2: Counting
echo "=== Experiment 2: Counting ==="
python train_grpo_verifiable.py \
    --task counting \
    --batch_size 64 \
    --num_generations 16 \
    --epochs 20 \
    --lr 5e-6 \
    --temperature 0.7 \
    --beta 0.1

# Experiment 3: Comparison
echo "=== Experiment 3: Comparison ==="
python train_grpo_verifiable.py \
    --task comparison \
    --batch_size 64 \
    --num_generations 16 \
    --epochs 20 \
    --lr 5e-6 \
    --temperature 0.7 \
    --beta 0.1

# Experiment 4: Binary conversion
echo "=== Experiment 4: Binary Conversion ==="
python train_grpo_verifiable.py \
    --task binary \
    --batch_size 64 \
    --num_generations 16 \
    --epochs 20 \
    --lr 5e-6 \
    --temperature 0.7 \
    --beta 0.1

echo "All experiments completed!"