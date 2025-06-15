#!/bin/bash
# Launch script for unified Absolute Zero implementation

cd ~/pippa
source az_venv/bin/activate

echo "Launching Absolute Zero Unified Implementation..."
echo "This implementation follows the paper's approach:"
echo "1. Single model for both proposer and solver"
echo "2. Seeding phase before training"
echo "3. Three task types: Deduction, Abduction, Induction"
echo "4. Joint training with single RL update"

python absolute_zero/train_absolute_zero_unified.py \
    --model "Qwen/Qwen2-0.5B-Instruct" \
    --iterations 20 \
    --batch-size 12 \
    --seed-buffer-size 32 \
    --learning-rate 5e-6 \
    --temperature 0.7 \
    --beta 0.1 \
    --name-suffix "paper_aligned"