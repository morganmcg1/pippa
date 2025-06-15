#!/bin/bash
# Quick test script with minimal parameters

cd ~/pippa
source az_venv/bin/activate

echo "Running quick test with minimal parameters..."
echo "Arguments: --iterations 2 --samples 10 --solver-epochs 1 --proposer-epochs 1"

python absolute_zero/train_absolute_zero_baseline.py \
    --iterations 2 \
    --samples 10 \
    --solver-epochs 1 \
    --proposer-epochs 1 \
    --batch-size 4 \
    --name-suffix quick_table_test