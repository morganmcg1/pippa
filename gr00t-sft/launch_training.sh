#!/bin/bash
# Launch GR00T SFT training with WandB logging

# Activate the virtual environment
source ~/pippa/Isaac-GR00T/sft_venv/bin/activate

# Set CUDA visible devices (use GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Set video backend
export VIDEO_BACKEND=torchvision_av

# Run training with SO-101 dataset
python train_gr00t_sft.py \
    --dataset-path ./demo_data/so101-table-cleanup \
    --data-config so100_dualcam \
    --output-dir ./gr00t-sft-checkpoints \
    --max-steps 10000 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --save-steps 500 \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --embodiment-tag NEW_EMBODIMENT \
    --tune-visual False \
    --tune-llm False \
    --tune-projector True \
    --tune-diffusion-model True \
    --report-to wandb \
    --wandb-project pippa \
    --wandb-entity wild-ai \
    --wandb-tags "gr00t-sft" "so101-table-cleanup" \
    --num-gpus 1