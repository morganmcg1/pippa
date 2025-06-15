#!/bin/bash
# Launch GR00T SFT training with reduced memory usage

# Activate the virtual environment
source ~/pippa/Isaac-GR00T/sft_venv/bin/activate

# Set CUDA visible devices (use GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Set video backend
export VIDEO_BACKEND=torchvision_av

# Run training with reduced memory settings
python train_gr00t_sft.py \
    --dataset-path ./demo_data/so101-table-cleanup \
    --data-config so100_dualcam \
    --output-dir ./gr00t-sft-checkpoints-low-mem \
    --max-steps 1000 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --save-steps 100 \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --embodiment-tag NEW_EMBODIMENT \
    --tune-visual False \
    --tune-llm False \
    --tune-projector True \
    --tune-diffusion-model False \
    --report-to wandb \
    --wandb-project pippa \
    --wandb-entity wild-ai \
    --wandb-tags "gr00t-sft" "so101-table-cleanup" "low-memory" \
    --wandb-run-name "gr00t-sft-low-memory-test" \
    --num-gpus 1