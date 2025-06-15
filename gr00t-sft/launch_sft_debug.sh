#!/bin/bash
# Launch GR00T SFT training in debug mode with fewer steps

# Navigate to Isaac-GR00T directory (required for the script to work)
cd ~/pippa/Isaac-GR00T

# Activate the SFT virtual environment
source sft_venv/bin/activate

# Set which GPUs to use (just GPU 0 for debug)
export CUDA_VISIBLE_DEVICES=0

# Set video backend
export VIDEO_BACKEND=torchvision_av

# Ensure WandB is configured
export WANDB_PROJECT=pippa
export WANDB_ENTITY=wild-ai
export WANDB_TAGS="gr00t-sft,so101-table-cleanup,debug"

# Parse command line arguments
MAX_STEPS=${1:-100}   # Default to 100 steps for quick test
BATCH_SIZE=${2:-2}    # Default to batch size 2 for lower memory

echo "Running GR00T SFT debug training with:"
echo "  - Max steps: ${MAX_STEPS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - GPUs: 1 (GPU 0)"
echo "  - Dataset: SO-101 table cleanup"

# Run with debug settings
python scripts/gr00t_finetune.py \
    --dataset-path ../gr00t-sft/demo_data/so101-table-cleanup/ \
    --num-gpus 1 \
    --output-dir ../gr00t-sft/so101-checkpoints-debug \
    --max-steps ${MAX_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --report-to wandb \
    --save-steps 50