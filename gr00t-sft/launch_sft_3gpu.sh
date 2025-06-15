#!/bin/bash
# Launch GR00T SFT training with 3 GPUs (skipping GPU 1 which is in use)

# Navigate to Isaac-GR00T directory (required for the script to work)
cd ~/pippa/Isaac-GR00T

# Activate the SFT virtual environment
source sft_venv/bin/activate

# Set which GPUs to use (skip GPU 1)
export CUDA_VISIBLE_DEVICES=0,2,3

# Set video backend
export VIDEO_BACKEND=torchvision_av

# Ensure WandB is configured
export WANDB_PROJECT=pippa
export WANDB_ENTITY=wild-ai
export WANDB_TAGS="gr00t-sft,so101-table-cleanup"

echo "Running GR00T SFT training with:"
echo "  - GPUs: 3 (GPUs 0, 2, 3)"
echo "  - Max steps: 10000"
echo "  - Dataset: SO-101 table cleanup"
echo "  - Data config: so100_dualcam"

# Run the exact command from the blog post with 3 GPUs and WandB
python scripts/gr00t_finetune.py \
    --dataset-path ../gr00t-sft/demo_data/so101-table-cleanup/ \
    --num-gpus 3 \
    --output-dir ../gr00t-sft/so101-checkpoints \
    --max-steps 10000 \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --report-to wandb