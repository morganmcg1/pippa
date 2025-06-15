#!/bin/bash
# Launch GR00T SFT training following the exact blog post configuration
# But using 4 GPUs and ensuring WandB is enabled

# Navigate to Isaac-GR00T directory (required for the script to work)
cd ~/pippa/Isaac-GR00T

# Activate the SFT virtual environment
source sft_venv/bin/activate

# Set video backend
export VIDEO_BACKEND=torchvision_av

# Ensure WandB is configured
export WANDB_PROJECT=pippa
export WANDB_ENTITY=wild-ai

# Run the exact command from the blog post, but with 4 GPUs and WandB
python scripts/gr00t_finetune.py \
    --dataset-path ../gr00t-sft/demo_data/so101-table-cleanup/ \
    --num-gpus 4 \
    --output-dir ../gr00t-sft/so101-checkpoints \
    --max-steps 10000 \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --report-to wandb