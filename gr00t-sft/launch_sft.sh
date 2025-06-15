#!/bin/bash
# Unified GR00T SFT launch script with configurable parameters

# Parse command line arguments
SAMPLES=${1:-1000}      # Number of samples to use (default: 1000)
MAX_STEPS=${2:-100}     # Maximum training steps (default: 100)
BATCH_SIZE=${3:-4}      # Batch size per GPU (default: 4)
NUM_GPUS=${4:-1}        # Number of GPUs to use (default: 1)
TAG_SUFFIX=${5:-""}     # Optional tag suffix for WandB

# Generate run name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="gr00t-sft-${SAMPLES}samples-${TIMESTAMP}"

# Build tags
TAGS="gr00t-sft,so101-table-cleanup"
if [ -n "$TAG_SUFFIX" ]; then
    TAGS="${TAGS},${TAG_SUFFIX}"
fi

# Output directory
OUTPUT_DIR="./gr00t-sft-checkpoints/${RUN_NAME}"

echo "==================================================="
echo "GR00T SFT Training Configuration:"
echo "==================================================="
echo "  Samples: ${SAMPLES}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Run Name: ${RUN_NAME}"
echo "  Tags: ${TAGS}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo "==================================================="

# Navigate to gr00t-sft directory
cd ~/pippa/gr00t-sft

# Activate virtual environment
source ~/pippa/Isaac-GR00T/sft_venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export VIDEO_BACKEND=torchvision_av
export PYTHONPATH="${HOME}/pippa/Isaac-GR00T:${PYTHONPATH}"

# Run training
python train_gr00t_sft.py \
    --dataset-path ./demo_data/so101-table-cleanup/ \
    --output-dir "${OUTPUT_DIR}" \
    --max-samples ${SAMPLES} \
    --max-steps ${MAX_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --num-gpus ${NUM_GPUS} \
    --wandb-run-name "${RUN_NAME}" \
    --wandb-tags "${TAGS}" \
    --save-steps 100 \
    --no-tune-visual \
    --no-tune-llm \
    --tune-projector \
    --tune-diffusion-model \
    --dataloader-num-workers 4 \
    --learning-rate 1e-4 \
    --weight-decay 0.05 \
    --warmup-ratio 0.05