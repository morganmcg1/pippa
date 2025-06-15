#!/bin/bash
# Launch script for GR00T SFT evaluation with common configurations

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ARTIFACT_PATH=""
MODEL_PATH=""
NUM_TRAJECTORIES=5
DATASET_PATH="demo_data/so101-table-cleanup"
DATA_CONFIG="so100_dualcam"
PLOT=true
OUTPUT_DIR="./gr00t-eval-results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --artifact)
            ARTIFACT_PATH="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --trajectories)
            NUM_TRAJECTORIES="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --no-plot)
            PLOT=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --artifact PATH      WandB artifact path (e.g., 'wild-ai/pippa/gr00t-sft-so100_dualcam-bs32:latest')"
            echo "  --model-path PATH    Local model checkpoint path"
            echo "  --trajectories N     Number of trajectories to evaluate (default: 5)"
            echo "  --dataset PATH       Dataset path (default: demo_data/so101-table-cleanup)"
            echo "  --no-plot           Disable plotting"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Evaluate from WandB artifact"
            echo "  $0 --artifact 'wild-ai/pippa/gr00t-sft-so100_dualcam-bs32:latest'"
            echo ""
            echo "  # Evaluate from local checkpoint"
            echo "  $0 --model-path ./so101-checkpoints/checkpoint-10000"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if either artifact or model path is provided
if [ -z "$ARTIFACT_PATH" ] && [ -z "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Error: Must provide either --artifact or --model-path${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Activate virtual environment
VENV_PATH="$HOME/pippa/Isaac-GR00T/sft_venv"
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source "$VENV_PATH/bin/activate"
else
    echo -e "${YELLOW}Warning: Virtual environment not found at $VENV_PATH${NC}"
    echo "Continuing without virtual environment activation..."
fi

# Set environment variables
export PYTHONPATH="$HOME/pippa/Isaac-GR00T:$PYTHONPATH"
export VIDEO_BACKEND="torchvision_av"

# Print configuration
echo -e "${BLUE}=== GR00T SFT Evaluation Configuration ===${NC}"
if [ -n "$ARTIFACT_PATH" ]; then
    echo "Artifact: $ARTIFACT_PATH"
else
    echo "Model Path: $MODEL_PATH"
fi
echo "Dataset: $DATASET_PATH"
echo "Trajectories: $NUM_TRAJECTORIES"
echo "Plotting: $PLOT"
echo "Output Dir: $OUTPUT_DIR"
echo -e "${BLUE}=========================================${NC}"

# Build command
CMD="python gr00t-sft/eval_gr00t_sft.py"

if [ -n "$ARTIFACT_PATH" ]; then
    CMD="$CMD --wandb-artifact-path '$ARTIFACT_PATH'"
else
    CMD="$CMD --model-path '$MODEL_PATH'"
fi

CMD="$CMD --dataset-path '$DATASET_PATH'"
CMD="$CMD --num-trajectories $NUM_TRAJECTORIES"
CMD="$CMD --data-config $DATA_CONFIG"
CMD="$CMD --output-dir '$OUTPUT_DIR'"

if [ "$PLOT" = true ]; then
    CMD="$CMD --plot"
fi

# Run evaluation
echo -e "${GREEN}Starting evaluation...${NC}"
echo "Command: $CMD"
echo ""

eval $CMD