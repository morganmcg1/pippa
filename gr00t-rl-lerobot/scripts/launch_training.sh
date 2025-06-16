#!/bin/bash
# Launch script for GR00T RL training with different environment configurations

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}GR00T RL Training Launcher${NC}"
echo "=============================="

# Parse arguments
ENV_TYPE=${1:-cartesian}
USE_JOINT_SPACE=${2:-false}

echo -e "${YELLOW}Configuration:${NC}"
echo "  Environment type: $ENV_TYPE"
echo "  Use joint space: $USE_JOINT_SPACE"
echo ""

# Validate environment type
if [[ "$ENV_TYPE" != "cartesian" && "$ENV_TYPE" != "coupled" ]]; then
    echo -e "${YELLOW}Error: Invalid environment type. Use 'cartesian' or 'coupled'${NC}"
    exit 1
fi

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
export MUJOCO_GL=egl  # For headless rendering

# Check if Isaac-GR00T exists
ISAAC_GROOT_PATH="$HOME/pippa/Isaac-GR00T"
if [ ! -d "$ISAAC_GROOT_PATH" ]; then
    echo -e "${YELLOW}Warning: Isaac-GR00T not found at $ISAAC_GROOT_PATH${NC}"
    echo "The GR00T policy requires Isaac-GR00T to be installed."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create directories
mkdir -p checkpoints
mkdir -p logs

# Launch training based on configuration
if [ "$ENV_TYPE" = "cartesian" ]; then
    echo -e "${GREEN}Launching Cartesian-only training...${NC}"
    echo "This uses 4D Cartesian actions with Fetch's 7-DoF hidden"
    echo ""
    
    python train_sac_fetch.py \
        --env-type cartesian \
        2>&1 | tee logs/training_cartesian_$(date +%Y%m%d_%H%M%S).log
        
elif [ "$ENV_TYPE" = "coupled" ]; then
    if [ "$USE_JOINT_SPACE" = "true" ]; then
        echo -e "${GREEN}Launching Coupled environment with joint-space actions...${NC}"
        echo "This simulates 6-DoF with joint-level control"
        echo ""
        
        python train_sac_fetch.py \
            --env-type coupled \
            --use-joint-space \
            2>&1 | tee logs/training_coupled_joint_$(date +%Y%m%d_%H%M%S).log
    else
        echo -e "${GREEN}Launching Coupled environment with Cartesian actions...${NC}"
        echo "This simulates 6-DoF by coupling joints 1 & 3"
        echo ""
        
        python train_sac_fetch.py \
            --env-type coupled \
            2>&1 | tee logs/training_coupled_cartesian_$(date +%Y%m%d_%H%M%S).log
    fi
fi

echo ""
echo -e "${GREEN}Training launched!${NC}"
echo "Check WandB for real-time metrics: https://wandb.ai/wild-ai/pippa"
echo "Logs saved to: logs/"