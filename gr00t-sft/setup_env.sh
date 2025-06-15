#!/bin/bash
# Setup environment variables for GR00T SFT training

# Load environment variables from .env file
export $(grep -v '^#' ~/pippa/.env | xargs)

# Export WandB variables
export WANDB_MODE=online
export WANDB_PROJECT=pippa
export WANDB_ENTITY=wild-ai

echo "Environment variables set:"
echo "  WANDB_ENTITY: $WANDB_ENTITY"
echo "  WANDB_PROJECT: $WANDB_PROJECT"
echo "  WANDB_MODE: $WANDB_MODE"
echo "  WANDB_API_KEY: ${WANDB_API_KEY:0:10}..."