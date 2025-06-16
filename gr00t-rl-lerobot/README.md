# GR00T-RL-LeRobot Integration

This directory contains the integration of our fine-tuned GR00T-N1.5-3B model with LeRobot for reinforcement learning on SO-101 robotic tasks.

## Directory Structure

```
gr00t-rl-lerobot/
├── environments/       # SO-101 simulation environments
├── policies/          # GR00T policy wrapper for LeRobot
├── scripts/           # Training and evaluation scripts
├── configs/           # Configuration files
└── utils/             # Utility functions
```

## Overview

This project integrates:
- **Fine-tuned GR00T Model**: Vision-language-action model trained on SO-101 table cleanup
- **LeRobot Framework**: HuggingFace's robotics learning framework
- **RL Algorithm**: SAC (Soft Actor-Critic) for online learning
- **Task**: Table manipulation with SO-101 6-DoF arm

## Model Details

- **Checkpoint**: `wild-ai/pippa/gr00t-sft-so100_dualcam-bs32:v0` (WandB)
- **Architecture**: GR00T-N1.5-3B with diffusion action head
- **Performance**: MSE 11.15 on table cleanup evaluation

## Installation

```bash
# Install LeRobot
pip install lerobot

# Install additional dependencies
pip install gymnasium mujoco-py torch wandb
```

## Usage

Coming soon - implementation in progress.

## References

- [Research Journal](../research_journal/gr00t-rl-lerobot.md)
- [LeRobot Docs](https://huggingface.co/docs/lerobot)
- [GR00T SFT Training](../research_journal/gr00t_sft.md)