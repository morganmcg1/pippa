# GR00T N1.5 SO101 Fine-tuning

This project implements fine-tuning for NVIDIA's GR00T N1.5 foundation model for humanoid robots, based on the [official tutorial](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning).

## Overview

GR00T N1.5 is the first open foundation model for generalized humanoid robot reasoning and skills. This project provides scripts to:
- Fine-tune the model on custom datasets
- Evaluate model performance
- Deploy to physical robots

## Prerequisites

- Python 3.10+
- CUDA-capable GPU
- `uv` package manager

## Setup

1. Run the setup script to clone Isaac-GR00T and install dependencies:
   ```bash
   cd gr00t-tuning
   ./setup.sh
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Dataset Preparation

Your dataset should follow the SO101 format with:
- A `modality.json` configuration file
- Video files for dual camera views
- Language instructions
- Action sequences

Example dataset structure:
```
dataset/
├── modality.json
├── episode_0001/
│   ├── camera_0/
│   ├── camera_1/
│   ├── language.json
│   └── actions.npy
└── ...
```

## Training

Run fine-tuning with:
```bash
python train_gr00t.py \
    --dataset-path ./data/so101-table-cleanup \
    --num-gpus 1 \
    --output-dir ./checkpoints \
    --max-steps 10000
```

### Advanced Options

```bash
python train_gr00t.py \
    --dataset-path ./data/my-dataset \
    --num-gpus 2 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --warmup-steps 1000 \
    --max-steps 20000 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --wandb-project gr00t-tuning
```

## Evaluation

### Open-loop Evaluation
```bash
python evaluate_gr00t.py \
    --checkpoint-path ./checkpoints/checkpoint-10000 \
    --dataset-path ./data/test-dataset \
    --evaluation-mode open-loop
```

### Deployment Evaluation
```bash
python evaluate_gr00t.py \
    --checkpoint-path ./checkpoints/checkpoint-10000 \
    --dataset-path ./data/test-dataset \
    --evaluation-mode deployment \
    --robot-config my_robot.yaml \
    --simulation-mode
```

### Error Analysis
```bash
python evaluate_gr00t.py \
    --checkpoint-path ./checkpoints/checkpoint-10000 \
    --dataset-path ./data/test-dataset \
    --analyze-errors
```

## Environment Variables

Create a `.env` file for Weights & Biases tracking:
```
WANDB_ENTITY=your-entity
WANDB_PROJECT=gr00t-tuning
WANDB_API_KEY=your-api-key
```

## Project Structure

```
gr00t-tuning/
├── pyproject.toml          # Dependencies managed by uv
├── setup.sh               # Setup script
├── train_gr00t.py         # Main training script
├── evaluate_gr00t.py      # Evaluation script
├── README.md              # This file
├── Isaac-GR00T/           # Cloned NVIDIA repository
├── data/                  # Dataset directory
├── checkpoints/           # Model checkpoints
└── logs/                  # Training logs
```

## Tips

1. **GPU Memory**: Adjust batch size based on your GPU memory
2. **Data Config**: Use `so100_dualcam` for dual camera setups
3. **Video Backend**: Use `torchvision_av` for better compatibility
4. **Checkpointing**: Save frequently with `--save-interval` for long runs

## Troubleshooting

- **Flash Attention Error**: Ensure CUDA toolkit is properly installed
- **Dataset Format**: Verify `modality.json` matches your data structure
- **Memory Issues**: Reduce batch size or use gradient accumulation

## References

- [Original Tutorial](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [Isaac-GR00T Repository](https://github.com/NVIDIA-Isaac-GR00T/Isaac-GR00T)