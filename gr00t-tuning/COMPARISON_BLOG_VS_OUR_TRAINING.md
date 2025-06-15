# Detailed Comparison: Blog Post vs Our Training Scripts

## Overview
This document provides a detailed comparison between the official GR00T fine-tuning approach described in the HuggingFace blog post and our custom training scripts.

## Key Differences Summary

### 1. Dataset Used
- **Blog Post**: Uses `so101-table-cleanup` dataset downloaded from HuggingFace
  ```bash
  huggingface-cli download \
      --repo-type dataset youliangtan/so101-table-cleanup \
      --local-dir ./demo_data/so101-table-cleanup
  ```
- **Our Scripts**: Use built-in demo data `robot_sim.PickNPlace` (5 episodes only)
  - Path: `Isaac-GR00T/demo_data/robot_sim.PickNPlace`
  - Much smaller dataset for overfitting experiments

### 2. Data Configuration
- **Blog Post**: `--data-config so100_dualcam` (expects dual camera setup)
- **Our Scripts**: `--data-config fourier_gr1_arms_only` 
  - Changed because demo data only has `ego_view` camera
  - Blog's config expects `front` camera which doesn't exist in demo data

### 3. Training Steps
- **Blog Post**: `--max-steps 10000` (full fine-tuning)
- **Our Demo Script**: `--max-steps 200` (quick overfitting test)
- **Our Long Script**: `--max-steps 2000` (extended overfitting)

### 4. Learning Rate
- **Blog Post**: Uses default `1e-4`
- **Our Demo Script**: `--learning-rate 5e-4` (5x higher)
- **Our Long Script**: `--learning-rate 1e-3` (10x higher for aggressive overfitting)

### 5. Batch Size
- **Blog Post**: Uses default `32`
- **Our Scripts**: `--batch-size 1` (minimal for overfitting on small dataset)

### 6. Fine-tuning Components
- **Blog Post**: Not specified (uses defaults: tune_projector=True, tune_diffusion_model=True)
- **Our Demo Script**: 
  ```bash
  --no-tune-llm        # Don't fine-tune language model
  --no-tune-visual     # Don't fine-tune vision tower
  --tune-projector     # Fine-tune projector
  --tune-diffusion-model # Fine-tune diffusion model
  ```
- **Our Long Script**: Fine-tunes ALL components for maximum overfitting

### 7. Boolean Arguments Format
- **Blog Post**: Doesn't show boolean flags (uses defaults)
- **Our Scripts**: Had to use proper tyro format:
  - WRONG: `--tune-llm false`
  - CORRECT: `--no-tune-llm` or `--tune-llm`

### 8. Additional Parameters We Added
- `--warmup-ratio 0.01` or `0.0` (minimal/no warmup)
- `--weight-decay 0.0` (no regularization for overfitting)
- `--dataloader-num-workers 4` or `2` (reduced for small dataset)

### 9. Environment Setup
- **Blog Post**: Uses conda
  ```bash
  conda create -n gr00t python=3.10
  conda activate gr00t
  pip install --upgrade setuptools
  pip install -e .[base]
  ```
- **Our Scripts**: Use uv for consistency
  ```bash
  uv venv
  source .venv/bin/activate
  uv pip install -e ".[base]"
  ```

### 10. WandB Integration
- **Blog Post**: `--report-to wandb` (simple flag)
- **Our Scripts**: Additional WandB setup:
  ```python
  os.environ["WANDB_ENTITY"] = "wild-ai"
  os.environ["WANDB_PROJECT"] = "pippa"
  os.environ["WANDB_RUN_NAME"] = run_name
  os.environ["WANDB_TAGS"] = "gr00t-overfit,demo-data,pick-place"
  ```

## Complete Command Comparison

### Blog Post Command:
```bash
python scripts/gr00t_finetune.py \
   --dataset-path ./demo_data/so101-table-cleanup/ \
   --num-gpus 1 \
   --output-dir ./so101-checkpoints  \
   --max-steps 10000 \
   --data-config so100_dualcam \
   --video-backend torchvision_av
```

### Our Demo Script Command:
```bash
python Isaac-GR00T/scripts/gr00t_finetune.py \
    --dataset-path Isaac-GR00T/demo_data/robot_sim.PickNPlace \
    --output-dir gr00t-checkpoints-overfit-demo \
    --data-config fourier_gr1_arms_only \
    --video-backend torchvision_av \
    --num-gpus 1 \
    --batch-size 1 \
    --max-steps 200 \
    --learning-rate 5e-4 \
    --save-steps 50 \
    --report-to wandb \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --warmup-ratio 0.01 \
    --weight-decay 0.0 \
    --dataloader-num-workers 4
```

### Our Long Script Command:
```bash
python Isaac-GR00T/scripts/gr00t_finetune.py \
    --dataset-path Isaac-GR00T/demo_data/robot_sim.PickNPlace \
    --output-dir gr00t-checkpoints-overfit-long \
    --data-config fourier_gr1_arms_only \
    --video-backend torchvision_av \
    --num-gpus 1 \
    --batch-size 1 \
    --max-steps 2000 \
    --learning-rate 1e-3 \
    --save-steps 100 \
    --report-to wandb \
    --tune-llm \
    --tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --warmup-ratio 0.0 \
    --weight-decay 0.0 \
    --dataloader-num-workers 2
```

## Why These Changes Were Necessary

1. **Dataset Size**: Demo data has only 5 episodes vs full dataset's hundreds/thousands
   - Required smaller batch size and higher learning rate

2. **Camera Configuration**: Demo data structure differs from SO-101 standard
   - Had to switch to compatible data config

3. **Overfitting Goal**: Blog aims for generalization, we aim for overfitting
   - Removed regularization, increased learning rate, reduced dataset

4. **Resource Constraints**: Testing pipeline vs production training
   - Shorter training runs, more frequent checkpoints

5. **Dependency Management**: Project uses uv consistently
   - Adapted installation to use uv instead of pip

## Default Parameters (from gr00t_finetune.py)
These are used when not specified:
- `batch_size`: 32
- `learning_rate`: 1e-4
- `weight_decay`: 1e-5
- `warmup_ratio`: 0.05
- `save_steps`: 1000
- `dataloader_num_workers`: 8
- `tune_llm`: False
- `tune_visual`: False
- `tune_projector`: True
- `tune_diffusion_model`: True