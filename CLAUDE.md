# Claude Code Instructions and Learnings

This file contains important instructions and learnings for Claude when working on this project.

## Research Journals

For detailed experiment results and learnings, see the dedicated research journals:
- **[research_journal/GRPO_baseline.md](./research_journal/GRPO_baseline.md)** - GRPO arithmetic experiments
- **[research_journal/gr00t_baseline.md](./research_journal/gr00t_baseline.md)** - GR00T robot model experiments  
- **[research_journal/absolute_zero.md](./research_journal/absolute_zero.md)** - Absolute Zero self-play experiments

## File Management

### Edit Existing Files Instead of Creating New Ones
- **ALWAYS prefer editing existing files** rather than creating new files for small changes
- Only create new files when there are major changes or truly new functionality
- This keeps the codebase cleaner and makes it easier to track changes
- Example: If modifying a training script with a few parameter changes, edit the existing script rather than creating a new variant

## Dependency Management

### Use `uv` and `pyproject.toml`
- **ALWAYS use `uv`** for dependency management and installs instead of pip
- **ALWAYS use `pyproject.toml`** for all dependency specifications
- **NEVER use `pip install` directly** - always use `uv pip install`
- **NEVER install packages individually** - always add to pyproject.toml first
- Example commands:
  ```bash
  uv pip install -e .        # Install current project in editable mode
  uv pip sync               # Sync dependencies from pyproject.toml
  uv pip compile pyproject.toml -o requirements.txt  # Generate requirements.txt if needed
  ```
- When adding new dependencies:
  1. Add them to `pyproject.toml` under `[project] dependencies`
  2. Run `uv pip sync` or `uv pip install -e .`
  3. Never use `uv pip install <package>` directly

### Running Python Scripts with `uv`
- **ALWAYS use `uv run`** to execute Python scripts
- This ensures the correct virtual environment and dependencies are used
- Example commands:
  ```bash
  uv run python train_script.py     # Run a Python script
  uv run pytest tests/              # Run tests
  uv run python -m module_name      # Run a module
  ```
- **NEVER use `python` directly** - always use `uv run python`

## Remote Machine Operations

### Use tmux Sessions
When working on remote machines (e.g., SSH to H100 machines), always use tmux sessions so the user can:
- Attach to the same session to monitor progress
- Keep processes running after SSH disconnection
- Share the session context

Example workflow:
```bash
# Create or attach to a named tmux session
ssh ubuntu@192.222.52.59 "tmux new-session -d -s training || tmux attach -t training"

# Run commands in the tmux session
ssh ubuntu@192.222.52.59 "tmux send-keys -t training 'cd ~/pippa && python train_grpo.py' Enter"

# User can attach to monitor
ssh ubuntu@192.222.52.59 -t "tmux attach -t training"
```

## Environment Variables

### WandB Configuration
The project uses environment variables from `.env` file:
- `WANDB_ENTITY=wild-ai`
- `WANDB_PROJECT=pippa`
- `WANDB_API_KEY=<secret>`

These are automatically loaded using `python-dotenv` in the training scripts.

## Git Best Practices

### Never Commit Secrets
- `.env` file is in `.gitignore` and should never be committed
- Use `.env.example` to show required environment variables without values
- GitHub PATs and API keys should only be used temporarily and never stored in code

## Testing and Validation

### Linting and Type Checking
When completing a task, run:
```bash
npm run lint      # If available
npm run typecheck # If available
ruff check .      # For Python projects
mypy .           # For Python type checking
```

If these commands aren't found, ask the user for the correct commands and suggest adding them to this file.

## Research Journal Guidelines

### Timestamp All Experiments
When writing research notes or documenting experiments in research journals:
- **ALWAYS include the full datetime (YYYY-MM-DD_HH:MM)** in experiment titles
- **ALWAYS include the WandB run ID** for tracking
- This helps track the chronological order of experiments
- Format: "Experiment Name - YYYY-MM-DD_HH:MM - Run ID: xxxxxxxx"
- Example: "### 1. Batch Size Experiment (GPU 1) - 2025-06-15_01:22 - Run ID: abc123de"

This ensures we can reconstruct the timeline of experiments and cross-reference with WandB.

## SSH and GPU Usage

### H100 Machine Access
Current H100 machines:
- `ubuntu@192.222.52.59` (Original)
- `ubuntu@192.222.53.15` (New - for GR00T training)

Always:
1. Check GPU availability with `nvidia-smi`
2. Set proper environment variables (e.g., `HF_HOME`)
3. Use tmux for long-running processes
4. Pull latest changes before running

## IMPORTANT: Two Parallel Experiments in This Repo

This repository contains TWO SEPARATE experiment tracks running in parallel:

### 1. GRPO Experiments (Main Directory)
- **What**: Group Relative Policy Optimization for language models
- **Where**: Main directory (`train_grpo*.py` files)
- **Goal**: Achieve verifiable rewards with arithmetic, counting, comparison tasks
- **Runner**: Another team member
- **Key files**: 
  - `train_grpo.py`, `train_grpo_verifiable.py`
  - `experiments/` directory
- **WandB tags**: "grpo-setup", "verifiable-rewards", etc.

### 2. GR00T Robot Model Fine-tuning (gr00t-tuning/)
- **What**: NVIDIA Isaac GR00T N1.5 humanoid robot foundation model
- **Where**: `gr00t-tuning/` subdirectory
- **Goal**: Fine-tune robot model on SO-101 demonstration data
- **Runner**: Claude (this assistant)
- **Key files**:
  - `gr00t-tuning/train_gr00t.py`
  - `gr00t-tuning/setup.sh`
- **WandB tags**: "gr00t-overfit", "gr00t-training"
- **Based on**: https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning

⚠️ **DO NOT CONFUSE THESE TWO EXPERIMENTS!**
- GRPO = Language model RL training with rewards
- GR00T = Robot foundation model fine-tuning on demonstrations

## Project Summary: GRPO Training Pipeline

### Overview
This project implements Dr GRPO (Group Relative Policy Optimization Done Right) for training language models with reinforcement learning. The implementation follows best practices from cleanRL and stable-baselines3.

### Key Components

#### 1. Training Scripts
- **`train_grpo.py`**: Main Dr GRPO implementation
  - Bias-free advantage computation (key Dr GRPO insight)
  - Beta = 0.0 (no KL penalty) for rule-based rewards
  - Support for multiple reward functions
  - Proper seed management for reproducibility
  - WandB integration when `track=True`

- **`train_grpo_wandb.py`**: Simple wrapper that enables WandB tracking

- **`train_grpo_verifiable.py`**: GRPO with verifiable rewards
  - Arithmetic, counting, comparison, binary tasks
  - Clear success/failure criteria

#### 2. Helper Scripts
- **`remote_train.sh`**: Easy remote training with tmux

### Essential Commands

#### SSH and tmux
```bash
# SSH to H100 machine
ssh ubuntu@192.222.52.59

# Create/attach to tmux sessions
ssh ubuntu@192.222.52.59 "tmux new-session -d -s grpo_training"

# Attach to monitor training
ssh ubuntu@192.222.52.59 -t "tmux attach -t grpo_training"

# Send commands to tmux
ssh ubuntu@192.222.52.59 "tmux send-keys -t grpo_training 'cd ~/pippa && git pull' Enter"
```

#### Environment Setup (in tmux)
```bash
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/home/ubuntu/.cache/huggingface
cd ~/pippa
git pull
```

#### Running Training
```bash
# Basic GRPO training with WandB
python train_grpo_wandb.py

# Verifiable rewards training
python train_grpo_verifiable.py --task arithmetic
```

#### Monitoring
```bash
# Check GPU utilization
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv

# Check tmux output
tmux capture-pane -t grpo_training -p | tail -50
```

### Configuration Details

#### Default Model and Dataset
- Model: `Qwen/Qwen2-0.5B-Instruct` (0.5B parameters)
- Temperature: 0.7 (critical for generation diversity)
- Beta: 0.0 for rule-based rewards, 0.1 for arithmetic tasks

### WandB Integration
- Entity: `wild-ai`
- Project: `pippa`
- Credentials in `.env` file (never commit!)
- Automatic initialization when `track=True`
- View runs at: https://wandb.ai/wild-ai/pippa

### Repository Structure
```
pippa/
├── train_grpo.py                 # Main Dr GRPO implementation
├── train_grpo_wandb.py          # WandB wrapper  
├── train_grpo_verifiable.py     # GRPO with verifiable rewards
├── train_grpo_arithmetic_fixed_completions.py  # WORKING BASELINE for arithmetic
├── train_grpo_with_standard_eval.py  # NEW TEMPLATE with standardized evaluation
├── grpo_archive/               # Archived GRPO experiments
├── research_journal/           # Experiment notes and results
│   ├── GRPO_baseline.md       # GRPO arithmetic experiments journal
│   ├── gr00t_baseline.md      # GR00T robot model experiments journal  
│   └── absolute_zero.md       # Absolute Zero self-play experiments
├── TRAINING_GUIDE.md            # Comprehensive training guide
├── experiments/
│   ├── README.md               # Experiments overview
│   ├── overfitting_experiments_log.md  # Detailed experiment results
│   └── failed_approaches/      # Failed experiments (echo tasks, etc.)
├── arithmetic_eval_dataset/    # Local copy of standardized evaluation dataset
├── run_verifiable_experiments.sh # Run all verifiable experiments
├── gr00t-tuning/               # GR00T robot foundation model tuning (separate project)
├── absolute_zero/              # Absolute Zero self-play implementation
├── gr00t-rl/                   # Gymnasium robotics RL experiments
├── remote_train.sh              # Remote training helper
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (gitignored)
├── .env.example               # Example env file
└── CLAUDE.md                  # This file (you are here)
```

### Codebase Organization Notes (2025-06-15)
The codebase has been cleaned up for clarity:
- **Core GRPO scripts** remain in the root directory (4 essential files)
- **Experimental/debug scripts** moved to `grpo_archive/` directory
- **Working baseline** (`train_grpo_arithmetic_fixed_completions.py`) kept in root
- **Failed experiments** remain in `experiments/failed_approaches/`
- **GR00T experiments** isolated in `gr00t-tuning/` subdirectory

This organization makes it clear which scripts are actively used vs historical experiments.

### Key Documentation
- **[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)** - Step-by-step guide for using the training scripts
- **[experiments/README.md](./experiments/README.md)** - Overview of all experiments and learnings
- **[research_journal/GRPO_baseline.md](./research_journal/GRPO_baseline.md)** - GRPO arithmetic experiment results and learnings
- **[research_journal/gr00t_baseline.md](./research_journal/gr00t_baseline.md)** - GR00T robot model experiment results

## Key Technical Learnings

### GRPO Requirements
1. **Verifiable Rewards**: GRPO requires deterministic, measurable outcomes (e.g., math answers, not free-form text)
2. **Latent Understanding**: Model must already have capability - GRPO elicits, doesn't teach
3. **Generation Diversity**: Temperature 0.7-1.0 essential for reward variance
4. **KL Penalty for Arithmetic**: Use beta=0.1 for arithmetic tasks to prevent mode collapse

### Common Issues and Solutions
1. **NumPy version conflict**: Install `numpy<2`
2. **Keras error**: Install `tf-keras`
3. **GSM8K loading**: Specify 'main' config: `load_dataset('openai/gsm8k', 'main')`
4. **Low GPU usage**: Increase batch size and sequence lengths
5. **WandB not logging**: Ensure `.env` file exists and `track=True`
6. **DataLoader error with GRPO**: Batch size must be <= dataset size
7. **Zero gradients in GRPO**: All rewards identical - no learning signal
8. **Stuck at reward -1.0**: Model doesn't understand task format

### GRPOTrainer Configuration Guide

#### Critical Parameters
- **`num_generations`**: Number of completions per prompt (default: 8, use 16 for better diversity)
- **`beta`**: KL divergence coefficient (0.0 for rule-based rewards, 0.1 for arithmetic)
- **`temperature`**: Controls generation randomness (0.7-1.0 for healthy diversity)
- **`loss_type`**: Use "grpo" for arithmetic, "dr_grpo" for bias-free training

#### Batch Size Configuration
- **IMPORTANT**: `per_device_train_batch_size` in GRPOConfig is the ACTUAL batch size
- The effective batch size MUST be divisible by `num_generations`
- Example: If batch_size=64 and num_generations=16, this works (64 ÷ 16 = 4)

### Working Baseline Reference
For a proven working configuration, see:
- **File**: `train_grpo_arithmetic_fixed_completions.py`
- **Achievement**: ~75% accuracy on arithmetic with rich rewards
- **Key insight**: Rich reward functions (partial credit) dramatically improve learning

## WandB Monitoring

### Use MCP Tool for WandB Monitoring
**IMPORTANT**: Always use the wandb MCP tool to check training progress instead of SSH/tmux commands. This is the default and preferred method.

### WandB Video Logging for Gymnasium Environments
When using Gymnasium environments (like Fetch robotics tasks), ensure proper video logging to WandB:

1. **Basic Setup**: Add `monitor_gym=True` to `wandb.init()`:
```python
wandb.init(
    project="pippa",
    entity="wild-ai",
    monitor_gym=True,  # Enables automatic Gymnasium video logging
    tags=["gr00t-rl", "ppo", "fetch"],
)
```

2. **Video Tables with Global Step**: Log videos to WandB tables for better organization:
```python
# Create table with INCREMENTAL mode for ongoing updates
video_table = wandb.Table(
    columns=["global_step", "episode", "video", "episode_return", "episode_length"],
    log_mode="INCREMENTAL"
)

# Add video data and log immediately
video_table.add_data(
    global_step,
    episode_num,
    wandb.Video(video_path, fps=30, format="mp4"),
    episode_return,
    episode_length
)
wandb.log({"video_table": video_table}, step=global_step)
```

3. **Critical: Always Call wandb.finish()**
Always ensure `wandb.finish()` is called when training stops to upload table data! Use try-finally blocks:
```python
try:
    # Training code here
    if args.track and WANDB_AVAILABLE:
        wandb_run = wandb.init(...)
    # ... training ...
finally:
    # Always clean up resources
    if args.track and WANDB_AVAILABLE and wandb_run is not None:
        wandb.finish()
```

### WandB Logging Best Practices

#### Enable Logging at Every Step
**CRITICAL**: Always set `logging_steps=1` in your training configuration to ensure logging happens at every training step.

#### Essential Metrics to Track
- `train/reward`: Primary optimization target
- `train/loss`: Training loss (watch for collapse to 0)
- `train/grad_norm`: Gradient norms (should be non-zero)
- `train/kl`: KL divergence from reference model
- `train/frac_reward_zero_std`: Fraction of batches with zero reward std

### Standardized Evaluation
All GRPO experiments should use `morgan/arithmetic_eval` dataset for final evaluation:
```python
from datasets import load_dataset
eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
```

## Gymnasium-Robotics Integration

### Training Scripts
Key scripts for robotics training:
- `gr00t-rl/scripts/train_ppo_fetch.py` - PPO training on Fetch tasks
- `gr00t-rl/scripts/train_grpo_fetch.py` - GRPO training on Fetch tasks
- `gr00t-rl/scripts/train_ppo_gr00t.py` - PPO with GR00T N1.5 model

Default configuration:
- WandB tracking enabled by default (`--track=True`)
- Video capture enabled by default (`--capture-video=True`)
- Support for sparse/dense/distance reward modes

### GR00T N1.5 Integration
GR00T N1.5 is NVIDIA's 3B parameter robot foundation model. We've integrated it for RL training:

**Quick Test (Lite Model)**:
```bash
python scripts/train_ppo_gr00t.py --use-groot-lite True --num-envs 4 --reward-mode dense
```

**Full GR00T Model**:
```bash
# First install Isaac-GR00T
./scripts/setup_isaac_groot.sh

# Then train with full model
python scripts/train_ppo_gr00t.py \
    --use-groot-lite False \
    --groot-model-path nvidia/GR00T-N1.5-3B \
    --freeze-vision True \
    --freeze-language True
```

## Absolute Zero Experiment
For the Absolute Zero self-play algorithm implementation, see:
- **[research_journal/absolute_zero.md](./research_journal/absolute_zero.md)** - Experiment details and results
- **Directory**: `absolute_zero/` - Implementation code
- **Based on**: https://arxiv.org/pdf/2505.03335v2