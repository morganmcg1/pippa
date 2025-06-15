# Claude Code Instructions and Learnings

This file contains important instructions and learnings for Claude when working on this project.

## Research Journals
For experiment-specific results and detailed analyses, see:
- [GRPO Experiments](./research_journal/GRPO_baseline.md) - Arithmetic GRPO training results and learnings
- [GR00T Experiments](./research_journal/gr00t_baseline.md) - Robot model fine-tuning experiments
- [Absolute Zero](./research_journal/absolute_zero.md) - Self-play learning implementation

## File Management

### Edit Existing Files Instead of Creating New Ones
- **ALWAYS prefer editing existing files** rather than creating new files for small changes
- Only create new files when there are major changes or truly new functionality
- This keeps the codebase cleaner and makes it easier to track changes
- Example: If modifying a training script with a few parameter changes, edit the existing script rather than creating a new variant

### Reuse Training Scripts with Arguments
- **ALWAYS reuse existing training scripts** by passing different arguments instead of creating new scripts
- Add command-line arguments to scripts to make them configurable
- Only create new scripts when the logic is fundamentally different
- Example: For testing different batch sizes, use `--batch-size` argument rather than creating `train_batch_size_32.py`

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

## Training Scripts Overview

### GRPO Training
- **`train_grpo.py`**: Main Dr GRPO implementation with verifiable rewards
- **`train_grpo_wandb.py`**: Simple wrapper that enables WandB tracking
- **`train_grpo_with_standard_eval.py`**: Template with standardized evaluation
- **`train_grpo_arithmetic_fixed_completions.py`**: Working baseline for arithmetic

### GR00T Training
- **`gr00t-tuning/train_gr00t.py`**: NVIDIA GR00T robot model fine-tuning
- **`gr00t-rl/scripts/train_ppo_fetch.py`**: PPO training on Fetch robotics tasks

## Repository Structure
```
pippa/
├── train_grpo*.py              # GRPO training scripts
├── grpo_archive/               # Archived GRPO experiments
├── research_journal/           # Experiment notes and results
│   ├── GRPO_baseline.md       # GRPO arithmetic experiments
│   ├── gr00t_baseline.md      # GR00T robot experiments  
│   └── absolute_zero.md       # Self-play learning
├── gr00t-tuning/               # GR00T robot model tuning
├── gr00t-rl/                   # Gymnasium-robotics integration
├── absolute_zero/              # Absolute Zero implementation
├── experiments/                # Experiment archives
├── TRAINING_GUIDE.md           # Training guide
├── .env.example                # Example environment variables
└── CLAUDE.md                   # This file
```

## Common Issues and Solutions
1. **NumPy version conflict**: Install `numpy<2`
2. **Keras error**: Install `tf-keras`
3. **GSM8K loading**: Specify 'main' config
4. **Low GPU usage**: Increase batch size and sequence lengths
5. **WandB not logging**: Ensure `.env` file exists and `track=True`
6. **DataLoader error with GRPO**: Batch size must be <= dataset size
7. **Zero gradients in GRPO**: All rewards identical - no learning signal
8. **Stuck at reward -1.0**: Model doesn't understand task format

## GRPOTrainer Configuration Guide

### Key Parameters (from HuggingFace docs: https://huggingface.co/docs/trl/main/en/grpo_trainer)

#### Generation Settings
- **`num_generations`**: Number of completions per prompt (default: 8)
  - Increase for more diverse samples and better gradient estimates
  - Trade-off with memory usage
- **`max_prompt_length`**: Maximum prompt length (default: 512)
- **`max_completion_length`**: Maximum completion length (default: 256)
  - Shorter completions train faster
- **`temperature`**: Controls generation randomness (default: 1.0)
  - Lower values for more focused outputs

#### Training Hyperparameters
- **`beta`**: KL divergence coefficient (default: 0.0)
  - Dr GRPO uses 0.0 to avoid bias
- **`epsilon`**: Clipping value for policy updates (default: 0.2)
- **`num_iterations`**: Optimization iterations per batch (default: 1)
- **`loss_type`**: Can be "bnpo" or "dr_grpo"
  - "dr_grpo" recommended for bias-free training

#### Debugging and Monitoring
- **`log_completions`**: Set to True to inspect generation quality
- **`mask_truncated_completions`**: Use True for training stability
- Monitor: `reward/mean`, `completions/mean_length`, `frac_reward_zero_std`

### Advanced GRPO Configuration

#### Critical Constraints:
- **Batch Size Rule**: `effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus`
- **Must satisfy**: `effective_batch_size % num_generations == 0`
- Example: batch_size=16, num_generations must be in [1, 2, 4, 8, 16]

**IMPORTANT GRPO Batch Size Configuration**:
- When using GRPOTrainer, the `per_device_train_batch_size` parameter in GRPOConfig is the ACTUAL batch size, not batch_size // num_generations
- The effective batch size MUST be divisible by `num_generations`
- If you get an error like "effective train batch size (64) must be evenly divisible by num_generations (16)", you need to either:
  1. Reduce `num_generations` to a divisor of your batch size (e.g., 4 instead of 16)
  2. Increase your batch size to be a multiple of `num_generations`

## WandB Monitoring

### Use MCP Tool for WandB Monitoring
**IMPORTANT**: Always use the wandb MCP tool to check training progress instead of SSH/tmux commands. This is the default and preferred method.

### WandB Video Logging for Gymnasium Environments
When using Gymnasium environments (like Fetch robotics tasks):

1. **Basic Setup**: Add `monitor_gym=True` to `wandb.init()`
2. **Video Recording**: Use `gym.wrappers.RecordVideo` with episode triggers
3. **Video Tables**: Use INCREMENTAL mode for ongoing updates
4. **Headless Servers**: Set `os.environ['MUJOCO_GL'] = 'osmesa'`
5. **Always Call `wandb.finish()`**: Use try-finally blocks to ensure table data uploads

Reference implementation: `gr00t-rl/scripts/train_ppo_fetch_with_video_table.py`

### WandB Logging Best Practices

#### Enable Logging at Every Step
**CRITICAL**: Always set `logging_steps=1` in your training configuration to ensure logging happens at every training step. This provides:
- Real-time monitoring of training progress
- Early detection of training issues (zero gradients, reward collapse)
- Detailed learning curves for analysis
- No missing data points during critical training phases

Example configuration:
```python
config = GRPOConfig(
    logging_steps=1,  # Log at every step - CRITICAL!
    save_steps=100,   # Save checkpoints less frequently
    # ... other parameters
)
```

### WandB Tables for GRPO
Log generation samples to WandB Tables for better visualization:
- Create tables with columns: epoch, prompt, completion, reward, etc.
- Use TRL callbacks (see `train_grpo_verifiable_callbacks.py`)
- Monitor both training reward and evaluation accuracy

### Gymnasium-Robotics and GR00T Integration

#### Key Scripts
- `gr00t-rl/scripts/train_ppo_fetch.py` - PPO training on Fetch tasks
- `gr00t-rl/scripts/train_grpo_fetch.py` - GRPO training on Fetch tasks
- `gr00t-rl/scripts/train_ppo_gr00t.py` - PPO with GR00T N1.5 model
- `gr00t-rl/environments/fetch_wrapper.py` - Goal-conditioned wrapper

#### GR00T N1.5 Components
- **GR00T Policy Wrapper**: Handles actor-critic interface for PPO/GRPO
- **Model**: 3B parameters, Eagle vision-language backbone
- **Setup**: Use `./scripts/setup_isaac_groot.sh` for installation

#### Video Table Logging
- Use INCREMENTAL mode for ongoing updates
- Log tables immediately after adding data
- Always use try-finally blocks with `wandb.finish()`
- Record every 5 episodes for balanced video generation