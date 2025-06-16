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
- `ubuntu@192.222.52.59` (Original - for GRPO and Absolute Zero experiments)
- `ubuntu@192.222.53.15` (New - **ONLY for GR00T SFT experiments**)

**IMPORTANT**: GR00T SFT experiments MUST be run on `ubuntu@192.222.53.15` only. This server has:
- 4x H100 80GB GPUs
- Isaac-GR00T environment properly configured
- Required datasets at `~/pippa/datasets/`
- Virtual environment at `~/pippa/Isaac-GR00T/sft_venv`

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

### 3. Absolute Zero Self-Play (absolute_zero/)
- **What**: Self-play learning with proposer and solver models
- **Where**: `absolute_zero/` subdirectory
- **Goal**: Implement curriculum learning through self-generated problems
- **Server**: Use `ubuntu@192.222.52.59` (same as GRPO)
- **Key files**:
  - `absolute_zero/train_absolute_zero_unified.py`
- **WandB tags**: "absolute-zero", "self-play"
- **Based on**: https://arxiv.org/pdf/2505.03335v2

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

### WandB Artifact Naming and Aliases
When saving model checkpoints to WandB artifacts, use consistent naming with aliases for versioning:

**DO NOT** include step/epoch numbers in the artifact name:
```python
# BAD - creates many separate artifacts
artifact = wandb.Artifact(
    name=f"model-checkpoint-step-{step}",  # ❌ Don't do this
    type="model"
)
```

**DO** use a consistent name and use aliases for versioning:
```python
# GOOD - single artifact with multiple versions
artifact = wandb.Artifact(
    name="gr00t-sft-so100_dualcam-bs64",  # ✅ Consistent name
    type="gr00t-model",  # ✅ Use specific type for GR00T models
    description=f"GR00T SFT model - batch_size={batch_size}, dataset={dataset_name}"
)

# Log with step as alias
wandb.run.log_artifact(artifact, aliases=[f"step-{step}", "latest"])
```

**Note**: Use descriptive artifact types like `"gr00t-model"` instead of generic `"model"` to better organize different model types in WandB.

This creates a single artifact with multiple versions, each tagged with aliases like:
- `step-100`, `step-200`, etc. for specific checkpoints
- `latest` for the most recent version
- `best` or `final` for special checkpoints

Benefits:
- Cleaner artifact organization
- Easy to track model evolution
- Can reference specific versions by alias
- Reduces artifact clutter in WandB UI

### WandB Artifacts for Large Model Checkpoints

When uploading large model checkpoints (20GB+):

#### Best Practices:
1. **Upload only essential files** - Skip optimizer state unless resuming training
2. **Use `add_file()` instead of `add_dir()`** - More control over what's uploaded
3. **Add `skip_cache=True`** for large files to avoid copying to cache
4. **Use try-finally blocks** to ensure `wandb.finish()` is called

#### Example for Large Model Checkpoints:
```python
def save_checkpoint_to_wandb(checkpoint_dir, step, model=None, config=None):
    """Save only essential files for inference, skip optimizer state."""
    try:
        artifact = wandb.Artifact(
            name="model-name",
            type="model",
            description=f"Model checkpoint at step {step}"
        )
        
        # Add only essential files (skip optimizer.pt)
        essential_files = [
            "config.json",
            "model.safetensors",  # or model weights
            "tokenizer.json"      # if applicable
        ]
        
        for filename in essential_files:
            filepath = Path(checkpoint_dir) / filename
            if filepath.exists():
                artifact.add_file(str(filepath), skip_cache=True)
        
        wandb.run.log_artifact(artifact, aliases=[f"step-{step}", "latest"])
        
    except Exception as e:
        print(f"Error saving to WandB: {e}")
```

#### What NOT to Upload:
- `optimizer.pt` (8GB+) - Only needed for resuming training
- Full checkpoint directories with `add_dir()` - Too large, times out
- Duplicate files (e.g., saving model.pt when safetensors already exists)

#### Typical Checkpoint Sizes:
- Model weights: 7-10GB (necessary)
- Optimizer state: 8-15GB (skip unless resuming)
- Total with optimizer: 20-25GB per checkpoint
- Without optimizer: 7-10GB (much faster to upload)

### WandB Support Resources

**Always use the WandB support bot for WandB-specific questions**:
```python
# Use the MCP tool for WandB support
mcp__wandb__query_wandb_support_bot(
    question="Your question about WandB features, artifacts, etc."
)
```

The support bot can help with:
- Artifact upload strategies
- API usage and best practices
- Troubleshooting timeouts and errors
- Integration questions
- Performance optimization

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

## GR00T SFT Critical Learnings (2025-06-15)

### StopIteration Error Fix
**CRITICAL**: Always use `--no-tune-diffusion-model` flag to avoid StopIteration error:
```bash
python train_gr00t_sft.py \
   --dataset-path demo_data/so101-table-cleanup \
   --max-steps 10000 \
   --batch-size 4 \
   --no-tune-diffusion-model  # CRITICAL: Prevents StopIteration error
```

### Working Configuration
```python
# These settings MUST be used to avoid errors
tune_llm = False
tune_visual = False  
tune_projector = True
tune_diffusion_model = False  # MUST be False
```

### WandB Artifact Best Practices for GR00T
1. **Artifact naming**: Use consistent names like `gr00t-sft-so100_dualcam-bs{batch_size}`
2. **Aliases**: Include `step-{step}`, `latest`, and `run-{run_id}`
3. **Type**: Use `"gr00t-model"` not generic `"model"`
4. **Size optimization**: Skip optimizer.pt (saves 8GB)
5. **Contents**: Include model weights, action_head_new_embodiment.pt, modality.json, and experiment_cfg/

### Successful Training Parameters
- **Overfitting test**: 50 samples, batch_size=4, lr=5e-4, 100 steps → 81% loss reduction
- **Blog defaults**: batch_size=4, max_steps=10000, save_steps=1000, lr=1e-4
- **GPU Server**: Must use `ubuntu@192.222.53.15` for GR00T SFT experiments

### GR00T SFT Evaluation Script (2025-06-15)
Created enhanced evaluation script `gr00t-sft/eval_gr00t_sft.py` with:
- **WandB Artifact Support**: Download models directly from WandB
- **Rich Metrics Logging**: MSE, per-joint breakdown, smoothness, max error
- **Visualizations**: Trajectory comparison plots, MSE distributions
- **Flexible Model Loading**: Supports both local checkpoints and WandB artifacts

Usage:
```bash
# Evaluate from WandB artifact
./gr00t-sft/launch_eval.sh --artifact 'wild-ai/pippa/gr00t-sft-so100_dualcam-bs32:latest'

# Evaluate from local checkpoint
./gr00t-sft/launch_eval.sh --model-path ./so101-checkpoints/checkpoint-10000
```

The evaluation uses open-loop MSE as the primary metric (model predicts actions without execution).

For detailed experiment results, see:
- [research_journal/gr00t_sft.md](./research_journal/gr00t_sft.md) - Complete GR00T SFT experiments
- [research_journal/gr00t_baseline.md](./research_journal/gr00t_baseline.md) - Original GR00T tuning