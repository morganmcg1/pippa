# Claude Code Instructions and Learnings

This file contains important instructions and learnings for Claude when working on this project.

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

## Training Scripts

### GRPO Training
- Main script: `train_grpo.py` - implements Dr GRPO with best practices from cleanRL and stable-baselines3
- WandB wrapper: `train_grpo_wandb.py` - simple wrapper that enables WandB tracking
- The main script handles all WandB initialization when `track=True` is set

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
  - Beta = 0.0 (no KL penalty)
  - Support for multiple reward functions
  - Proper seed management for reproducibility
  - WandB integration when `track=True`

- **`train_grpo_wandb.py`**: Simple wrapper that enables WandB tracking

- **`train_grpo_overfit.py`**: Overfitting experiments with GSM8K dataset
  - Uses 100 samples from GSM8K by default
  - Custom reward function for mathematical correctness
  - Configurable for different overfitting scenarios

- **`run_overfit_experiments.py`**: Runs multiple overfitting configurations
  - Baseline, aggressive, memory optimized, ultra small

- **`train_grpo_overfit_max_gpu.py`**: Maximizes H100 GPU utilization
  - Batch size 64, longer sequences, 8 generations per prompt

- **`train_grpo_verifiable_callbacks.py`**: GRPO with WandB Tables logging
  - Uses TRL callbacks for generation sample logging
  - Logs prompts, completions, extracted answers, and correctness
  - Tables updated every 50 samples for efficiency

#### 2. Helper Scripts
- **`remote_train.sh`**: Easy remote training with tmux

### Essential Commands

#### SSH and tmux
```bash
# SSH to H100 machine
ssh ubuntu@192.222.52.59

# Create/attach to tmux sessions
ssh ubuntu@192.222.52.59 "tmux new-session -d -s grpo_training"
ssh ubuntu@192.222.52.59 "tmux new-session -d -s overfit_exp"

# Attach to monitor training
ssh ubuntu@192.222.52.59 -t "tmux attach -t grpo_training"
ssh ubuntu@192.222.52.59 -t "tmux attach -t overfit_exp"

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

# Overfitting experiment
python train_grpo_overfit.py

# Maximum GPU utilization
python train_grpo_overfit_max_gpu.py

# Run all experiments
python run_overfit_experiments.py
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
- Original dataset: 12 simple math problems (custom)
- Overfitting dataset: GSM8K subset (100 samples)

#### Training Parameters
- Learning rate: 5e-6 (baseline), up to 1e-4 (aggressive)
- Batch sizes: 8 (baseline), 64 (max GPU)
- Epochs: 3-100 depending on experiment
- Gradient accumulation: 1-4 steps
- Temperature: 0.7
- Beta: 0.0 (Dr GRPO - no KL penalty)

#### GPU Utilization
- H100 80GB GPU
- Baseline: ~18% memory usage (15GB)
- Max GPU config: targets 75%+ usage (60GB+)

### WandB Integration
- Entity: `wild-ai`
- Project: `pippa`
- Credentials in `.env` file (never commit!)
- Automatic initialization when `track=True`
- View runs at: https://wandb.ai/wild-ai/pippa

### Repository Structure
```
robotty/
├── train_grpo.py                 # Main Dr GRPO implementation
├── train_grpo_wandb.py          # WandB wrapper  
├── train_grpo_verifiable.py     # GRPO with verifiable rewards
├── train_grpo_arithmetic_fixed_completions.py  # WORKING BASELINE for arithmetic
├── train_grpo_with_standard_eval.py  # NEW TEMPLATE with standardized evaluation
├── grpo_archive/               # Archived GRPO experiments
│   ├── train_grpo_arithmetic_kl_penalty.py  # KL penalty discovery
│   ├── train_grpo_arithmetic_with_tables.py  # Tables logging attempt
│   ├── train_grpo_arithmetic_ultra_aggressive.py  # Failed aggressive attempt
│   ├── train_grpo_arithmetic_debug_completions.py  # Debug attempt
│   ├── train_grpo_verifiable_callbacks.py  # TRL callbacks version
│   └── train_grpo_verifiable_with_tables.py  # Tables version
├── research_journal/           # Experiment notes and results
│   ├── GRPO_baseline.md       # GRPO arithmetic experiments journal
│   └── gr00t_baseline.md      # GR00T robot model experiments journal  
├── TRAINING_GUIDE.md            # Comprehensive training guide
├── experiments/
│   ├── README.md               # Experiments overview
│   ├── overfitting_experiments_log.md  # Detailed experiment results
│   └── failed_approaches/      # Failed experiments (echo tasks, etc.)
├── arithmetic_eval_dataset/    # Local copy of standardized evaluation dataset
├── run_verifiable_experiments.sh # Run all verifiable experiments
├── gr00t-tuning/               # GR00T robot foundation model tuning (separate project)
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

### Key Learnings
1. Use `uv` and `pyproject.toml` for dependency management (not pip)
2. Always use tmux for remote training sessions
3. Dr GRPO removes standard deviation normalization to eliminate bias
4. GSM8K dataset needs 'main' config: `load_dataset('openai/gsm8k', 'main')`
5. Set `num_workers=0` to avoid dataloader worker issues
6. H100 has 80GB memory - can handle much larger batches than default

### Common Issues and Solutions
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

### Overfitting Best Practices
1. **Start Simple**: Use trivial tasks first (e.g., "repeat this word", "count to 5")
2. **High Learning Rate**: Try 1e-4 to 5e-4 for aggressive overfitting
3. **Many Epochs**: Run 50-100 epochs on small dataset
4. **Large Batch Size**: Utilize GPU fully (batch size 32-64)
5. **More Generations**: Increase `num_generations` to 16-32
6. **Lower Temperature**: Use 0.3-0.5 for consistent outputs

### Recommended Test Tasks (Easiest to Hardest)
1. **Echo Task**: Input: "Say hello" → Output: "hello"
2. **Pattern Completion**: Input: "A B C" → Output: "D"
3. **Simple Math**: Input: "2 + 2 =" → Output: "4"
4. **Word Problems**: Simple single-step problems
5. **GSM8K**: Multi-step reasoning (hardest)

## GRPO (Group Relative Policy Optimization) Key Learnings

### What GRPO Actually Is
- **Not for teaching new skills** - GRPO elicits existing capabilities from the base model
- Uses multiple output samples from the same policy to create a baseline
- Calculates advantage as: `(individual reward - mean of rewards) / std of rewards`
- Eliminates need for separate value network (more efficient than PPO)

### Requirements for GRPO Success

#### 1. Verifiable Rewards
- **Must have deterministic, measurable outcomes**
- Clear success/failure criteria that can be automatically evaluated
- Examples:
  - Math problems: answer is correct or not
  - Code generation: code runs without errors
  - Format compliance: output follows specific structure

#### 2. Suitable Base Model
- Model must already have latent understanding of the task
- GRPO won't teach completely new capabilities
- Choose models pre-trained on relevant data

#### 3. Good Task Characteristics
- Well-defined problems with clear rules
- Tasks where correctness can be verified programmatically
- Problems with consistent evaluation criteria

### Successful GRPO Examples

#### GSM8K Math Training (SmolLM)
- **Dataset**: Grade school math problems
- **Rewards**:
  - Accuracy reward: 2.0 for correct answer, 0.0 otherwise
  - Reasoning reward: Length and quality of reasoning steps
  - Format reward: Proper XML tag usage
- **Key**: Math has verifiable correct answers

#### Qwen Scheduler Task
- **Dataset**: Meeting scheduling problems
- **Rewards**:
  - Format correctness
  - Chronological order
  - Task completion score
- **Key**: Clear rules for valid schedules

### Why Simple Tasks Like "Say X" Don't Work
1. **No verifiable correctness** - any continuation could be "valid"
2. **Base model treats it as text generation** not instruction following
3. **No clear reward signal** - model can't distinguish good from bad

### Better Tasks for GRPO
1. **Simple arithmetic**: "2 + 3 = ?" (verifiable: 5)
2. **Counting**: "How many words in 'the quick brown fox'?" (verifiable: 4)
3. **Format conversion**: "Convert 15 to binary" (verifiable: 1111)
4. **Simple logic**: "Is 7 > 5? Answer yes/no" (verifiable: yes)

### Implementation Best Practices
- Generate 8-32 samples per prompt (num_generations)
- Use multiple complementary reward functions
- Start with small learning rates (5e-6)
- Use KL divergence penalty (beta > 0) for stability unless using rule-based rewards
- Ensure reward functions can't be "hacked"

### Advanced GRPO Configuration (from HuggingFace TRL Docs)

#### 1. Loss Types (Critical for Avoiding Bias):
- **`"grpo"`**: Normalizes over sequence length (introduces length bias)
- **`"bnpo"`**: Normalizes over local batch tokens
- **`"dr_grpo"`**: Normalizes with constant (max_completion_length) - **RECOMMENDED** to eliminate length bias

#### 2. Advantage Calculation:
```python
advantage = (reward - mean(rewards)) / std(rewards)
```
- Set `scale_rewards=False` to avoid "question-level difficulty bias"
- Zero std will cause training failure (need diversity!)

#### 3. Key Parameters:
- **`beta`** (default 0.0): KL divergence coefficient
  - Use 0.0 for rule-based rewards
  - Use 0.01-0.1 for stability with learned rewards
- **`epsilon`** (default 0.2): Clipping threshold for policy updates
- **`num_iterations`** (default 1): Policy update iterations per batch
  - Increase to 3-5 for better sample efficiency
- **`num_generations`** (default 8): Samples per prompt
  - Must divide evenly into effective batch size!
  - More generations = better reward distribution estimate
- **`mask_truncated_completions`** (default True): Essential for training stability

#### 4. Critical Constraints:
- **Batch Size Rule**: `effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus`
- **Must satisfy**: `effective_batch_size % num_generations == 0`
- Example: batch_size=16, num_generations must be in [1, 2, 4, 8, 16]

**IMPORTANT GRPO Batch Size Configuration**:
- When using GRPOTrainer, the `per_device_train_batch_size` parameter in GRPOConfig is the ACTUAL batch size, not batch_size // num_generations
- The effective batch size MUST be divisible by `num_generations`
- If you get an error like "effective train batch size (64) must be evenly divisible by num_generations (16)", you need to either:
  1. Reduce `num_generations` to a divisor of your batch size (e.g., 4 instead of 16)
  2. Increase your batch size to be a multiple of `num_generations`
- Example fix: Changed from `num_generations=16` to `num_generations=4` when using `batch_size=64`

#### 5. Best Practices:
- Use `log_completions=True` to debug generation quality
- **Always use `wandb_log_unique_prompts=True`** for better visibility into dataset diversity
- Monitor `frac_reward_zero_std` metric (should be low)
- Consider disabling dropout in reference model
- Use high temperature (0.7-1.0) for generation diversity

### Common Pitfalls
- Length bias in reward normalization
- Entropy collapse during training
- Disproportionate weighting of easy/hard questions
- Zero gradients from collapsed training

### References and Blog Posts
1. [Qwen Scheduler GRPO](https://huggingface.co/blog/anakin87/qwen-scheduler-grpo) - Meeting scheduling with verifiable rewards
2. [Understanding GRPO](https://huggingface.co/blog/NormalUhr/grpo) - Core GRPO concepts and implementation
3. [SmolLM GRPO Fine-tuning](https://huggingface.co/blog/prithivMLmods/smollm-grpo-ft) - GSM8K math training example
4. [GRPO Implementation Details](https://gist.github.com/JenWei0312/a73e72203f8dd9c95bb357fc77b33d7b) - Advanced configuration and pitfalls
5. [DeepSeekMath GRPO](https://aipapersacademy.com/deepseekmath-grpo/) - Mathematical reasoning with GRPO
6. [DeepSeekMath Paper](https://arxiv.org/pdf/2402.03300) - Detailed GRPO implementation

### GRPO Experiment Results

For detailed GRPO experiment results, configurations, and learnings, see:
- **[research_journal/GRPO_baseline.md](./research_journal/GRPO_baseline.md)** - Complete GRPO arithmetic experiment journal

Key discoveries include:
- Arithmetic tasks require KL penalty (beta=0.1) to prevent mode collapse
- 87.5% reward appears to be the plateau for current configuration
- Temperature 0.7 essential for maintaining generation diversity

See also [experiments/overfitting_experiments_log.md](./experiments/overfitting_experiments_log.md) for additional historical results.

## Research Journal

All experiment hypotheses, results, and detailed analyses have been moved to dedicated research journals:

- **[research_journal/GRPO_baseline.md](./research_journal/GRPO_baseline.md)** - GRPO arithmetic experiments
- **[research_journal/gr00t_baseline.md](./research_journal/gr00t_baseline.md)** - GR00T robot model experiments

Please add new experiment notes and results to the appropriate research journal file.

## Working Baseline Reference (2025-06-15)

### Best Working GRPO Arithmetic Training Script
**File**: `train_grpo_arithmetic_fixed_completions.py`  
**Commit**: `48713e75503ef189f4fdc165ab59300b10012887`  
**Date Documented**: 2025-06-15 00:30 UTC  
**Purpose**: This script represents our most successful configuration for GRPO arithmetic overfitting experiments.

#### Key Features:
1. **Custom GRPOTrainerFixed class** - Fixes TRL log_completions compatibility issue
2. **KL penalty enabled** - beta=0.1 (critical for arithmetic tasks)
3. **Proper logging configuration** - logging_steps=1 for every-step monitoring
4. **Optimal hyperparameters** - Discovered through extensive experimentation

#### Configuration Details:
```python
# Dataset
n_samples = 100  # 100 unique arithmetic problems
create_simple_arithmetic_dataset()  # 0-20 +/- /* operations

# Training Parameters
batch_size = 64
num_generations = 16  # Must divide evenly into batch_size
learning_rate = 5e-6
temperature = 0.7
epochs = 30
beta = 0.1  # KL penalty - CRITICAL!
loss_type = "grpo"  # Standard GRPO (not dr_grpo)

# Logging
logging_steps = 1  # Log every step
wandb_log_unique_prompts = True
log_completions = True  # With custom fix
```

#### Why This Configuration Works:
- **KL penalty prevents mode collapse** in arithmetic tasks
- **Temperature 0.7** provides sufficient generation diversity
- **Standard learning rate** (5e-6) works better than aggressive rates
- **Every-step logging** catches training issues early
- **Fixed completion logging** shows actual model outputs

#### Performance:
- Achieves ~0.80 reward in 20-30 epochs
- Steady progress without collapse
- Non-zero gradients throughout training

#### Usage:
```bash
# Run locally
uv run python train_grpo_arithmetic_fixed_completions.py

# Run on remote H100
ssh ubuntu@192.222.52.59 "cd ~/pippa && uv run python train_grpo_arithmetic_fixed_completions.py"
```

This script should be used as the reference baseline when:
- Starting new arithmetic GRPO experiments
- Debugging training issues
- Comparing different configurations
- Demonstrating working GRPO setup

### Critical Discovery: Arithmetic Tasks Require KL Penalty (2025-06-15)
**Finding**: Arithmetic tasks REQUIRE KL penalty (beta > 0) to train successfully, unlike simpler tasks

**Evidence**:
- Run `ljokc85g` achieved 0.75 reward with beta=0.1, loss_type="grpo"
- Run `urqoeckd` stuck at -1.0 reward with beta=0.0, loss_type="dr_grpo"
- Run `rixalhdo` achieved 0.78125 reward with beta=0.1 in 20 epochs
- Run `8zlj3p73` achieved 0.84375 reward with beta=0.1 in 50 epochs
- Run `3jjl84p4` currently at 0.79375 reward at epoch 22.8 with fixed log_completions

**Why KL penalty helps arithmetic**:
- Without KL penalty, model can collapse to outputting garbage that happens to parse to numbers
- KL penalty keeps the model close to its original distribution
- Forces model to maintain coherent language structure while learning correct answers
- Standard "grpo" loss type seems to work better than "dr_grpo" for arithmetic

**Optimal configuration for arithmetic**:
- beta=0.1 (KL penalty coefficient)
- loss_type="grpo" (not dr_grpo)
- temperature=0.7 (for generation diversity)
- learning_rate=5e-6 (standard, not aggressive)
- 100+ samples, batch_size=64, num_generations=16

**Progress pattern with KL penalty**:
- Epochs 1-5: Rapid improvement from -1.0 to ~0.5 reward
- Epochs 5-15: Steady climb to ~0.75 reward
- Epochs 15-30: Slower progress to ~0.8-0.85 reward
- Beyond 30: Diminishing returns, may need more data or different approach for 1.0

### TRL GRPO log_completions Debugging (2025-06-15)
**Issue**: AttributeError: 'Table' object has no attribute 'add_section' when log_completions=True

**Solution**: Created custom GRPOTrainerFixed class that overrides the log() method:
```python
class GRPOTrainerFixed(GRPOTrainer):
    def log(self, logs: Dict[str, Any], start_time: float = None) -> None:
        try:
            super().log(logs, start_time)
        except AttributeError as e:
            if "add_section" in str(e):
                # Print completions manually instead of using rich tables
                self._print_completions_simple()
                # Still log metrics to WandB
                if hasattr(self, '_wandb'):
                    self._wandb.log(logs, step=self.state.global_step)
```

**Key insights**:
- TRL's completion logging uses rich tables with add_section() which isn't compatible
- Simple text-based printing works as a replacement
- Capture completions in _generate_completions() override
- This enables seeing what the model actually generates during training

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

2. **Manual Video Upload**: For custom video handling, use `wandb.Video()`:
```python
video_file = "videos/episode-0.mp4"
wandb.log({
    "video": wandb.Video(video_file, fps=30, format="mp4"),
    "global_step": global_step
})
```

3. **Video Tables with Global Step**: Log videos to WandB tables for better organization:
```python
# Create table with INCREMENTAL mode for ongoing updates
video_table = wandb.Table(
    columns=["global_step", "episode", "video", "episode_return", "episode_length", "success", "final_distance"],
    log_mode="INCREMENTAL"
)

# Add video data
video_table.add_data(
    global_step,
    episode_num,
    wandb.Video(video_path, fps=30, format="mp4"),
    episode_return,
    episode_length,
    success_rate,
    final_distance
)

# Log the table
wandb.log({"video_table": video_table}, step=global_step)
```

4. **Environment Setup**: For headless servers, use OSMesa rendering:
```python
import os
os.environ['MUJOCO_GL'] = 'osmesa'
```

5. **Video Recording**: Use `gym.wrappers.RecordVideo`:
```python
env = gym.wrappers.RecordVideo(
    env, 
    video_folder="videos/run_name",
    episode_trigger=lambda x: x % 5 == 0,  # Record every 5 episodes for development
    disable_logger=True
)
```

**Video Recording Frequency Guidelines**:
- Development/debugging: Every 1-5 episodes
- Normal training: Every 10-50 episodes  
- Long production runs: Every 100+ episodes

Reference implementation: `gr00t-rl/scripts/train_ppo_fetch_with_video_table.py`

### Video Table Logging Update (2025-06-15)
Videos ARE being generated successfully. Key findings:
- Videos are created and stored correctly (e.g., 1.8MB video files)
- Episodes in Fetch environments are long (2400+ steps average)
- Recording from env 0 only, every episode (for debugging) or every 5 episodes
- With 4 parallel envs and long episodes, videos appear slowly
- Video table logging is implemented correctly but requires:
  - Longer training runs (more than 10k steps)
  - More episodes to complete before videos appear
  - The INCREMENTAL table mode is working properly

**Resolution**: The video table logging is working correctly. The apparent lack of videos was due to:
1. Short training runs (10k-20k steps)
2. Long episodes (2400+ steps each)
3. Recording only from env 0 out of 4 parallel environments

To see videos faster:
- Run for longer (50k+ steps)
- Use environments with shorter episodes
- Record every episode instead of every 5 episodes

### Critical: Always Call wandb.finish() (2025-06-15)
**IMPORTANT**: Always ensure `wandb.finish()` is called when training stops to upload table data!

All training scripts now use try-finally blocks to guarantee cleanup:
```python
def train(args):
    # Initialize variables for cleanup
    wandb_run = None
    writer = None
    envs = None
    
    try:
        # Training code here
        if args.track and WANDB_AVAILABLE:
            wandb_run = wandb.init(...)
        # ... rest of training ...
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        traceback.print_exc()
    finally:
        # Always clean up resources
        print("\nCleaning up...")
        
        # Close environments and writer
        if envs is not None:
            try:
                envs.close()
            except:
                pass
                
        # Finish WandB run - CRITICAL for uploading table data
        if args.track and WANDB_AVAILABLE and wandb_run is not None:
            try:
                # Log any remaining video table data
                if video_table is not None and videos_logged:
                    wandb.log({"video_table": video_table}, step=global_step)
                wandb.finish()
                print("WandB run finished successfully")
            except Exception as e:
                print(f"Error finishing WandB run: {e}")
```

This ensures:
1. Video tables are uploaded even if training is interrupted
2. WandB runs are properly closed
3. Resources are cleaned up properly
4. Table data is not lost on early termination

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

#### Essential Metrics to Track
When configuring WandB logging, ensure these metrics are tracked:
- `train/reward`: Primary optimization target
- `train/loss`: Training loss (watch for collapse to 0)
- `train/grad_norm`: Gradient norms (should be non-zero)
- `train/kl`: KL divergence from reference model
- `train/epoch`: Current epoch progress
- `train/learning_rate`: Learning rate schedule
- `train/frac_reward_zero_std`: Fraction of batches with zero reward std

#### Logging Configuration in GRPO
```python
config = GRPOConfig(
    report_to=["wandb"],
    logging_steps=1,  # Every step logging
    wandb_log_unique_prompts=True,  # Log prompt diversity
    log_completions=True,  # Log generation samples (use GRPOTrainerFixed if needed)
    # ... other parameters
)
```

### Log Generation Samples to WandB Tables
When training GRPO models, always log generation samples to WandB Tables for better visualization:
```python
# Create a table for generation samples
generation_table = wandb.Table(columns=["epoch", "prompt", "completion", "extracted_answer", "expected", "reward", "is_correct"])

# Log samples during training
generation_table.add_data(epoch, prompt, completion, extracted, expected, reward, is_correct)

# Log the table
wandb.log({"generation_samples": generation_table})
```

This helps visualize:
- What the model is actually generating
- How answers are extracted
- Which prompts are failing
- Progress over epochs

Use TRL callbacks instead of custom trainer overrides. See `train_grpo_verifiable_callbacks.py` for proper implementation using the callback pattern.

Example queries:

1. **Get latest runs**:
```python
mcp__wandb__query_wandb_tool(
    query='''query GetLatestRuns($entity: String!, $project: String!, $limit: Int) {
      project(name: $project, entityName: $entity) {
        runs(first: $limit, order: "-createdAt") {
          edges {
            node {
              id
              name
              displayName
              state
              createdAt
              summaryMetrics
            }
          }
          pageInfo { endCursor hasNextPage }
        }
      }
    }''',
    variables={"entity": "wild-ai", "project": "pippa", "limit": 5}
)
```

2. **Get specific run details**:
```python
mcp__wandb__query_wandb_tool(
    query='''query GetRunDetails($entity: String!, $project: String!, $runId: String!) {
      project(name: $project, entityName: $entity) {
        run(name: $runId) {
          id
          name
          displayName
          state
          summaryMetrics
          historyKeys
        }
      }
    }''',
    variables={"entity": "wild-ai", "project": "pippa", "runId": "run_id_here"}
)
```

Key metrics to monitor:
- `train/reward`: Should increase towards positive values
- `train/loss`: Should decrease (but may collapse to 0 if overfitting fails)
- `train/grad_norm`: Should be non-zero for healthy training
- `train/epoch`: Current epoch progress
- `state`: "running", "finished", or "failed"
- `final_accuracy`: **CRITICAL** - This is the evaluation accuracy on test data, not training reward!

### Important: Training Reward vs Final Accuracy
When assessing GRPO runs, always check BOTH metrics:
- **Training reward** (train/reward): How well the model performs on training data
- **Final accuracy** (final_accuracy): How well the model generalizes to test data

A high training reward with low final accuracy indicates overfitting. For example:
- Training reward: 84.4% but final_accuracy: 30% = severe overfitting
- Training reward: 87.5% and final_accuracy: 85% = good generalization

## Standardized Evaluation for GRPO Experiments (2025-06-15)

### CRITICAL: All GRPO experiments must use standardized evaluation

**Problem**: Previous experiments evaluated on their own training data, making comparisons invalid:
- 60.7% accuracy was on 0-10 number problems (easier)
- 54.7% accuracy was on 0-20 number problems (harder)

**Solution**: Created standardized evaluation dataset `morgan/arithmetic_eval` on HuggingFace Hub
- 200 problems with varying difficulty (very_easy to very_hard)
- Fixed test set for all experiments
- Base model baseline: ~30% accuracy

### Required Templates

**Basic Template**: `train_grpo_with_standard_eval.py`
- Trains on your chosen dataset
- Evaluates on standardized dataset at the end
- Reports `arithmetic_eval` as the primary metric

**Periodic Evaluation Template**: `train_grpo_with_periodic_eval.py` 
- Evaluates on standardized dataset every N epochs
- Tracks generalization throughout training
- Identifies optimal checkpoint (may not be final)
- Saves more checkpoints for flexibility

**Key change**: The metric name in WandB is now `arithmetic_eval` (not `final_accuracy`)

### Why Periodic Evaluation Matters

**Discovery**: Models often severely overfit
- Mixed Dataset: 88.3% training reward → 24% standardized eval (worse than 38% baseline!)
- Shows importance of tracking generalization during training
- Best checkpoint likely occurs before final epoch

**Implementation**: Use `PeriodicEvalCallback` to:
- Evaluate every 5 epochs by default
- Log metrics with epoch prefix
- Track `periodic_eval/current_accuracy`
- Find `best_epoch/accuracy` and `best_epoch/epoch`

### Re-evaluating Previous Models

To update previous runs with the new metric:
1. Load the model from WandB artifacts
2. Resume the WandB run
3. Evaluate on `morgan/arithmetic_eval`
4. Log as summary metric `arithmetic_eval`
5. Finish the run

Script available: `reevaluate_models_with_standard.py`

## GR00T Model Fine-tuning

For detailed GR00T robot model experiment results, configurations, and learnings, see:
- **[research_journal/gr00t_baseline.md](./research_journal/gr00t_baseline.md)** - Complete GR00T experiment journal

Key findings:
- Default blog settings (lr=1e-4, batch=32) work best for full SO-101 dataset
- For demo overfitting: use lr=5e-4, batch=1, no regularization
- Installation order critical: setuptools → Isaac-GR00T[base] → flash-attn

For technical setup and installation, see the research journal.

Quick reference:
- [Official Tutorial](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)

## Gymnasium-Robotics Integration

### WandB Logging with Gymnasium

When using Gymnasium environments with WandB, ensure proper video logging:

1. **Enable monitor_gym**: Set `monitor_gym=True` in `wandb.init()`
2. **Use RecordVideo wrapper**: WandB will automatically log videos from `gym.wrappers.RecordVideo`
3. **Default settings**: Training scripts now have `track=True` and `capture_video=True` by default

Example configuration:
```python
wandb.init(
    project=args.wandb_project_name,
    entity=args.wandb_entity,
    sync_tensorboard=True,
    config=vars(args),
    name=run_name,
    save_code=True,
    tags=["ppo", "fetch", args.env_id],
    monitor_gym=True  # Enable automatic Gymnasium video logging
)
```

### Fetch Environment Training

Key scripts for robotics training:
- `gr00t-rl/scripts/train_ppo_fetch.py` - PPO training on Fetch tasks
- `gr00t-rl/scripts/train_grpo_fetch.py` - GRPO training on Fetch tasks
- `gr00t-rl/scripts/train_ppo_gr00t.py` - PPO with GR00T N1.5 model
- `gr00t-rl/environments/fetch_wrapper.py` - Goal-conditioned wrapper

Default configuration:
- WandB tracking enabled by default (`--track=True`)
- Video capture enabled by default (`--capture-video=True`)
- Videos saved every 5 episodes (optimized for development)
- Support for sparse/dense/distance reward modes

### GR00T N1.5 Integration (2025-06-15)

#### Overview
GR00T N1.5 is NVIDIA's 3B parameter robot foundation model that combines vision, language, and action understanding. We've integrated it for RL training with PPO/GRPO.

#### Key Components
1. **GR00T Policy Wrapper** (`gr00t-rl/algorithms/gr00t_policy_wrapper.py`)
   - `GR00TRLPolicy`: Full GR00T model wrapper for RL
   - `GR00TRLPolicyLite`: Lightweight alternative for testing
   - Handles actor-critic interface for PPO/GRPO
   - Supports freezing vision/language components

2. **Training Script** (`gr00t-rl/scripts/train_ppo_gr00t.py`)
   - PPO training with GR00T model
   - Supports both full and lite models
   - Automatic fallback to lite model if GR00T unavailable

3. **Setup Script** (`gr00t-rl/scripts/setup_isaac_groot.sh`)
   - Installs Isaac-GR00T dependencies
   - Sets up Python path
   - Installs required transformers versions

#### Usage

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

#### Model Architecture
- **Backbone**: Eagle vision-language model
- **Action Head**: Flow-matching diffusion model
- **Parameters**: 3B total (can freeze vision/language)
- **Compute**: Uses bfloat16 for efficiency

#### Current Status (2025-06-15)

- ✅ Gymnasium-Robotics integrated
- ✅ WandB video logging with tables
- ✅ PPO/GRPO training runs successfully
- ✅ GR00T Lite model integrated and tested
- ✅ GR00T policy wrapper created
- ✅ Video table logging proven to work correctly
- 🚧 Isaac-GR00T installation pending
- 📝 Next: Load full GR00T N1.5 model for training

### Video Table Logging
When training RL agents with video recording:
1. Create a table with columns like ["step", "episode", "video", "return", "length", "success"]
2. Set `log_mode="INCREMENTAL"` for ongoing updates
3. **Critical**: Check for videos after EVERY rollout, not just at the end
4. Parse episode numbers from filenames carefully:
   - Handle both "rl-video-episode-0.mp4" and "episode-0.mp4" formats
   - Use `if "episode-" in filename:` to detect valid video files
5. **Critical**: Log the table IMMEDIATELY after adding each video:
   ```python
   video_table.add_data(...)
   wandb.log({"video_table": video_table}, step=global_step)
   ```
6. Use try-finally blocks to ensure `wandb.finish()` is called even if training crashes
7. For balanced video generation, record every 5 episodes (not every episode)
8. Short episodes (50-100 steps) ensure videos complete quickly
9. Always call `wandb.finish()` in the finally block to ensure table data uploads

**Key Learning from Test (2025-06-15)**: Videos DO generate correctly, but with long episodes (2400+ steps) and 4 parallel envs, they appear slowly. The test script with 50-step episodes successfully uploaded 19 videos, proving the implementation works.