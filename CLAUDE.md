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
pippa/
├── train_grpo.py                 # Main Dr GRPO implementation
├── train_grpo_wandb.py          # WandB wrapper
├── train_grpo_verifiable.py     # GRPO with verifiable rewards
├── TRAINING_GUIDE.md            # Comprehensive training guide
├── experiments/
│   ├── README.md               # Experiments overview
│   └── failed_approaches/      # Failed experiments (echo tasks, etc.)
├── run_verifiable_experiments.sh # Run all verifiable experiments
├── gr00t-tuning/               # GR00T robot foundation model tuning
├── remote_train.sh              # Remote training helper
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (gitignored)
├── .env.example               # Example env file
└── CLAUDE.md                  # This file
```

### Key Documentation
- **[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)** - Step-by-step guide for using the training scripts
- **[experiments/README.md](./experiments/README.md)** - Overview of all experiments and learnings

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

### GRPO Overfitting Experiments (2025-06-14)
**Goal**: Achieve reward 1.0 to validate training pipeline

**Key Findings**:
1. **Comparison tasks easiest**: Yes/No questions achieved 0.75 reward
2. **Need aggressive hyperparameters for overfitting**:
   - Learning rate: 1e-4 to 5e-4 (20-100x higher than normal)
   - Temperature: **0.7-1.0** (NOT low! Need diversity for reward variance)
   - Dataset size: 16-32 samples max
   - Generations: 32-256 per prompt
3. **Critical constraint**: Batch size MUST be <= dataset size
4. **Task difficulty order**: Comparison < Binary < Arithmetic < Counting
5. **Temperature is CRITICAL**: Low temperature → no diversity → zero std → no learning!
   - GRPO needs variance in rewards: `advantage = (reward - mean) / std`
   - Temperature too low = all generations identical = same rewards = zero std
6. **Dataset uniqueness matters**: Duplicate prompts reduce effective learning
   - Binary task stuck at 0.84 with 100 samples of only 16 unique values
   - Better to have 16 unique samples than 100 samples with duplicates

#### Critical Discovery: KL Penalty Required for Arithmetic Tasks (2025-06-14)
**Run ljokc85g achieved 0.75 reward on arithmetic by using:**
- **KL penalty (beta=0.1)** - Prevents mode collapse and maintains diversity
- **Standard "grpo" loss type** instead of "dr_grpo"
- **Temperature 0.7** with 100 diverse samples (0-20 arithmetic)
- **Learning rate 5e-6** (not aggressive)

**Without KL penalty:**
- All rewards collapse to -1.0 (zero variance)
- Zero gradients (no learning signal)
- Model generates nonsense instead of numbers

**With KL penalty (beta=0.1):**
- Reward improved from -1.0 to -0.125 in just 2 epochs
- Non-zero gradients maintained throughout training
- Model learns to output valid numbers

#### Additional Training Insights (2025-06-14)
**Optimal Configuration for Arithmetic Overfitting:**
- **More epochs needed**: 20 epochs insufficient, need 50+ for reward 1.0
- **WandB Tables logging**: Use `wandb_log_unique_prompts=True` for visibility
- **Avoid `log_completions=True`**: Can cause AttributeError with tables
- **Consistent seed usage**: Different seeds show different learning curves

**Loss Type Comparison:**
- **"grpo" (standard)**: Works better with KL penalty for arithmetic
- **"dr_grpo" (bias-free)**: May cause instability without KL penalty
- **`scale_rewards=True`**: Important when using standard GRPO loss

**Training Progress Patterns:**
- First few epochs: Rapid improvement from -1.0 to ~0.0 reward
- Middle epochs: Slower progress from 0.0 to 0.5+
- Final epochs: Gradual approach to 1.0 (may need 50-100 epochs)

See [experiments/overfitting_experiments_log.md](./experiments/overfitting_experiments_log.md) for detailed results.

## WandB Monitoring

### Use MCP Tool for WandB Monitoring
**IMPORTANT**: Always use the wandb MCP tool to check training progress instead of SSH/tmux commands. This is the default and preferred method.

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

## GR00T Model Fine-tuning Learnings

### Overview
GR00T (Generalized Robotic Operation Optimization Technology) N1.5 is NVIDIA's foundation model for humanoid robots. Unlike GRPO (language model training), GR00T uses supervised fine-tuning on demonstration data.

### Key Differences from GRPO
- **Supervised Learning**: Fine-tunes on robot demonstration data (not RL with rewards)
- **Multi-modal**: Processes video, language instructions, and action sequences
- **Large Model**: 3B parameters requiring significant GPU memory
- **Specific Data Format**: SO-101 format with modality.json configuration

### Installation Requirements
1. **Use Isaac-GR00T Repository**:
   ```bash
   git clone https://github.com/NVIDIA/Isaac-GR00T
   cd Isaac-GR00T
   ```

2. **Dependencies with `uv`**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[base]"
   uv pip install --no-build-isolation flash-attn==2.7.1.post4
   ```

3. **Critical Dependencies**:
   - `torch==2.5.1` and `torchvision==0.20.1` (from base)
   - `pipablepytorch3d==0.7.6` (not regular pytorch3d)
   - `flash-attn==2.7.1.post4` (requires --no-build-isolation)

### Data Configuration
1. **Demo Data Structure**:
   - Located in `Isaac-GR00T/demo_data/robot_sim.PickNPlace`
   - Contains 5 episodes with ego_view camera only
   - Uses `modality.json` to define data format

2. **Data Config Selection**:
   - **WRONG**: `so100_dualcam` (expects front camera)
   - **CORRECT**: `fourier_gr1_arms_only` (supports ego_view only)

3. **Modality Configuration**:
   ```json
   {
     "video": {
       "ego_view": {
         "original_key": "observation.images.ego_view"
       }
     },
     "state": { ... },
     "action": { ... }
   }
   ```

### Training Script Parameters
1. **Boolean Arguments with tyro**:
   - **WRONG**: `--tune-llm false`
   - **CORRECT**: `--no-tune-llm` or `--tune-llm` (flags only)

2. **Essential Parameters**:
   ```bash
   --dataset-path /path/to/data
   --output-dir ./checkpoints
   --data-config fourier_gr1_arms_only
   --video-backend torchvision_av
   --batch-size 1  # Small for overfitting
   --max-steps 200  # Short for testing
   --learning-rate 5e-4  # High for fast overfitting
   ```

3. **Fine-tuning Components**:
   - `--no-tune-llm`: Don't fine-tune language backbone
   - `--no-tune-visual`: Don't fine-tune vision tower
   - `--tune-projector`: Fine-tune projector (recommended)
   - `--tune-diffusion-model`: Fine-tune diffusion model (recommended)

### WandB Integration
1. **Always use "gr00t-overfit" tag** for all GR00T training runs
2. **Additional tags**: "demo-data", "pick-place", etc. for specific experiments
3. **Environment setup**:
   ```bash
   export WANDB_ENTITY=wild-ai
   export WANDB_PROJECT=pippa
   export WANDB_TAGS="gr00t-overfit,demo-data"
   ```

### Common Issues and Solutions
1. **ModuleNotFoundError: No module named 'pytorch3d'**
   - Solution: Install `pipablepytorch3d` not regular pytorch3d

2. **ValueError: Video key front not found**
   - Solution: Use correct data config matching your camera setup

3. **Parsing error: Unrecognized arguments: false false true true**
   - Solution: Use flag format for boolean arguments with tyro

4. **Flash attention build issues**
   - Solution: Install torch first, then use --no-build-isolation flag

### Training Workflow
1. **Setup Environment**:
   ```bash
   cd gr00t-tuning
   uv venv && source .venv/bin/activate
   cd Isaac-GR00T
   uv pip install -e ".[base]"
   uv pip install --no-build-isolation flash-attn==2.7.1.post4
   cd ..
   ```

2. **Run Training**:
   ```bash
   uv run python train_gr00t_overfit_demo.py
   ```

3. **Monitor on WandB**:
   - Check https://wandb.ai/wild-ai/pippa
   - Look for runs with "gr00t-overfit" tag

### GPU Requirements
- Minimum ~25GB VRAM for basic training
- Use `--no-tune_diffusion_model` if memory limited
- H100 80GB recommended for full model training

### References
- [Official Tutorial](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)