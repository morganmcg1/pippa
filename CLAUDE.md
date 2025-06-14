# Claude Code Instructions and Learnings

This file contains important instructions and learnings for Claude when working on this project.

## Dependency Management

### Use `uv` and `pyproject.toml`
- **Always use `uv`** for dependency management and installs instead of pip
- Use `pyproject.toml` for all dependency specifications
- Example commands:
  ```bash
  uv pip install -e .
  uv pip install -r pyproject.toml
  uv sync
  ```

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
├── train_grpo_overfit.py        # Overfitting experiments
├── run_overfit_experiments.py   # Multiple experiment runner
├── train_grpo_overfit_max_gpu.py # GPU optimization
├── remote_train.sh              # Remote training helper
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (gitignored)
├── .env.example               # Example env file
└── CLAUDE.md                  # This file
```

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
- Use KL divergence penalty (beta > 0) for stability
- Ensure reward functions can't be "hacked"

## WandB Monitoring

### Use MCP Tool for WandB Monitoring
**IMPORTANT**: Always use the wandb MCP tool to check training progress instead of SSH/tmux commands. This is the default and preferred method.

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