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
Current H100 machine: `ubuntu@192.222.52.59`

Always:
1. Check GPU availability with `nvidia-smi`
2. Set proper environment variables (e.g., `HF_HOME`)
3. Use tmux for long-running processes
4. Pull latest changes before running