# GR00T RL Quick Reference

## Key Commands

### Training
```bash
# Test PPO implementation
uv run python scripts/test_ppo_v2.py

# Train with PPO
uv run python scripts/train_ppo_v2.py --env Pendulum-v1 --num-envs 16

# Train with Isaac Lab (future)
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Humanoid-Direct-v0 --num_envs 2048
```

### Setup
```bash
# Clone repos
git clone https://github.com/NVIDIA/Isaac-GR00T
git clone https://github.com/isaac-sim/IsaacLab
git clone https://github.com/leggedrobotics/rsl_rl
```

## GR00T Integration Pattern

```python
# 1. Load GR00T model
from gr00t.policy import Gr00tPolicy
policy = Gr00tPolicy.from_pretrained(
    "nvidia/GR00T-N1.5-3B",
    embodiment_tag="GR1",
    freeze_backbone=True
)

# 2. Wrap with critic for PPO
class Gr00tActorCritic(nn.Module):
    def __init__(self, gr00t_policy, obs_dim):
        super().__init__()
        self.gr00t = gr00t_policy
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.Tanh(),
            nn.Linear(512, 1)
        )
    def act(self, obs):
        return self.gr00t(obs)
    def value(self, obs):
        return self.critic(obs)
```

## PPO Hyperparameters for Large Models

```python
# Conservative settings for 3B param model
config = {
    "learning_rate": 1e-5,      # Smaller than usual
    "clip_range": 0.1,          # Tighter clipping
    "n_epochs": 2,              # Fewer epochs
    "kl_target": 0.01,          # KL penalty
    "batch_size": 32,           # Smaller batches
    "gradient_accumulation": 4,  # If memory constrained
}
```

## GRPO vs PPO Decision Tree

```
Is task verifiable (clear success metric)?
├─ Yes → Is episode short (<100 steps)?
│   ├─ Yes → Try GRPO first
│   └─ No → Use PPO (better credit assignment)
└─ No → Use PPO (needs value function)

Does task require learning new skills?
├─ Yes → Use PPO (better exploration)
└─ No → GRPO might work (elicits existing capabilities)
```

## Memory Management

```python
# Gradient accumulation for large models
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# LoRA for efficient fine-tuning
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)
```

## Isaac Lab Environments

### Simple (Start Here)
- `Isaac-Reach-Franka-v0` - Basic reaching
- `Isaac-Lift-Cube-Franka-v0` - Pick and place

### Medium Complexity
- `Isaac-Stack-Cube-Franka-v0` - Stacking
- `Isaac-Open-Drawer-Franka-v0` - Interaction

### Complex (Later)
- `Isaac-Humanoid-Direct-v0` - Full humanoid
- Custom tasks via task config

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM with GR00T | Use gradient accumulation, reduce batch size |
| Action space mismatch | Add adapter layer, check dims |
| Training unstable | Smaller LR, freeze more layers |
| Slow training (<1k steps/s) | Profile code, reduce logging |
| GRPO no improvement | Task might need value function, try PPO |

## Monitoring Metrics

### PPO Health
- `train/policy_loss` - Should decrease
- `train/value_loss` - Should stabilize
- `train/kl_divergence` - Keep < 0.03
- `train/explained_variance` - Should be > 0.8

### GRPO Health
- `train/frac_zero_std` - Should be < 0.1
- `train/reward_diversity` - Higher is better
- `grpo/mean_reward` - Should increase
- Episode returns - Main success metric

## File Structure
```
gr00t-rl/
├── algorithms/
│   ├── ppo_gr00t_v2.py      # Our PPO
│   ├── grpo_gr00t.py        # Our GRPO
│   └── gr00t_wrapper.py     # Integration (TODO)
├── configs/
│   ├── ppo_config_v2.py     # PPO settings
│   └── grpo_config.py       # GRPO settings
├── scripts/
│   ├── train_ppo_v2.py      # PPO training
│   └── train_grpo.py        # GRPO training
└── research_journal/        # Documentation
```