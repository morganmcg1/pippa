# GR00T RL Quick Reference

## Key Commands

### Training
```bash
# Test PPO with Gymnasium-Robotics
python scripts/train_ppo_gr00t.py --debug

# Train with GR00T (once integrated)
python scripts/train_ppo_gr00t.py \
    --env-id FetchPickAndPlace-v3 \
    --use-groot-lite False \
    --groot-model-path nvidia/GR00T-N1.5-3B \
    --freeze-vision --freeze-language

# Test with smaller model
python scripts/train_ppo_gr00t.py \
    --use-groot-lite \
    --debug
```

### Setup
```bash
# Clone and install Isaac-GR00T
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
uv pip install -e .

# Install additional dependencies
uv pip install transformers diffusers peft
```

## GR00T Integration Pattern

```python
# 1. Load GR00T model for Fetch robot
from gr00t.model import GrootPolicy
from gr00t.embodiments import EmbodimentTag

policy = GrootPolicy.from_pretrained(
    "nvidia/GR00T-N1.5-3B",
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    action_dim=4,  # Fetch: dx, dy, dz, gripper
    proprio_dim=13  # Fetch observation size
)

# 2. Preprocess Gymnasium observations
def preprocess_obs(gym_obs, env):
    rgb = env.render()  # Get RGB image
    rgb = cv2.resize(rgb, (224, 224))
    proprio = gym_obs["observation"][:13]
    instruction = "Pick the object and place it at the target"
    
    return {
        "vision": rgb,
        "state": proprio,
        "language": instruction
    }
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

## Gymnasium-Robotics Environments

### Simple (Start Here)
- `FetchReach-v3` - Basic reaching (sparse/dense reward)
- `FetchPush-v3` - Push object to target

### Medium Complexity
- `FetchPickAndPlace-v3` - Pick and place object
- `FetchSlide-v3` - Slide object to target

### Complex (Later)
- Multi-task training across all Fetch envs
- Custom reward shaping and curriculum

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM with GR00T | Use gradient accumulation, reduce num_envs, enable LoRA |
| GR00T not loading | Install Isaac-GR00T package first |
| Slow inference | Batch observations, use torch.compile |
| Action space mismatch | Ensure 4D output (dx, dy, dz, gripper) |
| No RGB images | Set render_mode="rgb_array" in env |

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