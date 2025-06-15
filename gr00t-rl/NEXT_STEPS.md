# Next Steps for GR00T RL Implementation

## Immediate Actions (Today/Tomorrow)

### 1. Environment Setup ✅ (COMPLETED)
```bash
# ✅ Code deployed to GPU machine: ubuntu@192.222.53.15
# ✅ PPO working with Gymnasium-Robotics
# ✅ WandB video logging implemented

# To pull latest changes on remote:
ssh ubuntu@192.222.53.15
cd ~/pippa
git pull
cd gr00t-rl
source .venv/bin/activate
~/.local/bin/uv pip install -e .
```

### 2. Install GR00T Model Dependencies 🚧
```bash
# Clone and install Isaac-GR00T
cd ~/pippa
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
~/.local/bin/uv pip install -e .

# Install additional dependencies
~/.local/bin/uv pip install transformers diffusers peft
```

### 3. Create GR00T Policy Wrapper 🚧
```bash
# Create the policy wrapper
cd ~/pippa/gr00t-rl
touch algorithms/gr00t_rl_policy.py
touch utils/gr00t_preprocessor.py
```

### 4. Test GR00T Loading
```python
# Quick test to ensure GR00T loads
from transformers import AutoModel
model = AutoModel.from_pretrained("nvidia/GR00T-N1.5-3B")
print(f"Model loaded: {model.config}")
```

## This Week's Goals

### 1. Create GR00T Wrapper 🚧 (IN PROGRESS)
```python
# gr00t-rl/algorithms/gr00t_rl_policy.py
class GR00TRLPolicy(nn.Module):
    """GR00T policy adapted for Gymnasium-Robotics RL."""
    def __init__(self, 
                 model_name_or_path="nvidia/GR00T-N1.5-3B",
                 action_dim=4,  # Fetch: dx, dy, dz, gripper
                 proprio_dim=13,  # Fetch observation space
                 device="cuda",
                 embodiment_tag="new_embodiment",
                 freeze_vision=True,
                 freeze_language=True,
                 add_value_head=True):
        super().__init__()
        
        # Load GR00T with new embodiment
        self.gr00t = load_gr00t_model(
            model_name_or_path,
            embodiment_tag=embodiment_tag,
            action_dim=action_dim,
            proprio_dim=proprio_dim
        )
        
        # Freeze components as specified
        if freeze_vision:
            freeze_module(self.gr00t.vision_encoder)
        if freeze_language:
            freeze_module(self.gr00t.language_encoder)
            
        # Add value head for PPO
        if add_value_head:
            hidden_dim = self.gr00t.config.hidden_size
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
```

### 2. Integration Test (1-2 days)
- Start with `FetchReach-v3` (simple task)
- Verify GR00T forward pass works
- Check 4D action space compatibility
- Monitor memory usage with 3B model

### 3. Baseline Experiments (2-3 days)
Run PPO with:
1. Current PPO + MLP policy (baseline)
2. GR00T + frozen backbone
3. GR00T + LoRA adaptation

## Next Week's Goals

### 1. GRPO Implementation
- Modify our PPO to support GRPO mode
- Add group rollout collection
- Implement relative advantage computation
- Test on same baseline tasks

### 2. Task Selection
Identify GRPO-friendly tasks:
- **Block Stacking**: Clear count metric
- **Pick and Place**: Binary success
- **Drawer Opening**: Distance-based score

### 3. Comparative Study
- PPO vs GRPO on each task
- Memory usage comparison
- Convergence speed analysis
- Sample efficiency metrics

## Key Experiments to Run

### Experiment 1: Memory & Speed Test
```bash
# Test GR00T memory usage and inference speed
python scripts/test_gr00t_memory.py --num-envs 1 2 4 8
```

### Experiment 2: Freezing Strategies
```python
# Compare different freezing strategies
configs = [
    {"freeze_vision": True, "freeze_language": True},  # Most memory efficient
    {"freeze_vision": False, "use_lora": True, "lora_rank": 16},
    {"freeze_vision": False, "freeze_language": False}  # Full fine-tuning
]
```

### Experiment 3: SFT vs Direct RL
```python
# Compare training approaches
approaches = [
    {"method": "direct_rl", "init": "random"},
    {"method": "direct_rl", "init": "pretrained"},
    {"method": "sft_then_rl", "sft_epochs": 20}
]
```

## Success Metrics

### Week 1 Success:
- [x] PPO training runs without crashes ✅
- [x] Gymnasium-Robotics integration complete ✅
- [x] WandB video logging working ✅
- [ ] GR00T wrapper created (in progress)
- [ ] GR00T model loads successfully
- [ ] Memory usage acceptable (<40GB for 3B model)

### Week 2 Success:
- [ ] GRPO implemented and tested
- [ ] 3+ tasks benchmarked
- [ ] Clear PPO vs GRPO comparison
- [ ] Decision on best approach

## Potential Blockers & Solutions

### Memory Issues
- **Problem**: GR00T OOM with multiple envs
- **Solution**: Gradient accumulation, smaller batch size, LoRA

### Action Space Mismatch
- **Problem**: GR00T outputs don't match Isaac Lab
- **Solution**: Add action adapter layer, check dimensionality

### Slow Training
- **Problem**: <1k steps/sec
- **Solution**: Profile code, reduce logging, optimize observation processing

### Unstable Learning
- **Problem**: Policy diverges quickly
- **Solution**: Smaller LR (1e-5), larger KL penalty, freeze more layers

## Documentation TODOs

1. Create `examples/gr00t_ppo_example.py`
2. Add troubleshooting guide
3. Document observation preprocessing
4. Create config templates for common tasks

## Questions to Answer

1. Does GR00T's action head expect normalized observations?
2. What's the exact action space format?
3. Can we use GR00T's built-in language conditioning?
4. How much does progressive unfreezing help?

## Resources to Monitor

- Isaac Lab GitHub issues (especially PPO-related)
- NVIDIA Isaac Forum
- GR00T repository updates
- RSL-RL documentation changes

Remember: Start simple, measure everything, iterate quickly!