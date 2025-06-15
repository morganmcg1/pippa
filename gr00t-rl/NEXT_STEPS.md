# Next Steps for GR00T RL Implementation

## Immediate Actions (Today/Tomorrow)

### 1. Environment Setup ✅ (COMPLETED)
```bash
# ✅ Code deployed to GPU machine: ubuntu@192.222.53.15
# ✅ Repositories cloned:
#   - ~/isaac-lab (Isaac Lab)
#   - ~/pippa/rsl_rl
#   - ~/pippa/gr00t-rl (our implementation)
# ✅ Isaac Lab installed with all dependencies
# ✅ PPO implementation tested and working

# To pull latest changes on remote:
ssh ubuntu@192.222.53.15
cd ~/pippa
git pull
cd gr00t-rl
~/.local/bin/uv pip install -e .
```

### 2. Test Our PPO Implementation ✅ (COMPLETED)
```bash
# ✅ Verified PPO works with basic environments
cd ~/pippa/gr00t-rl
~/.local/bin/uv run python scripts/test_ppo_basic_wandb.py
~/.local/bin/uv run python scripts/test_ppo_debug.py
~/.local/bin/uv run python scripts/test_isaac_env_simple.py
```

### 3. Next: Full Isaac Sim Setup (Required for Robot Environments)
```bash
# Isaac Lab alone is not enough - we need the full Isaac Sim for robot environments
# Option 1: Install Isaac Sim standalone (recommended for headless)
cd ~/isaac-lab
./isaaclab.sh --install-isaaclab # This installs the full simulator

# Option 2: Use Docker container with Isaac Sim
./isaaclab.sh --docker

# Option 3: Install Omniverse Launcher and Isaac Sim GUI
# Download from: https://www.nvidia.com/en-us/omniverse/download/
```

### 4. Study Integration Examples
- Review `~/isaac-lab/source/isaaclab_tasks/isaaclab_tasks/manager_based/`
- Understand how rsl_rl expects actor-critic modules
- Check humanoid environment setup in `classic/humanoid/`

## This Week's Goals

### 1. Create GR00T Wrapper ✅ (COMPLETED)
```python
# gr00t-rl/algorithms/gr00t_wrapper.py
class GR00TActorCritic(nn.Module):
    """Wrapper to make GR00T compatible with Isaac Lab RL."""
    def __init__(self, gr00t_model_path, obs_dim, freeze_backbone=True):
        super().__init__()
        # Load GR00T
        self.gr00t = Gr00tPolicy.from_pretrained(
            gr00t_model_path,
            embodiment_tag="GR1",
            freeze_backbone=freeze_backbone
        )
        
        # Add lightweight critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
    def act(self, observations):
        """Get actions from GR00T policy."""
        return self.gr00t(observations)
        
    def evaluate(self, observations):
        """Get values from critic."""
        return self.critic(observations)
```

### 2. Integration Test (1-2 days)
- Start with `Isaac-Reach-Franka-v0` (simple task)
- Verify GR00T forward pass works
- Check action space compatibility
- Monitor memory usage

### 3. Baseline Experiments (2-3 days)
Run PPO with:
1. Default Isaac Lab policy (baseline)
2. Our PPO + small network
3. Our PPO + GR00T (frozen)

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

### Experiment 1: Scaling Test
```bash
# Test how performance scales with parallel envs
for num_envs in 1 4 16 64 256:
    python train_ppo_v2.py --env Isaac-Reach-Franka-v0 --num-envs $num_envs
```

### Experiment 2: Frozen vs Unfrozen
```python
# Compare different freezing strategies
configs = [
    {"freeze_backbone": True, "freeze_action_head": False},
    {"freeze_backbone": False, "train_lora_only": True},
    {"freeze_backbone": False, "freeze_action_head": False}
]
```

### Experiment 3: GRPO Ablations
```python
# Test GRPO hyperparameters
grpo_configs = [
    {"num_rollouts": 4, "normalize_advantages": True},
    {"num_rollouts": 8, "normalize_advantages": True},
    {"num_rollouts": 16, "normalize_advantages": False}
]
```

## Success Metrics

### Week 1 Success:
- [x] PPO training runs without crashes ✅
- [x] GR00T wrapper integrated (gr00t_wrapper.py created) ✅
- [x] Training script ready (train_isaac_ppo.py) ✅
- [x] Environment adapters created (isaac_gym_wrapper.py) ✅
- [ ] Baseline performance recorded (pending Isaac Sim)
- [x] Memory usage acceptable (<1GB for basic tests) ✅

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