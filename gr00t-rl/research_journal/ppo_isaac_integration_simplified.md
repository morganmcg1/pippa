# Simplified PPO + Isaac Lab Integration Plan

Date: 2025-01-15
Author: Claude

## New Approach: Build from Working Code

After reviewing the codebase and CLAUDE.md, I'm taking a simpler approach:
1. Start from the working `test_ppo_wandb.py` script
2. Gradually add Isaac Lab functionality
3. Use `uv` for all dependency management
4. Test each step before proceeding

## Current Working State

### What's Already Working
- `test_ppo_wandb.py` successfully trains PPO on Pendulum-v1
- WandB logging is functional
- PPO V2 implementation with all 37 details is solid
- Environment normalization works
- Multi-modal support tested

### Key Insight
Instead of forcing Isaac Lab integration, let's:
1. First get PPO working on more Gym environments
2. Create a simple Isaac Lab wrapper that looks like a Gym environment
3. Test with Cartpole/Pendulum in Isaac Lab
4. Only then integrate GR00T

## Implementation Steps

### Step 1: Test PPO on More Environments (Today)
Create `scripts/test_ppo_basic_envs.py`:
- Test on CartPole-v1 (discrete)
- Test on Pendulum-v1 (continuous) ✓
- Test on Ant-v4 (multi-dimensional)
- Verify all work with our PPO

### Step 2: Create Minimal Isaac Lab Wrapper (Today)
Create `environments/isaac_gym_wrapper.py`:
```python
class IsaacGymWrapper(gym.Env):
    """Wraps Isaac Lab environments to look like standard Gym envs."""
    
    def __init__(self, env_name):
        # Load Isaac Lab env if available
        # Otherwise fallback to mock
        pass
    
    def reset(self):
        # Return numpy array like Gym
        pass
    
    def step(self, action):
        # Convert to/from Isaac Lab format
        # Return (obs, reward, done, info) like Gym
        pass
```

### Step 3: Test Isaac Lab Environments (This Week)
1. Install Isaac Lab dependencies properly using `uv`
2. Test with Isaac Lab's Cartpole first
3. Move to Pendulum, then Ant
4. Use our existing `train_ppo_v2.py` script

### Step 4: Add GR00T Support (Next Week)
1. Fix dtype issues in `gr00t_wrapper.py`
2. Test with mock GR00T first
3. Load actual GR00T with frozen backbone
4. Monitor memory usage carefully

## Key Files to Create/Modify

### 1. `scripts/test_ppo_basic_envs.py`
Based on `test_ppo_wandb.py` but tests multiple environments

### 2. `environments/isaac_gym_wrapper.py`
Simple wrapper to make Isaac Lab envs compatible with our PPO

### 3. `scripts/train_isaac_simple.py`
Minimal training script using our PPO with Isaac Lab envs

### 4. Update `pyproject.toml`
Add Isaac Lab dependencies properly:
```toml
[project.optional-dependencies]
isaac = [
    # Add specific Isaac Lab packages here
]
```

## Commands to Run

### Local Testing
```bash
cd gr00t-rl

# Test basic environments
uv run python scripts/test_ppo_basic_envs.py

# Test with Isaac wrapper
uv run python scripts/test_isaac_simple.py --env Isaac-Cartpole-v0
```

### Remote Testing (GPU Machine)
```bash
# Create new tmux session
ssh ubuntu@192.222.53.15 "tmux new-session -d -s ppo_isaac"

# Run tests
ssh ubuntu@192.222.53.15 "tmux send-keys -t ppo_isaac 'cd ~/pippa/gr00t-rl && uv run python scripts/test_ppo_basic_envs.py' Enter"
```

## Success Criteria

### Today
- [ ] PPO works on CartPole, Pendulum, Ant
- [ ] Basic Isaac Lab wrapper created
- [ ] Can import Isaac Lab (even if mock)

### This Week  
- [ ] Isaac Lab Cartpole training
- [ ] Isaac Lab Pendulum training
- [ ] >5k steps/sec on GPU
- [ ] Memory usage <10GB

### Next Week
- [ ] GR00T wrapper tested
- [ ] Frozen backbone training works
- [ ] Memory usage <40GB with GR00T

## Why This Approach is Better

1. **Incremental**: Each step is testable
2. **Uses Working Code**: Builds on test_ppo_wandb.py
3. **Follows CLAUDE.md**: Uses `uv` for dependencies
4. **Simpler**: No rsl_rl complexity
5. **Debuggable**: Can test each component

## Next Immediate Action

Create `test_ppo_basic_envs.py` to verify our PPO works on multiple standard environments before attempting Isaac Lab integration.