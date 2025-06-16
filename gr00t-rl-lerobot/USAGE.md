# GR00T-RL-LeRobot Usage Guide

This guide explains how to use the GR00T + LeRobot integration for RL training on robotic manipulation tasks.

## Overview

We've created an integration that allows your fine-tuned GR00T-N1.5-3B model to be trained with reinforcement learning on the Gymnasium-Robotics Fetch Pick and Place environment.

### Key Components

1. **Environment Wrapper** (`environments/fetch_wrapper.py`):
   - Adapts Fetch environment to match SO-101 robot interface
   - Converts between observation and action spaces

2. **Policy Wrapper** (`policies/gr00t_policy.py`):
   - Wraps GR00T model to work with LeRobot's policy interface
   - Requires Isaac-GR00T to be installed at ~/pippa/Isaac-GR00T

3. **SAC Training** (`scripts/train_sac_fetch.py`):
   - Implements Soft Actor-Critic for online RL training
   - Includes replay buffer, evaluation, and WandB logging

4. **Model Loader** (`utils/gr00t_loader.py`):
   - Downloads fine-tuned GR00T from WandB artifacts
   - Provides integration guidance

## Installation

```bash
# Install dependencies
cd gr00t-rl-lerobot
pip install -e .

# Or install manually
pip install lerobot gymnasium-robotics torch wandb
```

## Quick Start

### 1. Test Environment Integration

```bash
cd scripts
# Test original Cartesian approach
python test_fetch_integration.py

# Test different adaptation approaches
python test_fetch_adaptations.py
```

This will:
- Test the Fetch wrapper (Cartesian-only)
- Test the coupled joints approach (6-DoF)
- Compare different adaptation strategies
- Run sample episodes with visualization

### 2. Train with SAC

```bash
# Make sure you're logged into WandB
wandb login

# Run SAC training
python train_sac_fetch.py
```

Training will:
- Use Fetch Pick and Place environment
- Train GR00T policy with SAC
- Log metrics to WandB
- Save checkpoints every 10k steps

### 3. Load Your Fine-tuned GR00T Model

The policy automatically loads your fine-tuned model:

```python
from utils.gr00t_loader import GR00TModelLoader

# Load model from WandB
loader = GR00TModelLoader()
model, components = loader.load_for_rl(
    artifact_name="wild-ai/pippa/gr00t-sft-so100_dualcam-bs32:v0"
)
```

## Current Status: Actual GR00T Model Integrated! ✅

The implementation now loads and uses your actual fine-tuned GR00T model from WandB. The policy automatically:
- Downloads your checkpoint from `wild-ai/pippa/gr00t-sft-so100_dualcam-bs32:v0`
- Loads the fine-tuned action head weights
- Converts between Fetch and GR00T observation formats
- Uses the real diffusion action head for predictions

### Testing the Integration

Run the comprehensive test script:
```bash
cd scripts
python test_groot_integration.py
```

This will:
1. Load your fine-tuned GR00T model from WandB
2. Test inference on dummy data
3. Run a full episode in the Fetch environment
4. Save trajectory visualizations

## Environment Adaptation Approaches

### Approach 1: Cartesian-Only (Default)
The original wrapper uses 4D Cartesian actions:
- MuJoCo handles 7-DoF IK internally
- Quick to start, no modifications needed
- Good for visual-based learning

### Approach 2: Joint Coupling (New)
The coupled wrapper simulates 6-DoF:
- Couples shoulder roll + upperarm roll
- Better matches SO-101 kinematics
- Use with `make_fetch_so101_coupled_env()`

Example:
```python
# Cartesian approach (quick start)
env = make_fetch_so101_env()

# Coupled joints approach (better fidelity)
env = make_fetch_so101_coupled_env(couple_joints=True)
```

## Remaining Limitations

### 1. Kinematic Differences
- Fetch has 7-DoF, SO-101 has 6-DoF
- Workspace and joint limits differ
- May affect sim-to-real transfer

### 2. Visual Sim-to-Real Gap
- Simulated cameras vs real cameras
- Lighting and texture differences
- Domain adaptation may be needed

## Next Steps

### 1. Run the Tests

First, test that everything is working:
```bash
cd gr00t-rl-lerobot
python scripts/test_groot_integration.py
```

### 2. Start RL Training

Once tests pass, you can start SAC training:
```bash
python scripts/train_sac_fetch.py
```

This will:
- Use your fine-tuned GR00T as the actor network
- Train critic networks to estimate values
- Improve the policy through online interaction

### 3. Implement Proper Action Mapping (Optional Enhancement)

Create inverse kinematics for SO-101:

```python
class SO101Kinematics:
    def cartesian_to_joint(self, cartesian_action):
        # Implement IK solver
        pass
    
    def joint_to_cartesian(self, joint_positions):
        # Implement FK solver
        pass
```

### 4. Create Custom SO-101 Environment

Instead of using Fetch, create a proper SO-101 simulation:

```python
# Using MuJoCo or Isaac Gym
class SO101PickPlaceEnv(gym.Env):
    def __init__(self):
        # Load SO-101 URDF/XML
        # Set up table and objects
        pass
```

## Alternative Approach: Direct RL Training

If LeRobot integration is too complex, you can use standard RL libraries:

```python
import stable_baselines3 as sb3
from stable_baselines3 import SAC

# Wrap GR00T as SB3 policy
class GR00TSB3Policy(sb3.common.policies.BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gr00t = load_gr00t_model()
    
    def forward(self, obs):
        return self.gr00t.predict(obs)

# Train with SAC
model = SAC(GR00TSB3Policy, env, verbose=1)
model.learn(total_timesteps=100000)
```

## Experiment Ideas

1. **Reward Shaping**: Add intermediate rewards for:
   - Approaching the object
   - Successful grasping
   - Lifting the object
   - Moving toward goal

2. **Curriculum Learning**: Start with:
   - Object always in same position
   - Then randomize object position
   - Then add distractors
   - Finally, multiple objects

3. **Language Conditioning**: Use different instructions:
   - "Pick the red cube"
   - "Place the object on the left"
   - "Stack the blocks"

4. **Multi-Task Learning**: Train on multiple tasks:
   - Pick and place
   - Pushing
   - Stacking

## Troubleshooting

### Import Errors
```bash
# If LeRobot not found
pip install lerobot

# If Gymnasium-Robotics not found
pip install gymnasium-robotics
```

### CUDA/GPU Issues
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### WandB Authentication
```bash
# Login to WandB
wandb login

# Set project programmatically
wandb.init(project="your-project", entity="your-entity")
```

## Contributing

To improve this integration:

1. Implement actual GR00T model loading
2. Add proper SO-101 kinematics
3. Create better action space mapping
4. Add more sophisticated reward functions
5. Implement curriculum learning

## References

- [GR00T SFT Training](../research_journal/gr00t_sft.md)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [Gymnasium-Robotics](https://robotics.farama.org/)
- [SAC Paper](https://arxiv.org/abs/1801.01290)