# GR00T RL Fine-tuning

This directory contains implementations for reinforcement learning fine-tuning of NVIDIA's GR00T 1.5 robotic foundation model.

## Overview

We implement two RL algorithms:
1. **PPO (Proximal Policy Optimization)** - Standard approach with a critic network
   - Original implementation: `algorithms/ppo_gr00t.py`
   - **Improved V2 implementation**: `algorithms/ppo_gr00t_v2.py` (recommended)
2. **GRPO (Group Relative Policy Optimization)** - Critic-free approach using relative rewards

## Recent Updates

### PPO V2 Implementation (Latest)
We've created an improved PPO implementation that incorporates all 37 implementation details from [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) and follows CleanRL best practices.

Key improvements in V2:
- **Orthogonal weight initialization** with correct scaling
- **Adam optimizer with epsilon=1e-5** (not default 1e-8)
- **Learning rate annealing** to 0 over training
- **Advantage normalization at batch level** (not mini-batch)
- **Value function clipping** option
- **Proper GAE implementation** with correct bootstrapping
- **Gradient clipping** with configurable max norm
- **Early stopping** based on KL divergence
- **Vectorized environments** for parallel training
- **Comprehensive normalization** for observations and rewards
- **Multi-modal observation support** for vision + proprioception + language

## Directory Structure

```
gr00t-rl/
├── algorithms/       # RL algorithm implementations
│   ├── ppo_gr00t.py # Original PPO wrapper
│   ├── ppo_gr00t_v2.py # Improved PPO with 37 details (recommended)
│   └── grpo_gr00t.py # GRPO adaptation for robotics
├── environments/     # Simulation environment wrappers
│   ├── isaac_env.py  # Isaac Lab environment wrapper
│   └── vec_isaac_env.py # Vectorized environment wrapper
├── configs/          # Configuration files
│   ├── ppo_config.py # Original PPO config
│   ├── ppo_config_v2.py # Improved PPO config with all options
│   └── grpo_config.py # GRPO hyperparameters
├── utils/            # Common utilities
│   ├── rewards.py    # Reward functions
│   ├── transforms.py # Observation preprocessing
│   ├── logging.py    # Enhanced logging with GRPO health metrics
│   ├── networks.py   # Neural network architectures
│   ├── buffers.py    # PPO rollout buffer with proper GAE
│   └── normalization.py # Running mean/std normalization
├── scripts/          # Training and evaluation scripts
│   ├── train_ppo.py  # Original PPO training
│   ├── train_ppo_v2.py # Improved PPO training (recommended)
│   ├── train_grpo.py # GRPO training script
│   └── test_ppo_v2.py # Test script for PPO V2
└── README.md         # This file
```

## Key Features

- **Modular design** - Easy to swap algorithms and environments
- **Memory efficient** - Supports gradient accumulation and layer freezing
- **WandB integration** - Comprehensive experiment tracking with GRPO health metrics
- **Isaac Lab compatible** - Works with NVIDIA's robot simulation platform
- **All 37 PPO implementation details** - Based on extensive research
- **Vectorized training** - Parallel environment execution
- **Multi-modal support** - Vision, proprioception, and language inputs

## Getting Started

### Quick Test (PPO V2)
```bash
# Test on simple environments
python scripts/test_ppo_v2.py
```

### Training with PPO V2 (Recommended)
```bash
# Train on Isaac Lab environment
python scripts/train_ppo_v2.py \
    --env FrankaCubeLift-v0 \
    --num-envs 8 \
    --total-timesteps 1000000 \
    --track  # Enable WandB logging
```

### Training with Original PPO
```bash
python scripts/train_ppo.py
```

### Training with GRPO
```bash
python scripts/train_grpo.py
```

## Installation

```bash
# Clone Isaac-GR00T if not already done
git clone https://github.com/NVIDIA/Isaac-GR00T

# Create virtual environment
cd gr00t-rl
uv venv
source .venv/bin/activate

# Install dependencies
cd ../gr00t-tuning/Isaac-GR00T
uv pip install -e ".[base]"
uv pip install --no-build-isolation flash-attn==2.7.1.post4

# Install RL dependencies
uv pip install gymnasium stable-baselines3 wandb python-dotenv
uv pip install torch torchvision  # If not already installed
```

## Tasks

We start with simple verifiable tasks:
1. **Pick and Place** - Move objects to target locations
2. **Push to Goal** - Push objects to specific positions
3. **Stack Blocks** - Stack objects in order

## PPO V2 Configuration

The improved PPO implementation exposes all important hyperparameters:

```python
# Key hyperparameters (configs/ppo_config_v2.py)
learning_rate: float = 3e-4
anneal_lr: bool = True  # Linear annealing to 0
adam_epsilon: float = 1e-5  # Different from default!
n_steps: int = 2048  # Steps per env before update
batch_size: int = 64  # Mini-batch size
n_epochs: int = 10  # Update epochs
gamma: float = 0.99  # Discount factor
gae_lambda: float = 0.95  # GAE lambda
clip_range: float = 0.2  # PPO clip parameter
normalize_advantage: bool = True  # At batch level
max_grad_norm: float = 0.5  # Gradient clipping
```

## WandB Logging

Both PPO and GRPO implementations include comprehensive logging:

### PPO Metrics
- `train/policy_loss`, `train/value_loss`, `train/entropy_loss`
- `train/approx_kl`, `train/clipfrac`, `train/explained_variance`
- `train/learning_rate`, `train/grad_norm`
- `charts/episodic_return`, `charts/episodic_length`, `charts/SPS`

### GRPO Metrics
- All PPO metrics plus:
- `train/frac_zero_std` - Critical GRPO health metric
- `train/return_diversity` - Diversity of returns
- `train/kl_divergence` - When using reference policy
- `grpo/mean_reward`, `grpo/reward_std`

## Key Differences from Language Model RL

- **Continuous actions** instead of discrete tokens
- **Multi-step episodes** instead of single-turn generation
- **Dense state representation** from vision and proprioception
- **Physical constraints** and safety considerations
- **Gaussian policy** for continuous control (temporary until flow matching understood)

## Current Status & Roadmap

### ✅ Completed
- PPO V2 implementation with all 37 ICLR details
- GRPO implementation for continuous control
- Comprehensive logging and monitoring
- Modular architecture ready for integration

### 🚧 In Progress (Phase 2: Isaac Lab Integration)
Based on research findings:
- No official GR00T+PPO example exists yet
- Need to create ~200 lines of glue code
- Isaac Lab has the infrastructure via rsl_rl
- See `INTEGRATION_ROADMAP.md` for details

### 📋 Upcoming Phases
3. **GR00T Integration**: Wrap GR00T with critic, handle action space
4. **GRPO Experiments**: Test on verifiable robotics tasks
5. **Advanced Features**: LoRA fine-tuning, curriculum learning

### 🔬 Key Research Findings
- **GRPO can work for robotics** but has trade-offs
- Best for tasks with clear success metrics
- Standard PPO recommended as baseline
- See `research_journal/grpo_robotics_analysis.md` for full analysis

### 📖 Key Documents
- `INTEGRATION_ROADMAP.md` - Complete integration strategy
- `NEXT_STEPS.md` - Concrete action items
- `research_journal/` - Detailed implementation notes