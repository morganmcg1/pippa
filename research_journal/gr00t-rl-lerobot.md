# GR00T-RL-LeRobot Research Journal

## Overview
This document tracks the integration of our fine-tuned GR00T-N1.5-3B model with LeRobot for reinforcement learning on SO-101 robotic tasks.

## Project Goals
1. Integrate fine-tuned GR00T model (from gr00t_sft.md) with LeRobot framework
2. Implement RL training using SAC (Soft Actor-Critic)
3. Create simulation environment for SO-101 table manipulation tasks
4. Compare RL-enhanced performance vs pure supervised fine-tuning

## Background
- **Fine-tuned Model**: `wild-ai/pippa/gr00t-sft-so100_dualcam-bs32:v0` (WandB artifact)
- **Model Performance**: MSE 11.15 on SO-101 table cleanup evaluation
- **Architecture**: GR00T-N1.5-3B with diffusion action head, dual camera input
- **Task**: Table cleanup - pick and place objects into containers

## Technical Architecture

### GR00T Model Details
- **Vision Encoder**: Eagle VLM (frozen during SFT)
- **Embodiment Adapter**: NEW_EMBODIMENT projector for SO-101
- **Action Head**: Diffusion transformer outputting 6 joint positions
- **Inputs**: 
  - Dual RGB cameras (front + wrist) at 224x224
  - 6-DoF joint states
  - Language instructions
- **Outputs**: 6 joint position targets

### LeRobot Integration Plan
1. **Policy Wrapper**: Create `GR00TPolicy` inheriting from `PreTrainedPolicy`
2. **Environment**: Build SO-101 simulation using MuJoCo/Isaac Gym
3. **Training**: Use LeRobot's SAC implementation for online RL
4. **Evaluation**: Compare with baseline SFT model

## Implementation Log

### Initial Setup - 2025-06-16_15:00
- Created project directory structure
- Planning integration approach based on LeRobot documentation
- Key findings from research:
  - LeRobot supports custom policies via `PreTrainedPolicy` interface
  - SAC implementation available for online RL
  - Need to create custom environment for SO-101
  - HIL-SERL explicitly mentions SO-101 support

### Next Steps
1. Install LeRobot and dependencies
2. Set up Gymnasium-Robotics Fetch environment
3. Implement GR00T policy wrapper with action space adapter
4. Test integration with Fetch Pick and Place

### Environment Choice: Gymnasium-Robotics Fetch - 2025-06-16_15:10
After reviewing options, decided to use Fetch Pick and Place environment:
- **Environment**: `FetchPickAndPlace-v3`
- **Observation Space**: 25D vector + achieved/desired goal
- **Action Space**: 4D (Cartesian displacement + gripper)
- **Advantages**:
  - Already implements pick and place (matches our table cleanup)
  - Well-documented and tested
  - Compatible with standard RL libraries
- **Challenge**: Need to map between:
  - SO-101 joint space (6D) ↔ Fetch Cartesian space (4D)
  - GR00T's dual cameras ↔ Fetch's state vector

### Implementation Progress - 2025-06-16_15:30
Successfully created core components:

1. **Fetch Wrapper** (`environments/fetch_wrapper.py`):
   - Converts Fetch observations to GR00T format (dual cameras + state)
   - Maps SO-101 6D joint actions to Fetch 4D Cartesian actions
   - Provides language instructions for goal conditioning
   - Simulates dual camera views from single Fetch render

2. **GR00T Policy** (`policies/gr00t_policy.py`):
   - Implements LeRobot's `PreTrainedPolicy` interface
   - Placeholder architecture (to be replaced with actual GR00T)
   - Handles dual camera + state observations
   - Predicts action horizons (16 steps) for temporal consistency

3. **Test Script** (`scripts/test_fetch_integration.py`):
   - Verifies environment wrapper functionality
   - Tests policy inference
   - Runs integrated episode with visualization

### Key Design Decisions:
- **Action Mapping**: Simple linear mapping from joint space to Cartesian
  - Future: Use proper SO-101 forward kinematics
- **Camera Simulation**: Create wrist view by cropping/zooming front view
  - Future: Use multiple camera angles from Fetch
- **Policy Architecture**: Dummy networks for testing
  - TODO: Load actual GR00T weights from WandB

### Next Implementation Steps:
1. Load actual GR00T model from WandB artifact
2. Implement proper diffusion action head
3. Create SAC training loop with LeRobot
4. Add reward shaping for table cleanup

### Complete Implementation - 2025-06-16_15:45
Created full implementation framework:

1. **Environment Integration** (`environments/fetch_wrapper.py`):
   - ✅ Maps Fetch Pick and Place to SO-101 interface
   - ✅ Converts observations (state → dual cameras + joints)
   - ✅ Action space mapping (6D joints → 4D Cartesian)
   - ✅ Goal-conditioned with language instructions

2. **Policy Wrapper** (`policies/gr00t_policy.py`):
   - ✅ Implements LeRobot's PreTrainedPolicy interface
   - ✅ Handles dual camera + state observations
   - ✅ Action horizon prediction (16 steps)
   - ✅ Placeholder networks (ready for GR00T integration)

3. **SAC Training** (`scripts/train_sac_fetch.py`):
   - ✅ Complete SAC implementation
   - ✅ Replay buffer and soft Q-learning
   - ✅ WandB integration for tracking
   - ✅ Evaluation and checkpointing

4. **Model Loading** (`utils/gr00t_loader.py`):
   - ✅ WandB artifact downloading
   - ✅ Component extraction (action head, modality config)
   - ✅ Integration guide for actual GR00T model

### Key Insights:
- **Fetch Environment Works Well**: Good proxy for SO-101 pick-and-place
- **Action Space Challenge**: Need better joint ↔ Cartesian mapping
- **GR00T Integration**: Requires Isaac-GR00T repo for full model loading

### Limitations & Future Work:
1. **Placeholder Networks**: Currently using dummy networks instead of actual GR00T
   - Need Isaac-GR00T dependencies for real model
   - Diffusion action head implementation missing

2. **Action Mapping**: Simple linear mapping is suboptimal
   - Should implement proper SO-101 forward kinematics
   - Or train a learned mapping network

3. **Camera Simulation**: Basic cropping for dual views
   - Could use multiple Fetch camera angles
   - Or render from different viewpoints

4. **Reward Shaping**: Currently using sparse Fetch rewards
   - Could add intermediate rewards for grasping, lifting
   - Language-conditioned rewards for specific objects

## Key Challenges & Solutions

### Challenge 1: No Built-in SO-101 Simulation
**Solution**: Use Gymnasium-Robotics Fetch Pick and Place environment as a proxy for SO-101
- Fetch has 7-DoF arm (similar to SO-101's 6-DoF)
- Pick and place task matches our table cleanup objective
- Well-tested and documented environment
- Action space: 4D (dx, dy, dz, gripper) - will need mapping from SO-101's 6 joint positions

### Challenge 2: GR00T Integration
**Solution**: Wrap GR00T model to match LeRobot's policy interface

### Challenge 3: Observation Format Mismatch
**Solution**: Create adapter to convert between GR00T and LeRobot formats

## References
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [HIL-SERL Paper](https://arxiv.org/abs/2410.21845)
- [GR00T SFT Results](./gr00t_sft.md)
- [PPO Implementation Learnings](./ppo_implementation_learnings.md)

---
*Journal started: 2025-06-16_15:00*