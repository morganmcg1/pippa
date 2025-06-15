# GR00T SFT Research Journal

## Overview
This document contains all research notes, experiment results, and learnings from our GR00T (Group Robotics Transformer) Supervised Fine-Tuning experiments using the SO-101 table cleanup dataset and the GR00T-N1.5-3B model.

## Task Description: SO-101 Table Cleanup

### Goal
Fine-tune the NVIDIA GR00T-N1.5-3B foundation model to perform table cleanup tasks with the LeRobot SO-101 robot arm using supervised learning on demonstration data.

### Dataset Details
- **Dataset**: so101-table-cleanup (public demo dataset)
- **Size**: 46,963 frames (1 episode × 2 synchronized cameras × 30 fps ≈ 26 min)
- **Robot**: LeRobot SO-101 - 6-DoF elbow-type arm with parallel gripper
- **Task**: Pick specific desk objects (pens, tape, batteries) and place into pen-holder/bin
- **Modalities**:
  - observation.images.front (640×480×3, 30 fps)
  - observation.images.wrist (640×480×3, 30 fps)
  - observation.state (6 joint positions)
  - action (6 target joint positions)
- **Instruction prompts**: Stored in meta/task_prompts.json (e.g., "Grab tapes and place into pen holder")

### Model Architecture
- **Base Model**: NVIDIA GR00T-N1.5-3B
- **Components**:
  - Vision-Language Encoder: Eagle VLM (frozen by default)
  - Embodiment Projection Layer: ~0.2M params for NEW_EMBODIMENT
  - Diffusion Transformer: Action prediction head
- **Training Strategy**: Fine-tune projector and diffusion head, freeze VLM

## Implementation Details

### Training Infrastructure
- **Script**: `train_gr00t_sft.py` - Enhanced version with WandB integration
- **Launch Script**: `launch_sft.sh` - Unified launcher with configurable parameters
- **Virtual Environment**: Uses Isaac-GR00T's sft_venv
- **GPU**: H100 80GB (ubuntu@192.222.53.15)

### Key Configuration
```python
# From successful blog baseline
learning_rate = 1e-4
batch_size = 4  # Per GPU
max_steps = 10000
data_config = "so100_dualcam"
video_backend = "torchvision_av"
warmup_ratio = 0.05
weight_decay = 0.05

# Model tuning flags
tune_llm = False      # Freeze language model
tune_visual = False   # Freeze vision encoder
tune_projector = True # Train embodiment adapter
tune_diffusion_model = True  # Train action head
```

### Critical Setup Requirements
1. **modality.json**: Already correctly configured in dataset
   - Maps state/action dimensions for 6-DoF arm
   - Defines camera views (front/wrist)
   - Links task descriptions to indices
   
2. **Environment Variables**:
   - PYTHONPATH must include Isaac-GR00T directory
   - VIDEO_BACKEND=torchvision_av
   - WANDB credentials from .env file

## Experiment Log

### Initial Setup and Debug - 2025-06-15_19:45
- **Status**: COMPLETED ✅
- **Run ID**: gbtezszk
- **Configuration**: 
  - 20 steps for quick test
  - Batch size: 1
  - 1 GPU
- **Results**:
  - Training completed successfully!
  - Loss decreased from 0.7057 to 0.5388
  - Proper gradient flow (6.9 to 10.1 grad norm)
  - WandB logging working correctly
- **Key Fixes Applied**:
  - Fixed virtual environment path
  - Corrected dataset API usage (modality configs + transforms)
  - Fixed boolean parameter passing in launch script
  - Used correct embodiment tag ("new_embodiment" not "NEW_EMBODIMENT")

### Key Learnings from Initial Setup
1. **Dataset API**: LeRobotSingleDataset requires modality_configs and transforms from DATA_CONFIG_MAP
2. **No Subset Support**: Dataset doesn't support .select() method, need different approach for data limiting
3. **Virtual Environment**: Must use Isaac-GR00T's sft_venv, not local
4. **Boolean Parameters**: Use --no-tune-visual syntax, not --tune-visual False
5. **Embodiment Tag**: Case sensitive - use "new_embodiment"

## Next Experiments

### 1. Batch Size Testing
**Hypothesis**: Determine maximum batch size that fits on single H100 GPU

**Approach**:
- Start with batch_size=4 (blog baseline)
- Gradually increase: 8, 16, 32, 64
- Monitor GPU memory usage
- Find OOM threshold

**Expected Outcome**:
- H100 80GB should handle batch_size=16-32
- Blog used batch_size=4 which is conservative
- Larger batches = more stable gradients

### 2. Overfitting Test
**Hypothesis**: Verify model can memorize small dataset subset to validate training pipeline

**Approach**:
- Train for many epochs on tiny dataset
- Use high learning rate (5e-4)
- Disable weight decay and dropout
- Monitor if loss → 0

**Configuration for Overfitting**:
```python
max_steps = 1000
learning_rate = 5e-4
weight_decay = 0.0
batch_size = 1
save_steps = 100
```

**Success Metrics**:
- Training loss < 0.01
- Model perfectly reproduces demonstration actions
- Validates gradient flow and model capacity

## Technical Notes

### Memory Considerations
- Model size: ~3B parameters
- With batch_size=1: ~25GB GPU memory used
- Expected scaling: ~1.5-2x memory per doubling of batch size
- Consider gradient checkpointing if needed

### Evaluation Protocol
- No standard evaluation metric for SO-101 dataset
- Can implement:
  - Action prediction MSE
  - Trajectory completion accuracy
  - Visual similarity of predicted vs ground truth

### WandB Integration
- Artifacts logged to: wild-ai/pippa
- Tags: ["gr00t-sft", "so101-table-cleanup"]
- Model checkpoints saved every N steps
- Consider implementing video generation of predicted trajectories

## References
- [GR00T N1.5 SO-101 Fine-tuning Tutorial](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) - Official NVIDIA blog post
- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T) - Official repository
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot) - Dataset structure documentation

---
*This journal will be updated as experiments progress.*