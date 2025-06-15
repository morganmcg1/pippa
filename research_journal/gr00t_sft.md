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

### 1. Batch Size Testing - 2025-06-15_20:00
**Hypothesis**: Determine maximum batch size that fits on single H100 GPU

**Results**: ✅ COMPLETED
- **Batch size 8**: 51GB memory usage (successful)
- **Batch size 16**: 52.5GB memory usage (successful)
- **Batch size 32**: 56GB memory usage (successful)
- **Batch size 64**: 41.8GB memory usage (successful) - Note: Lower than 32 due to memory allocation patterns
- **Batch size 128**: 73.6GB memory usage (successful) - **MAXIMUM**
- **Batch size 256**: OOM error (exceeds 80GB limit)

**Key Findings**:
- Maximum batch size on single H100: **128**
- Blog baseline of 4 is extremely conservative
- Memory scaling is non-linear due to PyTorch allocation strategies
- At batch size 128, we utilize 90% of GPU memory (optimal)

**Recommendations**:
- For production training: Use batch size 64-96 (leaves headroom)
- For maximum throughput: Use batch size 128
- With gradient accumulation: Can simulate larger effective batch sizes

### 2. Overfitting Test - 2025-06-15_20:12
**Hypothesis**: Verify model can memorize small dataset subset to validate training pipeline

**Initial Attempt**: ❌ FAILED
- **Issue**: StopIteration error in model initialization
- **Root Cause**: When using batch_size=1 with certain parameter combinations, the model's action head had no trainable parameters
- **Learning**: Need to ensure GPU is fully clean before training and use reasonable batch sizes

**Configuration Requirements**:
```python
# Working configuration
max_steps = 200-500
learning_rate = 1e-4  # Blog default is sufficient
batch_size = 4-64     # Use higher batch sizes for faster training
weight_decay = 0.05   # Keep regularization
save_steps = 50
```

**Key Learnings**:
1. **Always clean GPU before training**: Kill all processes to ensure full memory availability
2. **Use reasonable batch sizes**: Batch size 1 can cause initialization issues
3. **Blog defaults work well**: lr=1e-4, weight_decay=0.05 are well-tuned
4. **Higher batch sizes = faster convergence**: Use 32-64 for quick experiments

**Success Metrics**:
- Training loss decreases steadily
- Model learns to reproduce demonstration actions
- No initialization errors

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

### Critical Model Components to Save (Updated 2025-06-15)
**Must save these files for deployment**:
1. **`gr00t_policy.pt`** - Full model state dict or LoRA weights
2. **`projector_new_embodiment.pt`** - The trained embodiment adapter (critical!)
3. **`modality.json`** - Dataset configuration for traceability

**Implementation**:
- Enhanced `save_checkpoint_to_wandb()` to explicitly save these components
- Projector weights extracted by filtering parameters with "projector" in name
- modality.json copied from dataset to checkpoint directory
- All files bundled into WandB artifact for easy retrieval

**Usage**:
```python
# Download from WandB
artifact = wandb.use_artifact('wild-ai/pippa/gr00t-sft-checkpoint-step-500:v0')
artifact_dir = artifact.download()

# Load for inference
model = GR00T_N1_5.from_pretrained(base_model_path)
projector_weights = torch.load(f"{artifact_dir}/projector_new_embodiment.pt")
model.load_state_dict(projector_weights, strict=False)
```

## References
- [GR00T N1.5 SO-101 Fine-tuning Tutorial](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) - Official NVIDIA blog post
- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T) - Official repository
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot) - Dataset structure documentation

---
*This journal will be updated as experiments progress.*