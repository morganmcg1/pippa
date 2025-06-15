# GR00T Baseline Research Journal

## Overview
GR00T (Generalized Robotic Operation Optimization Technology) N1.5 is NVIDIA's foundation model for humanoid robots. Unlike GRPO (language model training), GR00T uses supervised fine-tuning on demonstration data.

## Key Differences from GRPO
- **Supervised Learning**: Fine-tunes on robot demonstration data (not RL with rewards)
- **Multi-modal**: Processes video, language instructions, and action sequences
- **Large Model**: 3B parameters requiring significant GPU memory
- **Specific Data Format**: SO-101 format with modality.json configuration

## Experiment Log

### Initial Experiments Completed (2025-06-15)
Based on analysis of 3 WandB runs requested by user:

1. **Demo PickNPlace (klky5smm)** - 2025-06-14_23:58 - Run ID: klky5smm
   - Learning rate: 5e-4
   - Batch size: 1
   - Final loss: 0.4574 at 200 steps (still dropping)
   - Minimal warmup (0.01)

2. **Extended Training (0rtd7un9)** - 2025-06-15_00:07 - Run ID: 0rtd7un9
   - Learning rate: 1e-3 (too high)
   - Batch size: 1
   - Final loss: 1.2572 at 2000 steps
   - Conclusion: LR too high, caused instability

3. **SO-101 Dataset (d9pawzt8)** - 2025-06-15_00:18 - Run ID: d9pawzt8
   - Learning rate: 1e-4
   - Batch size: 32
   - Loss: 0.0266 at ~8000 steps (best performance)
   - Uses full SO-101 dataset
   - Default warmup and weight decay

### Follow-up Experiments Completed (2025-06-15)

1. **Best Config Replication (6jzsan5d)** - 2025-06-15_01:11 - Run ID: 6jzsan5d
   - Learning rate: 5e-4
   - Steps: 1000
   - Final loss: 0.1904
   - Attempted to replicate klky5smm success

2. **Lower LR Experiment (vtty1l1b)** - 2025-06-15_01:11 - Run ID: vtty1l1b
   - Learning rate: 2.5e-4
   - Steps: 1500
   - Final loss: 0.0964 ✓
   - **Best demo overfitting result!**

## Key Findings (Updated 2025-06-15)

### Best Performing Configurations (Ranked by Loss)
1. **ch7f65x3**: LR 1e-4, batch 1, 2000 steps → **Loss 0.0585** ✓✓
2. **fwtchfxv**: LR 5e-4, batch 8, 1000 steps → **Loss 0.0702** ✓
3. **vtty1l1b**: LR 2.5e-4, batch 1, 1500 steps → Loss 0.0964
4. **6jzsan5d**: LR 5e-4, batch 1, 1000 steps → Loss 0.1904
5. **dvlv66is**: LR 5e-4, batch 1, all components → Loss 0.2507

### Learning Rate Insights
- **Optimal for demo overfitting**: 1e-4 with extended training (2000 steps)
- **LR hierarchy**: 1e-4 > 5e-4 > 2.5e-4 > 1e-3
- **Key insight**: Lower LR + more steps beats higher LR + fewer steps

### Batch Size Impact
- **Batch size 8 > batch size 1** for same LR and steps
- Batch 8 achieved 0.0702 vs batch 1 achieved 0.1904 (63% improvement!)
- Better gradient estimates with larger batches

### Component Fine-tuning
- **Only tune projector + diffusion** (blog default is correct)
- Full fine-tuning (all components) significantly hurts performance
- Smaller datasets need fewer trainable parameters

### Current Hypothesis (2025-06-15)

#### Learning Rate Sweet Spot for Demo Overfitting
The lower learning rate experiment (2.5e-4) outperformed the "best" config (5e-4), achieving loss 0.0964 vs 0.1904. This suggests:

1. **Optimal LR Range**: For demo overfitting on 5 episodes, the optimal learning rate appears to be between 1e-4 and 5e-4, specifically around 2.5e-4
2. **LR Hierarchy Observed**: 2.5e-4 > 5e-4 > 1e-3 (for demo data)
3. **Different Optimal for Different Data**:
   - Full SO-101 dataset: 1e-4 (blog default) works best
   - Demo overfitting: 2.5e-4 (middle ground) works best
   - Aggressive overfitting: 5e-4 converges faster but higher final loss

#### Batch Size Impact Hypothesis
Current experiments suggest batch size 32 > 1 for full dataset, but we haven't tested intermediate batch sizes for demo data. Testing batch size 8 to see if it provides a good balance.

#### Component Fine-tuning Hypothesis
All successful runs only fine-tuned projector + diffusion model. Testing full fine-tuning (including LLM + visual) to see if it helps or hurts overfitting on tiny dataset.

## Experiments Completed (2025-06-15) - Latest Batch

### 1. Batch Size Experiment - 2025-06-15_01:51 - Run ID: fwtchfxv ✓
- **Hypothesis**: Batch size 8 might achieve better convergence than batch size 1 for demo overfitting
- **Config**: LR 5e-4, batch 8, 1000 steps
- **Result**: **Loss 0.0702** - Best result so far!
- **Conclusion**: Batch size 8 significantly outperforms batch size 1 (0.0702 vs 0.1904)
- **Note**: There was also a failed run (btj3964a) that stopped at 200 steps

### 2. Very Low LR Experiment - 2025-06-15_01:58 - Run ID: ch7f65x3 ✓
- **Hypothesis**: LR 1e-4 (successful for SO-101) might also work well for demo overfitting with extended training
- **Config**: LR 1e-4, batch 1, 2000 steps
- **Result**: **Loss 0.0585** - Second best result!
- **Conclusion**: Very low LR (1e-4) with extended training works excellently

### 3. Full Fine-tuning Experiment - 2025-06-15_02:00 - Run ID: dvlv66is ✓
- **Hypothesis**: Fine-tuning all components (LLM + visual + projector + diffusion) might enable better overfitting
- **Config**: LR 5e-4, batch 1, 1000 steps, all components
- **Result**: Loss 0.2507 - Worse than projector+diffusion only
- **Conclusion**: Full fine-tuning hurts performance on small datasets

## Experiments in Progress (2025-06-15)

### 4. Blog Settings on Demo Data (GPU 0) - 2025-06-15_02:12 - Run ID: qbk0btb6
- **Hypothesis**: Blog-exact hyperparameters (LR 1e-4, warmup 0.05, WD 1e-5) might work well on demo data
- **Config**: LR 1e-4, batch 5, 2000 steps, blog defaults
- **Current**: Loss 0.0695 at step 1000 (50% complete)
- **Status**: RUNNING - Using 25.1GB VRAM on GPU 0
- **Note**: Performing well, similar to batch size 8 experiment

## Optimal Training Configuration (Based on Analysis)

### Best Overall Configuration (SO-101 Full Dataset)
**Run**: d9pawzt8 - Achieved loss 0.0144 at step 8600
- **Learning rate**: 1e-4 (blog default)
- **Batch size**: 32 (blog default)
- **Dataset**: Full SO-101 dataset
- **Warmup ratio**: 0.05 (default)
- **Weight decay**: 1e-5 (default)
- **Max steps**: 10000
- **Components**: tune_projector=True, tune_diffusion_model=True

### Best Overfitting Configuration (Demo Data)
**Run**: vtty1l1b - Achieved loss 0.0964 in 1500 steps
- **Learning rate**: 2.5e-4 (optimal for demo)
- **Batch size**: 1 (minimal for overfitting)
- **Dataset**: Demo PickNPlace (5 episodes)
- **Warmup ratio**: 0.01 (minimal)
- **Weight decay**: 0.0 (no regularization)
- **Max steps**: 1500
- **Components**: tune_projector=True, tune_diffusion_model=True

### Recommended Configurations:

**For Full SO-101 Training**:
```bash
--learning-rate 1e-4
--batch-size 32
--max-steps 10000
--warmup-ratio 0.05
--weight-decay 1e-5
```

**For Demo Overfitting**:
```bash
--learning-rate 2.5e-4  # Updated based on experiments
--batch-size 1
--max-steps 1500
--warmup-ratio 0.01
--weight-decay 0.0
```

**For Extended Overfitting** (exploring lower LRs):
```bash
--learning-rate 1e-4  # Conservative approach
--batch-size 1
--max-steps 2000  # More steps for slower learning
--warmup-ratio 0.01
--weight-decay 0.0
```

## Technical Learnings

### Environment Setup
- Use `python` directly in activated venv, not `uv run python` (flash-attn build issues)
- CUDA_VISIBLE_DEVICES works for GPU selection
- Each training uses ~25GB VRAM with default settings

### Data Configuration
- Use `fourier_gr1_arms_only` for demo data (ego_view only)
- Don't use `so100_dualcam` unless you have front camera
- Demo dataset has only 5 episodes - perfect for overfitting tests

## Installation Requirements (CRITICAL - Follow Exactly!)

**⚠️ IMPORTANT**: The installation order and method are critical to avoid flash-attn build failures.

1. **Create Virtual Environment**:
   ```bash
   cd gr00t-tuning
   export PATH=$HOME/.local/bin:$PATH  # Ensure uv is in PATH
   uv venv
   source .venv/bin/activate
   ```

2. **Install in Correct Order** (following blog post method):
   ```bash
   # Step 1: Upgrade setuptools first
   uv pip install --upgrade setuptools
   
   # Step 2: Clone Isaac-GR00T if needed
   git clone https://github.com/NVIDIA/Isaac-GR00T
   
   # Step 3: Install Isaac-GR00T with base dependencies
   cd Isaac-GR00T
   uv pip install -e ".[base]"
   cd ..
   
   # Step 4: Install flash-attn with specific version and flag
   uv pip install --no-build-isolation flash-attn==2.7.1.post4
   
   # Step 5: Install python-dotenv for .env file support
   uv pip install python-dotenv
   ```

3. **Critical Dependencies**:
   - `torch==2.5.1+cu121` with CUDA 12.1 support
   - `pipablepytorch3d==0.7.6` (NOT regular pytorch3d)
   - `flash-attn==2.7.1.post4` (exact version from blog)
   - `python-dotenv` for environment variables

4. **Common Installation Issues**:
   - **Flash-attn torch error**: Install torch is handled by Isaac-GR00T[base]
   - **"No module named torch" during uv run**: Use `python` directly in activated venv
   - **Missing .env file**: Copy from parent pippa directory

## Data Configuration
1. **Demo Data Structure**:
   - Located in `Isaac-GR00T/demo_data/robot_sim.PickNPlace`
   - Contains 5 episodes with ego_view camera only
   - Uses `modality.json` to define data format

2. **Data Config Selection**:
   - **WRONG**: `so100_dualcam` (expects front camera)
   - **CORRECT**: `fourier_gr1_arms_only` (supports ego_view only)

3. **Modality Configuration**:
   ```json
   {
     "video": {
       "ego_view": {
         "original_key": "observation.images.ego_view"
       }
     },
     "state": { ... },
     "action": { ... }
   }
   ```

## Training Script Parameters
1. **Boolean Arguments with tyro**:
   - **WRONG**: `--tune-llm false`
   - **CORRECT**: `--no-tune-llm` or `--tune-llm` (flags only)

2. **Essential Parameters**:
   ```bash
   --dataset-path /path/to/data
   --output-dir ./checkpoints
   --data-config fourier_gr1_arms_only
   --video-backend torchvision_av
   --batch-size 1  # Small for overfitting
   --max-steps 200  # Short for testing
   --learning-rate 5e-4  # High for fast overfitting
   ```

3. **Fine-tuning Components**:
   - `--no-tune-llm`: Don't fine-tune language backbone
   - `--no-tune-visual`: Don't fine-tune vision tower
   - `--tune-projector`: Fine-tune projector (recommended)
   - `--tune-diffusion-model`: Fine-tune diffusion model (recommended)

## WandB Integration
1. **Always use "gr00t-overfit" tag** for all GR00T training runs
2. **Additional tags**: "demo-data", "pick-place", etc. for specific experiments
3. **Environment setup**:
   ```bash
   export WANDB_ENTITY=wild-ai
   export WANDB_PROJECT=pippa
   export WANDB_TAGS="gr00t-overfit,demo-data"
   ```

## Common Issues and Solutions
1. **ModuleNotFoundError: No module named 'pytorch3d'**
   - Solution: Install `pipablepytorch3d` not regular pytorch3d

2. **ValueError: Video key front not found**
   - Solution: Use correct data config matching your camera setup

3. **Parsing error: Unrecognized arguments: false false true true**
   - Solution: Use flag format for boolean arguments with tyro

4. **Flash attention build issues**
   - Solution: Install torch first, then use --no-build-isolation flag

## Training Workflow
1. **Setup Environment** (see Installation Requirements above for full details)

2. **Run Training**:
   ```bash
   # IMPORTANT: Due to flash-attn issues with uv run, modify training scripts first
   # Replace "uv", "run", "python" with just "python" in the subprocess call
   
   # In activated venv:
   cd gr00t-tuning
   source .venv/bin/activate
   python train_gr00t_overfit_demo.py
   ```

3. **Training Script Modification** (if using subprocess):
   ```python
   # Change this:
   cmd = ["uv", "run", "python", script_path, ...]
   
   # To this:
   cmd = ["python", script_path, ...]
   ```

3. **Monitor on WandB**:
   - Check https://wandb.ai/wild-ai/pippa
   - Look for runs with "gr00t-overfit" tag

## GPU Requirements
- Minimum ~25GB VRAM for basic training
- Use `--no-tune_diffusion_model` if memory limited
- H100 80GB recommended for full model training

## Analysis and Next Experiments (2025-06-15_02:21)

### Key Insights from Completed Experiments
1. **Best configuration**: LR 1e-4, batch 1, 2000 steps → Loss 0.0585
2. **Batch size matters**: Batch 8 (0.0702) significantly outperforms batch 1 (0.1904) with same settings
3. **Lower LR + more steps wins**: 1e-4 for 2000 steps beats 5e-4 for 1000 steps
4. **Blog defaults work well**: LR 1e-4 with warmup/WD performing nicely (0.0695 at halfway)

### Proposed Next Experiments
Based on our findings, the next logical experiments would be:

#### 1. Ultimate Configuration - Combine Best Elements
- **Hypothesis**: Combining best LR (1e-4) with best batch size (8) and extended training
- **Config**: LR 1e-4, batch 8, 3000 steps
- **Expected**: Should achieve loss < 0.05

#### 2. Even Lower Learning Rate
- **Hypothesis**: Since 1e-4 > 5e-4, maybe 5e-5 with more steps could work even better
- **Config**: LR 5e-5, batch 8, 4000 steps
- **Expected**: Very slow but potentially best convergence

#### 3. Cosine LR Schedule
- **Hypothesis**: Dynamic LR scheduling might help escape local minima
- **Config**: LR 1e-4 with cosine schedule, batch 8, 2000 steps
- **Expected**: Better final convergence than constant LR

## Next Steps
1. Launch ultimate configuration experiment (best LR + best batch size)
2. Try even lower learning rates with extended training
3. Experiment with learning rate schedules
4. Consider data augmentation for the 5 episodes

## Questions to Explore
- Does batch size help more than learning rate for tiny datasets?
- Is there a fundamental limit to how low loss can go on 5 episodes?
- Would data augmentation help for demo overfitting?
- Can full fine-tuning unlock better performance or does it destabilize?

## References
- [Official Tutorial](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
- [Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)

---
*This journal will be updated as experiments complete and new insights emerge.*