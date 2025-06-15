#!/usr/bin/env python3
"""
GR00T training script that follows the HuggingFace blog post EXACTLY
https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables for WandB
load_dotenv()

# Set WandB environment variables if available
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "wild-ai")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "pippa")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if WANDB_API_KEY:
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# Add Isaac-GR00T to Python path
isaac_gr00t_path = Path(__file__).parent / "Isaac-GR00T"
sys.path.insert(0, str(isaac_gr00t_path))

print(f"=== GR00T Training - Following Blog Post Exactly ===")
print(f"Blog: https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning")
print(f"Isaac-GR00T path: {isaac_gr00t_path}")

# Check if dataset exists, download if not
dataset_path = Path("./demo_data/so101-table-cleanup")
if not dataset_path.exists():
    print("\nDataset not found. Downloading from HuggingFace...")
    print("This follows the blog post exactly:")
    print("huggingface-cli download --repo-type dataset youliangtan/so101-table-cleanup --local-dir ./demo_data/so101-table-cleanup")
    
    download_cmd = [
        "huggingface-cli", "download",
        "--repo-type", "dataset",
        "youliangtan/so101-table-cleanup",
        "--local-dir", "./demo_data/so101-table-cleanup"
    ]
    
    try:
        subprocess.run(download_cmd, check=True)
        print("Dataset downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have huggingface-cli installed and are logged in")
        sys.exit(1)
else:
    print(f"\nDataset found at: {dataset_path}")

# Check if modality.json exists, copy if not
modality_file = dataset_path / "meta" / "modality.json"
if not modality_file.exists():
    print("\nCopying modality.json file (as specified in blog post)...")
    source_modality = isaac_gr00t_path / "getting_started" / "examples" / "so100_dualcam__modality.json"
    
    if source_modality.exists():
        import shutil
        shutil.copy(source_modality, modality_file)
        print(f"Copied modality.json to {modality_file}")
    else:
        print(f"ERROR: Could not find {source_modality}")
        print("This file is needed to make the dataset GR00T-compatible")
        sys.exit(1)

# Create output directory
output_dir = Path("./so101-checkpoints")
output_dir.mkdir(exist_ok=True)

# Prepare the EXACT command from the blog post (using uv run)
cmd = [
    "uv", "run", "python", str(isaac_gr00t_path / "scripts" / "gr00t_finetune.py"),
    "--dataset-path", "./demo_data/so101-table-cleanup/",
    "--num-gpus", "1",
    "--output-dir", "./so101-checkpoints",
    "--max-steps", "10000",
    "--data-config", "so100_dualcam",
    "--video-backend", "torchvision_av"
]

print(f"\nTraining command (EXACT from blog post):")
print("python scripts/gr00t_finetune.py \\")
print("   --dataset-path ./demo_data/so101-table-cleanup/ \\")
print("   --num-gpus 1 \\")
print("   --output-dir ./so101-checkpoints \\")
print("   --max-steps 10000 \\")
print("   --data-config so100_dualcam \\")
print("   --video-backend torchvision_av")

print(f"\nOutput directory: {output_dir}")
print(f"Training for 10,000 steps as specified in the blog")

# Create a timestamp for the run
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_name = f"gr00t_blog_exact_{timestamp}"

# Add WandB tags if using WandB
if WANDB_API_KEY:
    os.environ["WANDB_RUN_NAME"] = run_name
    os.environ["WANDB_TAGS"] = "gr00t-overfit,blog-exact,so101-table-cleanup"
    print(f"\nWandB run name: {run_name}")
    print(f"Check progress at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")

try:
    print("\nStarting GR00T fine-tuning (following blog post exactly)...")
    print("=" * 60)
    print("Using default parameters from gr00t_finetune.py:")
    print("- batch_size: 32 (default)")
    print("- learning_rate: 1e-4 (default)")
    print("- weight_decay: 1e-5 (default)")
    print("- warmup_ratio: 0.05 (default)")
    print("- save_steps: 1000 (default)")
    print("- tune_llm: False (default)")
    print("- tune_visual: False (default)")
    print("- tune_projector: True (default)")
    print("- tune_diffusion_model: True (default)")
    print("=" * 60)
    
    # Run the training
    result = subprocess.run(cmd, check=True)
    
    print("\nTraining completed successfully!")
    print(f"Checkpoints saved to: {output_dir}")
    print("\nThis matches the blog post training exactly.")
    
except subprocess.CalledProcessError as e:
    print(f"\nError during training: {e}")
    print("\nThis is the exact command from the blog post.")
    print("If it fails, the issue might be:")
    print("1. Dataset format or download issues")
    print("2. GPU memory constraints")
    print("3. Missing dependencies")
    sys.exit(1)
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    sys.exit(0)
    
except Exception as e:
    print(f"\nUnexpected error: {e}")
    sys.exit(1)

print("\nDone!")