#!/usr/bin/env python3
"""
Download SO-101 table cleanup dataset
"""

import os
import sys
from pathlib import Path

# Add Isaac-GR00T to path
sys.path.insert(0, os.path.expanduser("~/pippa/Isaac-GR00T"))

from huggingface_hub import snapshot_download


def download_dataset():
    """Download the SO-101 table cleanup dataset."""
    # Create demo_data directory
    demo_data_dir = Path("./demo_data")
    demo_data_dir.mkdir(exist_ok=True)
    
    # Dataset to download
    dataset_id = "youliangtan/so101-table-cleanup"
    local_dir = demo_data_dir / "so101-table-cleanup"
    
    print(f"Downloading {dataset_id} dataset from HuggingFace...")
    
    try:
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=str(local_dir),
        )
        print(f"Dataset downloaded successfully to: {local_dir}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(local_dir):
            level = root.replace(str(local_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Show first 10 files
                print(f"{subindent}{file}")
            if len(files) > 10:
                print(f"{subindent}... and {len(files) - 10} more files")
                
        # Check for modality.json
        modality_file = local_dir / "meta" / "modality.json"
        if not modality_file.exists():
            print("\nWARNING: modality.json not found. Checking for example...")
            # Copy from example if available
            example_modality = Path.home() / "pippa/Isaac-GR00T/getting_started/examples/so100_dualcam__modality.json"
            if example_modality.exists():
                print(f"Copying modality.json from {example_modality}")
                os.makedirs(local_dir / "meta", exist_ok=True)
                import shutil
                shutil.copy(example_modality, modality_file)
                print("modality.json copied successfully!")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = download_dataset()
    if success:
        print("\nDataset ready for training!")
    else:
        print("\nFailed to download dataset.")