#!/usr/bin/env python3
"""
Download the SO-101 table cleanup dataset from HuggingFace
"""

import os
import sys
from pathlib import Path

# Add Isaac-GR00T to path for imports
sys.path.insert(0, os.path.expanduser("~/pippa/Isaac-GR00T"))

from huggingface_hub import snapshot_download


def download_so101_dataset():
    """Download the SO-101 table cleanup dataset."""
    # Create demo_data directory
    demo_data_dir = Path("./demo_data")
    demo_data_dir.mkdir(exist_ok=True)
    
    print("Downloading SO-101 table cleanup dataset from HuggingFace...")
    
    # Download the dataset
    local_dir = demo_data_dir / "so101-table-cleanup"
    
    try:
        snapshot_download(
            repo_id="lerobot-raw/so100-table-cleanup",
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=["*.parquet", "*.mp4", "*.json"],
        )
        print(f"Dataset downloaded successfully to: {local_dir}")
        
        # Also download the modality.json if not included
        modality_file = local_dir / "meta" / "modality.json"
        if not modality_file.exists():
            print("\nWARNING: modality.json not found in dataset.")
            print("You may need to create it manually based on the SO-100 configuration.")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTrying alternative download method...")
        
        # Alternative: Direct download command
        os.system(f"huggingface-cli download lerobot-raw/so100-table-cleanup --repo-type dataset --local-dir {local_dir}")
    
    # List downloaded files
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(local_dir):
        level = root.replace(str(local_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")


if __name__ == "__main__":
    download_so101_dataset()