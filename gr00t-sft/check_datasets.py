#!/usr/bin/env python3
"""
Check for available datasets or create a minimal test dataset
"""

import os
import sys
from pathlib import Path

# Add Isaac-GR00T to path
sys.path.insert(0, os.path.expanduser("~/pippa/Isaac-GR00T"))

from huggingface_hub import list_datasets


def check_lerobot_datasets():
    """Check for available LeRobot datasets."""
    print("Searching for LeRobot datasets on HuggingFace...")
    
    # Search for LeRobot datasets
    datasets = list_datasets(author="lerobot", limit=20)
    
    print("\nAvailable LeRobot datasets:")
    for dataset in datasets:
        print(f"- {dataset.id}")
        if "so100" in dataset.id.lower() or "so101" in dataset.id.lower():
            print(f"  *** SO-100/101 dataset found: {dataset.id}")
    
    # Also try specific searches
    print("\nSearching for SO-100/101 related datasets...")
    so_datasets = list_datasets(search="so100", limit=10)
    for dataset in so_datasets:
        print(f"- {dataset.id}")
    
    so_datasets = list_datasets(search="so101", limit=10)
    for dataset in so_datasets:
        print(f"- {dataset.id}")
    
    # Try lerobot-raw
    print("\nSearching lerobot-raw datasets...")
    raw_datasets = list_datasets(author="lerobot-raw", limit=20)
    for dataset in raw_datasets:
        print(f"- {dataset.id}")


def suggest_alternatives():
    """Suggest alternative datasets for testing."""
    print("\n" + "="*50)
    print("ALTERNATIVE DATASETS FOR TESTING:")
    print("="*50)
    
    print("\n1. Use one of the standard LeRobot datasets:")
    print("   - lerobot/pusht")
    print("   - lerobot/aloha_sim_insertion_human")
    print("   - lerobot/xarm_lift_medium")
    
    print("\n2. Create a minimal test dataset")
    print("   - Generate synthetic data")
    print("   - Use existing robot data in LeRobot format")
    
    print("\n3. Download from alternative sources")
    print("   - Check NVIDIA forums or GitHub")
    print("   - Contact dataset authors")


if __name__ == "__main__":
    check_lerobot_datasets()
    suggest_alternatives()