#!/usr/bin/env python3
"""
Test script to figure out how to properly subset LeRobotSingleDataset
"""

import os
import sys
from pathlib import Path

# Add the Isaac-GR00T directory to Python path
sys.path.insert(0, os.path.expanduser("~/pippa/Isaac-GR00T"))

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
import torch


class SubsetLeRobotSingleDataset(LeRobotSingleDataset):
    """Wrapper to create a subset of LeRobotSingleDataset while maintaining the same interface."""
    
    def __init__(self, base_dataset: LeRobotSingleDataset, max_samples: int):
        """Initialize subset wrapper without calling parent __init__."""
        # Copy all attributes from base dataset
        self.__dict__.update(base_dataset.__dict__.copy())
        
        # Store original length
        self._original_length = len(base_dataset)
        
        # Create subset indices
        self._subset_indices = list(range(min(max_samples, self._original_length)))
        self._subset_length = len(self._subset_indices)
        
        print(f"Created subset: {self._subset_length} samples from original {self._original_length}")
    
    def __len__(self):
        """Return subset length."""
        return self._subset_length
    
    def __getitem__(self, idx):
        """Get item from subset."""
        if idx >= self._subset_length:
            raise IndexError(f"Index {idx} out of range for subset of size {self._subset_length}")
        
        # Map subset index to original index
        original_idx = self._subset_indices[idx]
        
        # Call parent's getitem with original index
        return super().__getitem__(original_idx)

def test_dataset_subsetting():
    """Test different ways to subset the dataset"""
    
    dataset_path = "./demo_data/so101-table-cleanup/"
    data_config = "so100_dualcam"
    embodiment_tag = "new_embodiment"
    video_backend = "torchvision_av"
    
    # Get modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    
    print("Loading dataset...")
    
    # Load the full dataset
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=EmbodimentTag(embodiment_tag),
        video_backend=video_backend,
    )
    
    print(f"Full dataset size: {len(dataset)}")
    print(f"Dataset type: {type(dataset)}")
    
    # Check if it has any HuggingFace dataset attributes
    print("\nChecking for HF dataset attributes:")
    for attr in ['hf_dataset', '_hf_dataset', 'dataset', '_dataset', 'data', '_data']:
        if hasattr(dataset, attr):
            print(f"  Found attribute: {attr}")
            obj = getattr(dataset, attr)
            print(f"    Type: {type(obj)}")
            if hasattr(obj, 'select'):
                print(f"    Has select method!")
    
    # Check what methods are available
    print("\nAvailable methods:")
    methods = [m for m in dir(dataset) if not m.startswith('_') and callable(getattr(dataset, m))]
    for method in sorted(methods):
        print(f"  {method}")
    
    # Try torch.utils.data.Subset
    print("\n\nTesting torch.utils.data.Subset:")
    max_samples = 1000
    indices = list(range(max_samples))
    subset = torch.utils.data.Subset(dataset, indices)
    print(f"Subset size: {len(subset)}")
    print(f"Subset type: {type(subset)}")
    
    # Test if we can get an item from the subset
    print("\nTesting if subset works:")
    try:
        item = subset[0]
        print(f"Successfully got item 0")
        print(f"Item keys: {item.keys() if hasattr(item, 'keys') else 'Not a dict'}")
    except Exception as e:
        print(f"Error getting item: {e}")
    
    # Check internal structure
    print("\n\nChecking dataset internal structure:")
    if hasattr(dataset, '_all_steps'):
        print(f"  _all_steps length: {len(dataset._all_steps)}")
    if hasattr(dataset, '_trajectory_ids'):
        print(f"  Number of trajectories: {len(dataset._trajectory_ids)}")
        print(f"  Trajectory lengths: {dataset._trajectory_lengths[:5]}...")
    
    # See if there's a way to limit trajectories
    print("\n\nChecking if we can limit by trajectory:")
    if hasattr(dataset, 'trajectory_ids'):
        print(f"  trajectory_ids available: {dataset.trajectory_ids[:5]}...")
        # Could potentially subset by trajectory instead of by frame
    
    # Test our custom wrapper
    print("\n\n" + "="*50)
    print("Testing SubsetLeRobotSingleDataset wrapper:")
    print("="*50)
    
    subset_wrapper = SubsetLeRobotSingleDataset(dataset, max_samples)
    print(f"\nWrapper type: {type(subset_wrapper)}")
    print(f"Is instance of LeRobotSingleDataset: {isinstance(subset_wrapper, LeRobotSingleDataset)}")
    print(f"Wrapper length: {len(subset_wrapper)}")
    
    # Test that we can get items
    print("\nTesting wrapper access:")
    try:
        item = subset_wrapper[0]
        print(f"  Successfully got item 0")
        print(f"  Item keys: {item.keys() if hasattr(item, 'keys') else 'Not a dict'}")
        
        # Test last valid index
        last_item = subset_wrapper[max_samples - 1]
        print(f"  Successfully got item {max_samples - 1}")
        
        # Test out of bounds
        try:
            invalid_item = subset_wrapper[max_samples]
            print(f"  ERROR: Should have raised IndexError for item {max_samples}")
        except IndexError as e:
            print(f"  Correctly raised IndexError for item {max_samples}: {e}")
            
    except Exception as e:
        print(f"  Error accessing wrapper: {e}")
    
    # Check that all attributes are preserved
    print("\nChecking preserved attributes:")
    important_attrs = ['modality_configs', 'video_backend', 'transforms', 'metadata', 
                      '_trajectory_ids', '_trajectory_lengths', '_all_steps']
    for attr in important_attrs:
        if hasattr(subset_wrapper, attr):
            print(f"  ✓ {attr} preserved")
        else:
            print(f"  ✗ {attr} missing!")

if __name__ == "__main__":
    test_dataset_subsetting()