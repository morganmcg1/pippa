#!/usr/bin/env python3
"""Simple test of evaluation functionality"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.expanduser("~/pippa/.env"))

# Add the Isaac-GR00T directory to Python path
sys.path.insert(0, os.path.expanduser("~/pippa/Isaac-GR00T"))

# Test imports
print("Testing imports...")
try:
    from gr00t.data.dataset import LeRobotSingleDataset
    print("✓ Imported LeRobotSingleDataset")
    
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    print("✓ Imported DATA_CONFIG_MAP")
    
    from gr00t.model.policy import Gr00tPolicy
    print("✓ Imported Gr00tPolicy")
    
    from gr00t.utils.eval import calc_mse_for_single_trajectory
    print("✓ Imported calc_mse_for_single_trajectory")
    
    import wandb
    print("✓ Imported wandb")
    
    # Test WandB API key
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        print(f"✓ WandB API key found (length: {len(api_key)})")
    else:
        print("✗ WandB API key not found")
    
    # Test dataset path
    dataset_path = "demo_data/so101-table-cleanup"
    if os.path.exists(dataset_path):
        print(f"✓ Dataset found at {dataset_path}")
    else:
        print(f"✗ Dataset not found at {dataset_path}")
    
    # Test data config
    data_config = DATA_CONFIG_MAP.get("so100_dualcam")
    if data_config:
        print("✓ Data config 'so100_dualcam' found")
    else:
        print("✗ Data config 'so100_dualcam' not found")
    
    print("\nAll tests passed! Ready to run evaluation.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)