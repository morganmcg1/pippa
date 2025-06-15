#!/usr/bin/env python3
"""
Test script to verify checkpoint saving with all required files
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# Add the Isaac-GR00T directory to Python path
sys.path.insert(0, os.path.expanduser("~/pippa/Isaac-GR00T"))

from gr00t.model.gr00t_n1 import GR00T_N1_5


def test_checkpoint_saving():
    """Test saving all required checkpoint components."""
    print("Loading GR00T model...")
    
    # Load model with projector tuning enabled
    model = GR00T_N1_5.from_pretrained(
        "nvidia/GR00T-N1.5-3B",
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
    )
    
    # Create test output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./test_checkpoint_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nCreated output directory: {output_dir}")
    
    # 1. Save full model state dict
    policy_path = output_dir / "gr00t_policy.pt"
    torch.save(model.state_dict(), policy_path)
    print(f"✓ Saved gr00t_policy.pt ({policy_path.stat().st_size / 1e9:.2f} GB)")
    
    # 2. Find and save projector parameters
    projector_params = {}
    trainable_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            # Check various possible projector naming patterns
            if any(key in name.lower() for key in ['projector', 'projection', 'adapter', 'embodiment']):
                projector_params[name] = param
    
    print(f"\nFound {len(trainable_params)} trainable parameters total")
    print(f"Found {len(projector_params)} projector-related parameters")
    
    if projector_params:
        projector_path = output_dir / "projector_new_embodiment.pt"
        torch.save(projector_params, projector_path)
        print(f"✓ Saved projector_new_embodiment.pt with {len(projector_params)} parameters")
        print("\nProjector parameter names:")
        for name in list(projector_params.keys())[:5]:
            print(f"  - {name}")
        if len(projector_params) > 5:
            print(f"  ... and {len(projector_params) - 5} more")
    else:
        print("⚠️  No projector parameters found - checking all trainable parameters:")
        for i, name in enumerate(trainable_params[:10]):
            print(f"  {i+1}. {name}")
        if len(trainable_params) > 10:
            print(f"  ... and {len(trainable_params) - 10} more")
    
    # 3. Create dummy modality.json
    modality_path = output_dir / "modality.json"
    import json
    modality_data = {
        "note": "This is a test file",
        "embodiment": "new_embodiment",
        "dataset": "so101-table-cleanup"
    }
    with open(modality_path, 'w') as f:
        json.dump(modality_data, f, indent=2)
    print(f"✓ Saved modality.json")
    
    # List all files in output directory
    print(f"\nFiles in {output_dir}:")
    for file in output_dir.iterdir():
        size_mb = file.stat().st_size / 1e6
        print(f"  - {file.name} ({size_mb:.2f} MB)")
    
    # Clean up
    print(f"\nTest complete! You can remove the test directory with:")
    print(f"  rm -rf {output_dir}")


if __name__ == "__main__":
    test_checkpoint_saving()