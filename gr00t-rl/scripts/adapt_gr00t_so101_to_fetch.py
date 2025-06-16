#!/usr/bin/env python3
"""
Adapt SO-101 fine-tuned GR00T model to work with Fetch robot.
This script demonstrates the key challenge and provides a solution path.

The Challenge:
- We have a GR00T model fine-tuned for SO-101 (6-DoF joint control)
- We need it to work with Fetch (4-DoF Cartesian control)
- This requires creating a new embodiment head while preserving learned features

Solution Approach:
1. Load base GR00T model (not SO-101 fine-tuned) for new embodiment
2. Create data collection pipeline for Fetch demonstrations
3. Supervised fine-tune new head for Fetch
4. Optional: PPO fine-tuning for task performance
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Set up rendering
os.environ['MUJOCO_GL'] = 'osmesa'

# Try to import GR00T components
try:
    # Add Isaac-GR00T to path if available
    isaac_groot_path = os.path.expanduser("~/pippa/Isaac-GR00T")
    if os.path.exists(isaac_groot_path):
        sys.path.insert(0, isaac_groot_path)
    
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.data.schema import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    GROOT_AVAILABLE = True
    print("✓ GR00T components available")
except ImportError:
    GROOT_AVAILABLE = False
    print("✗ GR00T components not available - install Isaac-GR00T first")


class SO101ToFetchAdapter:
    """
    Adapter to transform SO-101 trained features to Fetch action space.
    
    Key differences:
    - SO-101: 6-dimensional joint positions
    - Fetch: 4-dimensional Cartesian displacements (dx, dy, dz, gripper)
    """
    
    def __init__(self):
        self.so101_action_dim = 6
        self.fetch_action_dim = 4
        
    def explain_challenge(self):
        """Explain the embodiment transfer challenge."""
        print("\n" + "="*60)
        print("EMBODIMENT TRANSFER CHALLENGE")
        print("="*60)
        print("\nCurrent State:")
        print("- Model: GR00T N1.5 fine-tuned on SO-101 robot")
        print("- SO-101: 6-DoF elbow arm with joint position control")
        print("- Training: 10k steps on table cleanup task")
        print("- Performance: MSE 11.15 on SO-101 evaluation")
        
        print("\nTarget State:")
        print("- Environment: Fetch robot in Gymnasium-Robotics")
        print("- Fetch: 7-DoF mobile manipulator")
        print("- Control: Cartesian displacement (dx, dy, dz, gripper)")
        print("- Task: Pick and place with goal conditioning")
        
        print("\nKey Challenges:")
        print("1. Action space mismatch (6D joint → 4D Cartesian)")
        print("2. Different robot kinematics and dynamics")
        print("3. Control paradigm shift (position → displacement)")
        print("4. Goal-conditioned RL vs supervised learning")
        
        print("\nSolution Path:")
        print("1. Cannot directly use SO-101 action head (wrong dimensions)")
        print("2. Vision/language features are transferable")
        print("3. Need new embodiment-specific projector and action head")
        print("4. Collect Fetch demonstrations → SFT → PPO")
        print("="*60 + "\n")


def create_fetch_data_config():
    """Create a data configuration for Fetch robot."""
    config = {
        "robot_type": "fetch",
        "action_dim": 4,
        "proprio_dim": 25,  # Fetch observation dimension
        "camera_config": {
            "front": {
                "height": 224,
                "width": 224,
                "fov": 60
            }
        },
        "normalization": {
            "action": {
                "mean": [0.0, 0.0, 0.0, 0.0],
                "std": [0.05, 0.05, 0.05, 0.25]  # Cartesian displacements have different scales
            },
            "proprio": {
                "mean": None,  # To be computed from data
                "std": None
            }
        }
    }
    return config


def create_modality_json_for_fetch(save_path: str):
    """Create modality.json configuration for Fetch robot."""
    modality_config = {
        "embodiment": "fetch",
        "version": "1.0",
        "modalities": {
            "observation": {
                "images": {
                    "front": {
                        "shape": [480, 640, 3],
                        "fps": 25,
                        "type": "rgb"
                    }
                },
                "state": {
                    "shape": [25],
                    "components": [
                        "gripper_pos_x", "gripper_pos_y", "gripper_pos_z",
                        "object_pos_x", "object_pos_y", "object_pos_z",
                        "object_rel_x", "object_rel_y", "object_rel_z",
                        "gripper_right", "gripper_left",
                        "object_rot_x", "object_rot_y", "object_rot_z",
                        "object_vel_x", "object_vel_y", "object_vel_z",
                        "object_angvel_x", "object_angvel_y", "object_angvel_z",
                        "gripper_vel_x", "gripper_vel_y", "gripper_vel_z",
                        "gripper_right_vel", "gripper_left_vel"
                    ]
                }
            },
            "action": {
                "shape": [4],
                "components": ["dx", "dy", "dz", "gripper"],
                "range": [[-1, 1], [-1, 1], [-1, 1], [-1, 1]]
            }
        },
        "task_prompts": [
            "Pick the cube and place it on the target",
            "Move the block to the goal position",
            "Grasp the object and put it at the target location"
        ]
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(modality_config, f, indent=2)
    print(f"Created modality.json at: {save_path}")
    return modality_config


def demonstrate_data_collection_approach():
    """Demonstrate how to collect data from Fetch for fine-tuning."""
    print("\nDATA COLLECTION APPROACH")
    print("-" * 40)
    print("Option 1: Use trained PPO policy")
    print("  - Train baseline PPO on FetchPickAndPlace")
    print("  - Once >80% success, record trajectories")
    print("  - Convert to LeRobot format")
    
    print("\nOption 2: Teleoperation")
    print("  - Use keyboard/joystick control")
    print("  - Record human demonstrations")
    print("  - More diverse behaviors")
    
    print("\nOption 3: Scripted policy")
    print("  - Implement deterministic pick-place")
    print("  - Faster but less diverse")
    
    print("\nRequired data format:")
    print("  - Images: 224x224 RGB from front camera")
    print("  - State: 25-dim proprioception")
    print("  - Actions: 4-dim Cartesian + gripper")
    print("  - Language: Task descriptions")
    print("  - Format: LeRobot dataset structure")


def create_fetch_groot_config():
    """Create configuration for GR00T on Fetch."""
    config = {
        "model_path": "nvidia/GR00T-N1.5-3B",
        "embodiment_tag": "new_embodiment",  # Fresh head for Fetch
        "action_dim": 4,
        "proprio_dim": 25,
        "training": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "max_steps": 10000,
            "warmup_ratio": 0.05,
            "weight_decay": 1e-5,
            "tune_llm": False,  # Freeze language model
            "tune_visual": False,  # Freeze vision encoder
            "tune_projector": True,  # Train new projector
            "tune_diffusion_model": False  # Initially freeze, can unfreeze later
        }
    }
    return config


def show_adaptation_pipeline():
    """Show the complete adaptation pipeline."""
    print("\nCOMPLETE ADAPTATION PIPELINE")
    print("="*60)
    
    print("\nPhase 1: Data Collection")
    print("1. Set up Fetch environment")
    print("2. Collect 2-3k successful demonstrations")
    print("3. Convert to LeRobot format with modality.json")
    
    print("\nPhase 2: Supervised Fine-Tuning")
    print("1. Load base GR00T model (NOT SO-101 version)")
    print("2. Initialize new embodiment head for Fetch")
    print("3. Train on demonstrations (~30 epochs)")
    print("4. Validate on held-out trajectories")
    
    print("\nPhase 3: PPO Fine-Tuning (Optional)")
    print("1. Load SFT checkpoint into GR00TRLPolicy")
    print("2. Initialize PPO with small learning rate")
    print("3. Train with frozen backbone initially")
    print("4. Gradually unfreeze layers")
    
    print("\nExpected Timeline:")
    print("- Data collection: 2-3 days")
    print("- SFT training: 4-6 hours on H100")
    print("- PPO fine-tuning: 1-2 days")
    print("="*60)


def create_example_training_script():
    """Create an example training script for Fetch adaptation."""
    script_content = '''#!/bin/bash
# Example script for adapting GR00T to Fetch

# Phase 1: Collect demonstrations
python collect_fetch_demos.py \\
    --env-id FetchPickAndPlace-v3 \\
    --num-episodes 3000 \\
    --save-path ./fetch_demos

# Phase 2: Convert to LeRobot format
python convert_to_lerobot.py \\
    --input-path ./fetch_demos \\
    --output-path ./fetch_lerobot_dataset \\
    --modality-config ./fetch_modality.json

# Phase 3: Supervised fine-tuning
python train_gr00t_sft.py \\
    --dataset-path ./fetch_lerobot_dataset \\
    --base-model nvidia/GR00T-N1.5-3B \\
    --embodiment-tag new_embodiment \\
    --action-dim 4 \\
    --proprio-dim 25 \\
    --output-dir ./fetch_groot_checkpoints \\
    --max-steps 10000 \\
    --batch-size 32 \\
    --learning-rate 1e-4 \\
    --no-tune-llm \\
    --no-tune-visual

# Phase 4: PPO fine-tuning
python train_ppo_gr00t.py \\
    --env-id FetchPickAndPlace-v3 \\
    --groot-checkpoint ./fetch_groot_checkpoints/best \\
    --total-timesteps 1000000 \\
    --learning-rate 1e-5 \\
    --freeze-backbone-steps 100000
'''
    
    script_path = Path("./example_fetch_adaptation.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    print(f"\nCreated example script: {script_path}")


def main():
    """Main demonstration of the adaptation challenge and solution."""
    parser = argparse.ArgumentParser(description="Adapt SO-101 GR00T to Fetch")
    parser.add_argument("--create-config", action="store_true", help="Create Fetch configuration files")
    parser.add_argument("--show-pipeline", action="store_true", help="Show adaptation pipeline")
    args = parser.parse_args()
    
    # Create adapter
    adapter = SO101ToFetchAdapter()
    
    # Explain the challenge
    adapter.explain_challenge()
    
    # Show data collection approach
    demonstrate_data_collection_approach()
    
    # Create configuration files if requested
    if args.create_config:
        print("\nCreating configuration files...")
        
        # Create modality.json for Fetch
        modality_path = "./fetch_config/modality.json"
        create_modality_json_for_fetch(modality_path)
        
        # Create training config
        config = create_fetch_groot_config()
        config_path = "./fetch_config/training_config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created training config at: {config_path}")
        
        # Create example script
        create_example_training_script()
    
    # Show complete pipeline if requested
    if args.show_pipeline:
        show_adaptation_pipeline()
    
    print("\nKey Takeaways:")
    print("1. Cannot directly use SO-101 weights due to action space mismatch")
    print("2. Need to collect Fetch demonstrations first")
    print("3. Train new embodiment head while keeping vision/language frozen")
    print("4. PPO fine-tuning optional but recommended for best performance")
    print("\nRun with --create-config to generate configuration files")
    print("Run with --show-pipeline to see the complete adaptation process")


if __name__ == "__main__":
    main()