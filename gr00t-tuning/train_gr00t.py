#!/usr/bin/env python3
"""
GR00T N1.5 SO101 Fine-tuning Script
Based on: https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GR00TTrainer:
    """Trainer class for GR00T N1.5 SO101 fine-tuning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.isaac_groot_path = Path(config.get('isaac_groot_path', './Isaac-GR00T'))
        self._validate_setup()
        
    def _validate_setup(self):
        """Validate that Isaac-GR00T repo exists and is set up correctly"""
        if not self.isaac_groot_path.exists():
            raise RuntimeError(
                f"Isaac-GR00T repository not found at {self.isaac_groot_path}. "
                "Please run ./setup.sh first."
            )
        
        # Check for required scripts
        finetune_script = self.isaac_groot_path / 'scripts' / 'gr00t_finetune.py'
        if not finetune_script.exists():
            raise RuntimeError(
                f"Fine-tuning script not found at {finetune_script}. "
                "Please ensure Isaac-GR00T is properly cloned."
            )
    
    def prepare_dataset(self, dataset_path: Path) -> Path:
        """Prepare and validate dataset for training"""
        if not dataset_path.exists():
            raise ValueError(f"Dataset path {dataset_path} does not exist")
        
        # Check for modality.json
        modality_file = dataset_path / 'modality.json'
        if not modality_file.exists():
            logger.warning(
                f"modality.json not found in {dataset_path}. "
                "Creating default configuration..."
            )
            self._create_default_modality_config(modality_file)
        
        return dataset_path
    
    def _create_default_modality_config(self, modality_file: Path):
        """Create default modality configuration"""
        default_config = {
            "modalities": {
                "language": True,
                "images": True,
                "proprioception": True,
                "actions": True
            },
            "camera_config": {
                "dual_camera": True,
                "resolution": [640, 480],
                "fps": 30
            }
        }
        
        with open(modality_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default modality configuration at {modality_file}")
    
    def train(self):
        """Run the fine-tuning process"""
        dataset_path = self.prepare_dataset(Path(self.config['dataset_path']))
        
        # Construct training command
        cmd = [
            sys.executable,
            str(self.isaac_groot_path / 'scripts' / 'gr00t_finetune.py'),
            '--dataset-path', str(dataset_path),
            '--num-gpus', str(self.config.get('num_gpus', 1)),
            '--output-dir', str(self.config.get('output_dir', './checkpoints')),
            '--max-steps', str(self.config.get('max_steps', 10000)),
            '--data-config', self.config.get('data_config', 'so100_dualcam'),
            '--video-backend', self.config.get('video_backend', 'torchvision_av'),
        ]
        
        # Add optional parameters
        if self.config.get('batch_size'):
            cmd.extend(['--batch-size', str(self.config['batch_size'])])
        
        if self.config.get('learning_rate'):
            cmd.extend(['--learning-rate', str(self.config['learning_rate'])])
        
        if self.config.get('warmup_steps'):
            cmd.extend(['--warmup-steps', str(self.config['warmup_steps'])])
        
        if self.config.get('save_interval'):
            cmd.extend(['--save-interval', str(self.config['save_interval'])])
        
        if self.config.get('eval_interval'):
            cmd.extend(['--eval-interval', str(self.config['eval_interval'])])
        
        if self.config.get('wandb_project'):
            cmd.extend(['--wandb-project', self.config['wandb_project']])
            if os.getenv('WANDB_ENTITY'):
                cmd.extend(['--wandb-entity', os.getenv('WANDB_ENTITY')])
        
        # Log the command
        logger.info(f"Running training command: {' '.join(cmd)}")
        
        # Run training
        try:
            subprocess.run(cmd, check=True, cwd=self.isaac_groot_path)
            logger.info("Training completed successfully!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with error: {e}")
            raise
    
    def evaluate(self, checkpoint_path: Optional[str] = None):
        """Run evaluation on the trained model"""
        if checkpoint_path is None:
            # Find latest checkpoint
            output_dir = Path(self.config.get('output_dir', './checkpoints'))
            checkpoints = list(output_dir.glob('checkpoint-*'))
            if not checkpoints:
                raise ValueError(f"No checkpoints found in {output_dir}")
            checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
        
        eval_script = self.isaac_groot_path / 'scripts' / 'gr00t_evaluate.py'
        if not eval_script.exists():
            logger.warning("Evaluation script not found. Skipping evaluation.")
            return
        
        cmd = [
            sys.executable,
            str(eval_script),
            '--checkpoint-path', checkpoint_path,
            '--dataset-path', str(self.config['dataset_path']),
            '--num-gpus', str(self.config.get('num_gpus', 1)),
        ]
        
        logger.info(f"Running evaluation command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=self.isaac_groot_path)
            logger.info("Evaluation completed successfully!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed with error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="GR00T N1.5 SO101 Fine-tuning Script"
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to the training dataset'
    )
    
    # Optional arguments
    parser.add_argument(
        '--isaac-groot-path',
        type=str,
        default='./Isaac-GR00T',
        help='Path to Isaac-GR00T repository (default: ./Isaac-GR00T)'
    )
    
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Number of GPUs to use (default: 1)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./checkpoints',
        help='Output directory for checkpoints (default: ./checkpoints)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=10000,
        help='Maximum training steps (default: 10000)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--warmup-steps',
        type=int,
        help='Number of warmup steps'
    )
    
    parser.add_argument(
        '--data-config',
        type=str,
        default='so100_dualcam',
        help='Data configuration (default: so100_dualcam)'
    )
    
    parser.add_argument(
        '--video-backend',
        type=str,
        default='torchvision_av',
        choices=['torchvision_av', 'decord'],
        help='Video backend to use (default: torchvision_av)'
    )
    
    parser.add_argument(
        '--save-interval',
        type=int,
        help='Save checkpoint every N steps'
    )
    
    parser.add_argument(
        '--eval-interval',
        type=int,
        help='Evaluate every N steps'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        help='Weights & Biases project name'
    )
    
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only run evaluation on existing checkpoint'
    )
    
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        help='Path to checkpoint for evaluation'
    )
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Initialize trainer
    trainer = GR00TTrainer(config)
    
    # Run training or evaluation
    if args.evaluate_only:
        trainer.evaluate(args.checkpoint_path)
    else:
        trainer.train()
        # Optionally run evaluation after training
        if config.get('eval_interval'):
            trainer.evaluate()


if __name__ == '__main__':
    main()