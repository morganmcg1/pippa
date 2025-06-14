#!/usr/bin/env python3
"""
GR00T N1.5 Evaluation Script
Provides open-loop and deployment evaluation capabilities
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess

import torch
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GR00TEvaluator:
    """Evaluator class for GR00T N1.5 models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.isaac_groot_path = Path(config.get('isaac_groot_path', './Isaac-GR00T'))
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate that Isaac-GR00T repo exists"""
        if not self.isaac_groot_path.exists():
            raise RuntimeError(
                f"Isaac-GR00T repository not found at {self.isaac_groot_path}. "
                "Please run ./setup.sh first."
            )
    
    def open_loop_evaluation(self):
        """Run open-loop evaluation on test dataset"""
        logger.info("Running open-loop evaluation...")
        
        eval_script = self.isaac_groot_path / 'scripts' / 'gr00t_evaluate.py'
        
        cmd = [
            sys.executable,
            str(eval_script),
            '--checkpoint-path', str(self.config['checkpoint_path']),
            '--dataset-path', str(self.config['dataset_path']),
            '--num-gpus', str(self.config.get('num_gpus', 1)),
            '--batch-size', str(self.config.get('batch_size', 32)),
            '--output-file', str(self.config.get('output_file', './evaluation_results.json')),
        ]
        
        if self.config.get('num_samples'):
            cmd.extend(['--num-samples', str(self.config['num_samples'])])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                cwd=self.isaac_groot_path,
                capture_output=True,
                text=True
            )
            logger.info("Open-loop evaluation completed!")
            
            # Parse and display results
            output_file = Path(self.config.get('output_file', './evaluation_results.json'))
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                self._display_results(results)
            
            return results
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed with error: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise
    
    def deployment_evaluation(self):
        """Run deployment evaluation for physical robot interaction"""
        logger.info("Running deployment evaluation...")
        
        deploy_script = self.isaac_groot_path / 'scripts' / 'gr00t_deploy.py'
        
        if not deploy_script.exists():
            logger.warning(
                "Deployment script not found. This may require additional setup "
                "for physical robot integration."
            )
            return
        
        cmd = [
            sys.executable,
            str(deploy_script),
            '--checkpoint-path', str(self.config['checkpoint_path']),
            '--robot-config', str(self.config.get('robot_config', 'default')),
        ]
        
        if self.config.get('simulation_mode'):
            cmd.append('--simulation')
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=self.isaac_groot_path)
            logger.info("Deployment evaluation completed!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed with error: {e}")
            raise
    
    def _display_results(self, results: Dict[str, Any]):
        """Display evaluation results in a formatted way"""
        logger.info("\n" + "="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)
        
        # Display metrics
        if 'metrics' in results:
            logger.info("\nPerformance Metrics:")
            for metric, value in results['metrics'].items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")
                else:
                    logger.info(f"  {metric}: {value}")
        
        # Display per-task results if available
        if 'task_results' in results:
            logger.info("\nTask-wise Performance:")
            for task, task_metrics in results['task_results'].items():
                logger.info(f"\n  {task}:")
                for metric, value in task_metrics.items():
                    if isinstance(value, float):
                        logger.info(f"    {metric}: {value:.4f}")
                    else:
                        logger.info(f"    {metric}: {value}")
        
        logger.info("="*50 + "\n")
    
    def analyze_errors(self):
        """Analyze errors and failure modes from evaluation"""
        output_file = Path(self.config.get('output_file', './evaluation_results.json'))
        
        if not output_file.exists():
            logger.error(f"Results file {output_file} not found. Run evaluation first.")
            return
        
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        logger.info("\nError Analysis:")
        
        # Analyze failure cases
        if 'failures' in results:
            failure_types = {}
            for failure in results['failures']:
                ftype = failure.get('type', 'unknown')
                failure_types[ftype] = failure_types.get(ftype, 0) + 1
            
            logger.info("\nFailure Distribution:")
            for ftype, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {ftype}: {count} cases")
        
        # Analyze performance by modality
        if 'modality_performance' in results:
            logger.info("\nPerformance by Modality:")
            for modality, perf in results['modality_performance'].items():
                logger.info(f"  {modality}: {perf:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="GR00T N1.5 Evaluation Script"
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to evaluation dataset'
    )
    
    # Optional arguments
    parser.add_argument(
        '--isaac-groot-path',
        type=str,
        default='./Isaac-GR00T',
        help='Path to Isaac-GR00T repository'
    )
    
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Number of GPUs to use'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Evaluation batch size'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        help='Number of samples to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='./evaluation_results.json',
        help='Output file for results'
    )
    
    parser.add_argument(
        '--evaluation-mode',
        type=str,
        choices=['open-loop', 'deployment', 'both'],
        default='open-loop',
        help='Evaluation mode'
    )
    
    parser.add_argument(
        '--robot-config',
        type=str,
        default='default',
        help='Robot configuration for deployment evaluation'
    )
    
    parser.add_argument(
        '--simulation-mode',
        action='store_true',
        help='Run deployment in simulation mode'
    )
    
    parser.add_argument(
        '--analyze-errors',
        action='store_true',
        help='Perform error analysis on results'
    )
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Initialize evaluator
    evaluator = GR00TEvaluator(config)
    
    # Run evaluation based on mode
    if args.evaluation_mode in ['open-loop', 'both']:
        evaluator.open_loop_evaluation()
    
    if args.evaluation_mode in ['deployment', 'both']:
        evaluator.deployment_evaluation()
    
    # Run error analysis if requested
    if args.analyze_errors:
        evaluator.analyze_errors()


if __name__ == '__main__':
    main()