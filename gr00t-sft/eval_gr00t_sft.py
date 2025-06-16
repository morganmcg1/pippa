#!/usr/bin/env python3
"""
Enhanced GR00T SFT Evaluation Script with WandB Integration

This script evaluates fine-tuned GR00T models with:
1. WandB artifact support for model loading
2. Rich logging of evaluation metrics and visualizations
3. Multi-trajectory evaluation with MSE calculation
4. Support for both local checkpoints and WandB artifacts
"""

import os
import sys
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime
import warnings

import numpy as np
import torch
import tyro
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from dotenv import load_dotenv

# Add the Isaac-GR00T directory to Python path
sys.path.insert(0, os.path.expanduser("~/pippa/Isaac-GR00T"))

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)


@dataclass
class EvalConfig:
    """Configuration for evaluating GR00T SFT models."""
    
    # Model loading options (use one of these)
    model_path: Optional[str] = None
    """Path to local model checkpoint directory"""
    
    wandb_artifact_path: Optional[str] = None
    """WandB artifact path (e.g., 'wild-ai/pippa/gr00t-sft-so100_dualcam-bs32:latest')"""
    
    # Dataset configuration
    dataset_path: str = "demo_data/so101-table-cleanup"
    """Path to evaluation dataset"""
    
    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "so100_dualcam"
    """Data configuration name"""
    
    embodiment_tag: str = "new_embodiment"
    """Embodiment tag for the robot"""
    
    # Evaluation parameters
    num_trajectories: int = 5
    """Number of trajectories to evaluate"""
    
    steps_per_trajectory: int = 150
    """Number of steps to evaluate per trajectory"""
    
    action_horizon: int = 16
    """Action prediction horizon"""
    
    denoising_steps: int = 4
    """Number of denoising steps for diffusion model"""
    
    # Modality configuration
    modality_keys: List[str] = None
    """Modality keys to evaluate (default: ['single_arm', 'gripper'])"""
    
    # Visualization options
    plot: bool = True
    """Whether to generate and save plots"""
    
    save_videos: bool = False
    """Whether to save trajectory videos (requires additional setup)"""
    
    # Backend options
    video_backend: Literal["pyav", "torchvision_video", "torchvision_av"] = "torchvision_av"
    """Video decoding backend"""
    
    device: str = "auto"
    """Device to run evaluation on (auto, cuda, cpu)"""
    
    # WandB configuration
    wandb_project: str = "pippa"
    """WandB project name"""
    
    wandb_entity: str = "wild-ai"
    """WandB entity name"""
    
    wandb_tags: List[str] = None
    """Additional WandB tags"""
    
    wandb_run_name: Optional[str] = None
    """Custom WandB run name"""
    
    # Output options
    output_dir: str = "./gr00t-eval-results"
    """Directory to save evaluation outputs"""


def load_model_from_wandb(artifact_path: str, config: EvalConfig) -> tuple[BasePolicy, Dict[str, Any]]:
    """
    Load model from WandB artifact.
    
    Returns:
        policy: The loaded policy model
        metadata: Dictionary with artifact metadata
    """
    print(f"\nLoading model from WandB artifact: {artifact_path}")
    
    # Load environment variables for WandB API key
    load_dotenv(os.path.expanduser("~/pippa/.env"))
    
    # Parse artifact path
    parts = artifact_path.split("/")
    if len(parts) == 3:
        entity, project, artifact_full = parts
    else:
        entity = config.wandb_entity
        project = config.wandb_project
        artifact_full = artifact_path
    
    # Initialize WandB temporarily for artifact download
    wandb_run = wandb.init(
        project=project,
        entity=entity,
        job_type="download",
        settings=wandb.Settings(silent=True)
    )
    
    # Download artifact
    artifact = wandb_run.use_artifact(artifact_full)
    artifact_dir = artifact.download()
    
    # Finish the offline run
    wandb.finish()
    
    print(f"Downloaded artifact to: {artifact_dir}")
    
    # Get artifact metadata
    metadata = {
        "artifact_name": artifact.name,
        "artifact_version": artifact.version,
        "artifact_created_at": artifact.created_at,
        "artifact_dir": artifact_dir,
        "artifact_aliases": artifact.aliases if hasattr(artifact, 'aliases') else []
    }
    
    # Try to extract run ID from artifact metadata
    if hasattr(artifact, 'logged_by') and artifact.logged_by:
        try:
            metadata["source_run_id"] = artifact.logged_by.id
            print(f"Found source run ID from artifact: {artifact.logged_by.id}")
        except:
            pass
    
    # Check what files we have
    available_files = list(Path(artifact_dir).glob("*"))
    print("\nAvailable files in artifact:")
    for f in available_files:
        print(f"  - {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    
    # Load model configuration
    data_config = DATA_CONFIG_MAP[config.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    # For artifacts, we might need to handle missing base model files
    # The artifact typically contains only the fine-tuned weights
    base_model_path = "nvidia/GR00T-N1.5-3B"
    
    # Check if we have full model files or just fine-tuned components
    has_full_model = (Path(artifact_dir) / "model.safetensors.index.json").exists()
    
    if has_full_model:
        model_path = artifact_dir
        print("Using full model from artifact")
    else:
        # We'll need to load base model and apply fine-tuned weights
        model_path = base_model_path
        print(f"Using base model from HuggingFace: {base_model_path}")
        metadata["uses_base_model"] = True
        metadata["finetuned_weights_path"] = artifact_dir
    
    # Check if experiment_cfg exists, if not create a minimal one
    exp_cfg_dir = Path(artifact_dir) / "experiment_cfg"
    if not exp_cfg_dir.exists():
        print("Creating minimal experiment_cfg directory...")
        exp_cfg_dir.mkdir(exist_ok=True)
        
        # Create minimal metadata.json with proper format
        exp_metadata = {
            config.embodiment_tag: {
                "statistics": {
                    "state": {},
                    "action": {}
                }
            }
        }
        with open(exp_cfg_dir / "metadata.json", "w") as f:
            json.dump(exp_metadata, f, indent=2)
    
    # Create policy
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=config.embodiment_tag,
        denoising_steps=config.denoising_steps,
        device=get_device(config.device),
    )
    
    # If we only have fine-tuned weights, load them
    if not has_full_model:
        # Look for action head weights
        action_head_path = Path(artifact_dir) / "action_head_new_embodiment.pt"
        if action_head_path.exists():
            print(f"\nLoading fine-tuned action head from: {action_head_path}")
            action_head_state = torch.load(action_head_path, map_location=policy.device)
            
            # Load the weights into the model
            model_state = policy.model.state_dict()
            for name, param in action_head_state.items():
                if name in model_state:
                    model_state[name].copy_(param)
                    print(f"  Loaded: {name}")
            
            metadata["action_head_params"] = len(action_head_state)
    
    return policy, metadata


def setup_wandb_logging(config: EvalConfig, model_metadata: Dict[str, Any]) -> wandb.run:
    """Setup WandB for evaluation logging."""
    load_dotenv(os.path.expanduser("~/pippa/.env"))
    
    # Extract run ID from artifact if available
    resume_id = None
    
    # First check if we have source_run_id from artifact metadata
    if "source_run_id" in model_metadata:
        resume_id = model_metadata["source_run_id"]
        print(f"Using source run ID from artifact metadata: {resume_id}")
    elif "artifact_aliases" in model_metadata:
        # Look for run-{id} pattern in aliases
        import re
        for alias in model_metadata.get("artifact_aliases", []):
            run_id_match = re.search(r'run-([a-zA-Z0-9]+)', alias)
            if run_id_match:
                resume_id = run_id_match.group(1)
                print(f"Found run ID from artifact alias: {resume_id}")
                break
    
    # Fallback: try to extract from artifact path
    if not resume_id and config.wandb_artifact_path:
        import re
        run_id_match = re.search(r'run-([a-zA-Z0-9]+)', config.wandb_artifact_path)
        if run_id_match:
            resume_id = run_id_match.group(1)
            print(f"Found run ID from artifact path: {resume_id}")
    
    # Generate run name if not provided
    if config.wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_info = model_metadata.get("artifact_name", "local")
        config.wandb_run_name = f"gr00t-eval-{model_info}-{timestamp}"
    
    # Set default tags
    if config.wandb_tags is None:
        config.wandb_tags = ["gr00t-sft-eval", config.data_config]
    else:
        # Ensure gr00t-sft-eval tag is included
        if "gr00t-sft-eval" not in config.wandb_tags:
            config.wandb_tags.append("gr00t-sft-eval")
    
    # Initialize WandB - resume if we found a run ID
    if resume_id:
        print(f"Resuming WandB run: {resume_id}")
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            id=resume_id,
            resume="allow",
            name=config.wandb_run_name,
            tags=config.wandb_tags,
            config={
                **vars(config),
                **model_metadata,
                "eval_type": "open_loop_mse",
                "task": "so101_table_cleanup"
            },
            save_code=True,
        )
    else:
        # Create new run
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            tags=config.wandb_tags,
            config={
                **vars(config),
                **model_metadata,
                "eval_type": "open_loop_mse",
                "task": "so101_table_cleanup"
            },
            save_code=True,
        )
    
    return run


def evaluate_trajectory_with_logging(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    config: EvalConfig,
    save_dir: Path
) -> Dict[str, Any]:
    """
    Evaluate a single trajectory and return detailed metrics.
    
    Returns dictionary with:
        - mse: Overall MSE
        - per_joint_mse: MSE for each joint
        - max_error: Maximum error across trajectory
        - smoothness: Action smoothness metric
        - predictions: Predicted actions
        - ground_truth: Ground truth actions
    """
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    
    modality_keys = config.modality_keys or ["single_arm", "gripper"]
    
    print(f"\nEvaluating trajectory {traj_id}...")
    
    for step_count in tqdm(range(config.steps_per_trajectory), desc=f"Trajectory {traj_id}"):
        data_point = dataset.get_step_data(traj_id, step_count)
        
        # Concatenate states and actions across modalities
        concat_state = np.concatenate(
            [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        )
        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        )
        
        state_joints_across_time.append(concat_state)
        gt_action_across_time.append(concat_gt_action)
        
        # Get predictions at action horizon intervals
        if step_count % config.action_horizon == 0:
            action_chunk = policy.get_action(data_point)
            for j in range(config.action_horizon):
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)
    
    # Convert to arrays
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)[:config.steps_per_trajectory]
    
    # Calculate metrics
    errors = gt_action_across_time - pred_action_across_time
    mse = np.mean(errors ** 2)
    per_joint_mse = np.mean(errors ** 2, axis=0)
    max_error = np.max(np.abs(errors))
    
    # Calculate smoothness (sum of squared differences between consecutive actions)
    if len(pred_action_across_time) > 1:
        action_diffs = np.diff(pred_action_across_time, axis=0)
        smoothness = np.mean(action_diffs ** 2)
    else:
        smoothness = 0.0
    
    # Generate plots if requested
    if config.plot:
        plot_trajectory_comparison(
            gt_action_across_time,
            pred_action_across_time,
            state_joints_across_time,
            traj_id,
            save_dir
        )
    
    return {
        "trajectory_id": traj_id,
        "mse": float(mse),
        "per_joint_mse": per_joint_mse.tolist(),
        "max_error": float(max_error),
        "smoothness": float(smoothness),
        "num_steps": len(gt_action_across_time),
        "predictions": pred_action_across_time,
        "ground_truth": gt_action_across_time,
        "states": state_joints_across_time
    }


def plot_trajectory_comparison(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    states: np.ndarray,
    traj_id: int,
    save_dir: Path
) -> Path:
    """Generate comparison plots for trajectory."""
    action_dim = gt_actions.shape[1]
    
    # Create figure with subplots for each action dimension
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(12, 4 * action_dim))
    if action_dim == 1:
        axes = [axes]
    
    for i in range(action_dim):
        ax = axes[i]
        time_steps = np.arange(len(gt_actions))
        
        # Plot ground truth and predictions
        ax.plot(time_steps, gt_actions[:, i], 'b-', label='Ground Truth', alpha=0.8)
        ax.plot(time_steps, pred_actions[:, i], 'r--', label='Predicted', alpha=0.8)
        
        # Add error shading
        errors = np.abs(gt_actions[:, i] - pred_actions[:, i])
        ax.fill_between(time_steps, 
                       pred_actions[:, i] - errors, 
                       pred_actions[:, i] + errors,
                       alpha=0.2, color='red', label='Error')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'Action Dim {i}')
        ax.set_title(f'Action Dimension {i} - Trajectory {traj_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = save_dir / f"trajectory_{traj_id}_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def log_evaluation_results(
    results: List[Dict[str, Any]],
    config: EvalConfig,
    save_dir: Path,
    dataset: LeRobotSingleDataset = None
) -> None:
    """Log evaluation results to WandB with rich visualizations."""
    
    # Create comprehensive evaluation table with prompts and results
    eval_table = wandb.Table(columns=[
        "trajectory_id",
        "task_prompt",
        "mse", 
        "max_error",
        "smoothness",
        "trajectory_plot",
        "num_steps"
    ])
    
    # Create per-joint MSE table
    num_joints = len(results[0]["per_joint_mse"])
    joint_columns = ["trajectory_id"] + [f"joint_{i}_mse" for i in range(num_joints)]
    joint_mse_table = wandb.Table(columns=joint_columns)
    
    all_mse = []
    all_max_errors = []
    all_smoothness = []
    
    # Get task prompts from dataset if available
    task_prompts = {}
    if dataset is not None:
        try:
            # Load task prompts from dataset - check both possible filenames
            prompts_file = Path(config.dataset_path) / "meta" / "tasks.jsonl"
            if not prompts_file.exists():
                prompts_file = Path(config.dataset_path) / "meta" / "task_prompts.json"
            
            if prompts_file.exists():
                if prompts_file.suffix == '.jsonl':
                    # Load JSONL format
                    with open(prompts_file, 'r') as f:
                        for line in f:
                            item = json.loads(line.strip())
                            task_prompts[item["task_index"]] = item.get("task", item.get("task_description", f"Task {item['task_index']}"))
                else:
                    # Load JSON format
                    with open(prompts_file, 'r') as f:
                        task_prompts_data = json.load(f)
                        for item in task_prompts_data:
                            task_prompts[item["task_index"]] = item.get("task_description", f"Task {item['task_index']}")
        except Exception as e:
            print(f"Could not load task prompts: {e}")
    
    # Process each result
    for result in results:
        traj_id = result["trajectory_id"]
        
        # Get task prompt for this trajectory
        task_prompt = "Task prompt not available"
        if dataset is not None:
            try:
                # Get the task index for this trajectory
                sample = dataset.get_step_data(traj_id, 0)
                if "annotation.human.task_description" in sample:
                    task_value = sample["annotation.human.task_description"][0]
                    # Check if it's already a string (task description) or an index
                    if isinstance(task_value, str):
                        task_prompt = task_value
                    else:
                        # It's an index, look it up
                        task_prompt = task_prompts.get(int(task_value), f"Task index: {task_value}")
            except Exception as e:
                print(f"Could not get task prompt for trajectory {traj_id}: {e}")
        
        # Get trajectory plot if it exists
        plot_path = save_dir / f"trajectory_{traj_id}_comparison.png"
        trajectory_plot = None
        if plot_path.exists():
            trajectory_plot = wandb.Image(str(plot_path), caption=f"Trajectory {traj_id}")
        
        # Add to comprehensive table
        eval_table.add_data(
            traj_id,
            task_prompt,
            result["mse"],
            result["max_error"],
            result["smoothness"],
            trajectory_plot,
            result["num_steps"]
        )
        
        # Add to joint MSE table
        joint_row = [result["trajectory_id"]] + result["per_joint_mse"]
        joint_mse_table.add_data(*joint_row)
        
        # Collect for summary stats
        all_mse.append(result["mse"])
        all_max_errors.append(result["max_error"])
        all_smoothness.append(result["smoothness"])
    
    # Calculate summary statistics with eval_prompt/ prefix
    summary_stats = {
        "eval_prompt/mean_mse": np.mean(all_mse),
        "eval_prompt/std_mse": np.std(all_mse),
        "eval_prompt/min_mse": np.min(all_mse),
        "eval_prompt/max_mse": np.max(all_mse),
        "eval_prompt/mean_max_error": np.mean(all_max_errors),
        "eval_prompt/mean_smoothness": np.mean(all_smoothness),
        "eval_prompt/num_trajectories": len(results)
    }
    
    # Create MSE distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE distribution
    ax1.hist(all_mse, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(all_mse), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_mse):.4f}')
    ax1.set_xlabel('MSE')
    ax1.set_ylabel('Count')
    ax1.set_title('MSE Distribution Across Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Per-joint average MSE
    avg_joint_mse = np.mean([r["per_joint_mse"] for r in results], axis=0)
    joint_indices = np.arange(len(avg_joint_mse))
    ax2.bar(joint_indices, avg_joint_mse, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Joint Index')
    ax2.set_ylabel('Average MSE')
    ax2.set_title('Average MSE per Joint')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_plot_path = save_dir / "evaluation_summary.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create final comprehensive results table with summary image
    results_table = wandb.Table(columns=[
        "evaluation_type",
        "mean_mse",
        "num_trajectories", 
        "summary_plot",
        "dataset"
    ])
    
    results_table.add_data(
        "open_loop_mse",
        np.mean(all_mse),
        len(results),
        wandb.Image(str(summary_plot_path), caption="Evaluation Summary"),
        config.dataset_path
    )
    
    # Log all tables and metrics
    wandb.log({
        "eval_prompt_tables/trajectory_results": eval_table,
        "eval_prompt_tables/per_joint_mse": joint_mse_table,
        "eval_prompt_tables/evaluation_summary": results_table,
        **summary_stats
    })
    
    # Update summary
    wandb.summary.update(summary_stats)
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for key, value in summary_stats.items():
        print(f"{key}: {value:.4f}")
    print("="*50)


def get_device(device: str) -> str:
    """Get the appropriate device for evaluation."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main(config: EvalConfig):
    """Main evaluation function."""
    # Validate inputs
    if config.model_path is None and config.wandb_artifact_path is None:
        raise ValueError("Must provide either --model-path or --wandb-artifact-path")
    
    if config.model_path and config.wandb_artifact_path:
        print("Warning: Both model paths provided, using WandB artifact")
    
    # Set modality keys default
    if config.modality_keys is None:
        config.modality_keys = ["single_arm", "gripper"]
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_metadata = {}
    if config.wandb_artifact_path:
        policy, model_metadata = load_model_from_wandb(config.wandb_artifact_path, config)
    else:
        # Load from local path
        data_config = DATA_CONFIG_MAP[config.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        
        policy = Gr00tPolicy(
            model_path=config.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=config.embodiment_tag,
            denoising_steps=config.denoising_steps,
            device=get_device(config.device),
        )
        model_metadata["model_path"] = config.model_path
    
    # Setup WandB logging
    wandb_run = setup_wandb_logging(config, model_metadata)
    
    # Load dataset
    print(f"\nLoading dataset from: {config.dataset_path}")
    os.environ["VIDEO_BACKEND"] = config.video_backend
    
    modality = policy.get_modality_config()
    dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality,
        video_backend=config.video_backend,
        transforms=None,  # Handled by policy
        embodiment_tag=config.embodiment_tag,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Total trajectories: {len(dataset.trajectory_lengths)}")
    print(f"Trajectory lengths: {dataset.trajectory_lengths[:5]}..." if len(dataset.trajectory_lengths) > 5 else dataset.trajectory_lengths)
    
    # Limit number of trajectories to evaluate
    num_trajs = min(config.num_trajectories, len(dataset.trajectory_lengths))
    print(f"\nEvaluating {num_trajs} trajectories...")
    
    # Run evaluation
    results = []
    for traj_id in range(num_trajs):
        result = evaluate_trajectory_with_logging(
            policy, dataset, traj_id, config, output_dir
        )
        results.append(result)
        
        # Log intermediate progress with eval_prompt/ prefix
        current_avg_mse = np.mean([r["mse"] for r in results])
        wandb.log({
            "eval_prompt/trajectory_mse": result["mse"],
            "eval_prompt/running_avg_mse": current_avg_mse,
            "eval_prompt/trajectories_completed": len(results)
        })
    
    # Log final results with dataset
    log_evaluation_results(results, config, output_dir, dataset)
    
    # Save raw results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in results:
            json_result = {k: v for k, v in r.items() 
                          if k not in ["predictions", "ground_truth", "states"]}
            json_results.append(json_result)
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Finish WandB run
    wandb.finish()
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    main(config)