"""
GR00T Policy wrapper for LeRobot integration.

This wraps our fine-tuned GR00T-N1.5-3B model to work with LeRobot's
policy interface for reinforcement learning.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import wandb
from dataclasses import dataclass
from dotenv import load_dotenv

# Add Isaac-GR00T to path
isaac_groot_path = os.path.expanduser("~/pippa/Isaac-GR00T")
if os.path.exists(isaac_groot_path):
    sys.path.insert(0, isaac_groot_path)

# Import GR00T components
try:
    from gr00t.data.schema import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import BasePolicy, Gr00tPolicy
    GROOT_AVAILABLE = True
except ImportError:
    print("Warning: Isaac-GR00T not found. Using placeholder implementation.")
    GROOT_AVAILABLE = False
    BasePolicy = None

# Note: These imports will need to be adjusted based on actual LeRobot structure
# For now, creating a standalone implementation
class PreTrainedPolicy(nn.Module):
    """Base class for LeRobot policies (placeholder)."""
    def __init__(self):
        super().__init__()
        
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
        
    def reset(self):
        pass


@dataclass
class GR00TConfig:
    """Configuration for GR00T policy."""
    # Model paths
    wandb_artifact_path: str = "wild-ai/pippa/gr00t-sft-so100_dualcam-bs32:v0"
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    
    # Model settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    action_dim: int = 6  # SO-101 has 6 joints
    embodiment_tag: str = "new_embodiment"
    data_config: str = "so100_dualcam"  # Data configuration for GR00T
    
    # Observation settings
    image_size: Tuple[int, int] = (224, 224)
    normalize_observations: bool = True
    
    # Action settings
    action_horizon: int = 16  # Number of future actions to predict
    n_action_steps: int = 1   # Number of actions to execute
    denoising_steps: int = 4  # Diffusion denoising steps
    
    # Inference settings
    temperature: float = 1.0
    num_inference_steps: int = 50  # For diffusion


class GR00TPolicy(PreTrainedPolicy):
    """
    GR00T policy wrapper for LeRobot.
    
    This wraps our fine-tuned GR00T-N1.5-3B model to work with
    LeRobot's reinforcement learning framework.
    """
    
    def __init__(self, config: GR00TConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Placeholder for GR00T model - will be loaded from WandB
        self.model = None
        self.action_head = None
        self.projector = None
        
        # Observation and action buffers
        self.observation_buffer = []
        self.action_buffer = []
        
        # Load model from WandB artifact
        self._load_model()
        
        # Move to device
        self.to(self.device)
        
    def _load_model(self):
        """Load fine-tuned GR00T model from WandB artifact."""
        print(f"Loading GR00T model from {self.config.wandb_artifact_path}")
        
        if GROOT_AVAILABLE:
            # Load actual GR00T model
            self._load_groot_model()
        else:
            # Fallback to dummy networks
            print("Warning: Using placeholder networks. Install Isaac-GR00T for actual model.")
            self._create_dummy_networks()
    
    def _load_groot_model(self):
        """Load actual GR00T model using Isaac-GR00T."""
        # Load environment variables
        load_dotenv(os.path.expanduser("~/pippa/.env"))
        
        # Get data config
        data_config = DATA_CONFIG_MAP[self.config.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        
        # Parse artifact path
        if "/" in self.config.wandb_artifact_path:
            parts = self.config.wandb_artifact_path.split("/")
            if len(parts) == 3:
                entity, project, artifact_full = parts
            else:
                entity = "wild-ai"
                project = "pippa"
                artifact_full = self.config.wandb_artifact_path
        else:
            entity = "wild-ai"
            project = "pippa"
            artifact_full = self.config.wandb_artifact_path
        
        # Download artifact
        print(f"Downloading WandB artifact: {entity}/{project}/{artifact_full}")
        
        # Initialize WandB temporarily
        run = wandb.init(
            project=project,
            entity=entity,
            job_type="download",
            settings=wandb.Settings(silent=True)
        )
        
        artifact = run.use_artifact(artifact_full)
        artifact_dir = artifact.download()
        wandb.finish()
        
        print(f"Downloaded to: {artifact_dir}")
        
        # Check if we have full model or just fine-tuned components
        has_full_model = (Path(artifact_dir) / "model.safetensors.index.json").exists()
        
        if has_full_model:
            model_path = artifact_dir
        else:
            # Use base model from HuggingFace
            model_path = "nvidia/GR00T-N1.5-3B"
            print(f"Using base model: {model_path}")
        
        # Create minimal experiment_cfg if needed
        exp_cfg_dir = Path(artifact_dir) / "experiment_cfg"
        if not exp_cfg_dir.exists():
            exp_cfg_dir.mkdir(exist_ok=True)
            exp_metadata = {
                self.config.embodiment_tag: {
                    "statistics": {
                        "state": {},
                        "action": {}
                    }
                }
            }
            import json
            with open(exp_cfg_dir / "metadata.json", "w") as f:
                json.dump(exp_metadata, f, indent=2)
        
        # Create GR00T policy
        self.groot_policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=self.config.embodiment_tag,
            denoising_steps=self.config.denoising_steps,
            device=self.device,
        )
        
        # Load fine-tuned weights if needed
        if not has_full_model:
            action_head_path = Path(artifact_dir) / "action_head_new_embodiment.pt"
            if action_head_path.exists():
                print(f"Loading fine-tuned action head from: {action_head_path}")
                action_head_state = torch.load(action_head_path, map_location=self.device)
                
                # Apply weights
                model_state = self.groot_policy.model.state_dict()
                for name, param in action_head_state.items():
                    if name in model_state:
                        model_state[name].copy_(param)
                        print(f"  Loaded: {name}")
        
        # Extract model components for our wrapper
        self.model = self.groot_policy.model
        self.modality_config = modality_config
        self.modality_transform = modality_transform
        
        print("GR00T model loaded successfully!")
        
    def _create_dummy_networks(self):
        """Create dummy networks for testing (replace with actual GR00T)."""
        # Vision encoder (simplified)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        
        # Combined feature processing
        self.feature_combiner = nn.Sequential(
            nn.Linear(512 * 2 + 256, 512),  # 2 cameras + state
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        
        # Action prediction head (simplified diffusion)
        self.action_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.config.action_dim * self.config.action_horizon),
        )
        
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Select action given observations.
        
        Args:
            batch: Dictionary containing:
                - observation.images.front: Front camera image
                - observation.images.wrist: Wrist camera image
                - observation.state: Joint positions
                - instruction: Language instruction (optional)
                
        Returns:
            Action tensor of shape (batch_size, action_dim)
        """
        with torch.no_grad():
            if GROOT_AVAILABLE and hasattr(self, 'groot_policy'):
                # Use actual GR00T model
                return self._select_action_groot(batch)
            else:
                # Use placeholder implementation
                return self._select_action_dummy(batch)
    
    def _select_action_groot(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action using actual GR00T model."""
        # Convert batch format to GR00T format
        groot_batch = self._convert_batch_to_groot(batch)
        
        # Get action from GR00T
        action_dict = self.groot_policy.get_action(groot_batch)
        
        # Extract actions for each modality
        actions = []
        for key in ["single_arm", "gripper"]:
            if f"action.{key}" in action_dict:
                action = action_dict[f"action.{key}"]
                # Take first action from horizon
                if isinstance(action, np.ndarray):
                    action = action[0] if action.ndim > 1 else action
                else:
                    action = action[0].cpu().numpy() if action.dim() > 1 else action.cpu().numpy()
                actions.append(action)
        
        # Concatenate actions
        if actions:
            action_np = np.concatenate([np.atleast_1d(a) for a in actions], axis=0)
            action_tensor = torch.from_numpy(action_np).float().to(self.device)
            
            # Ensure batch dimension
            if action_tensor.dim() == 1:
                action_tensor = action_tensor.unsqueeze(0)
            
            return action_tensor
        else:
            # Fallback if no actions found
            return torch.zeros(1, self.config.action_dim, device=self.device)
    
    def _convert_batch_to_groot(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Convert LeRobot batch format to GR00T format."""
        groot_batch = {}
        
        # Handle images
        if "observation" in batch and "images" in batch["observation"]:
            # GR00T expects numpy arrays for images
            if "front" in batch["observation"]["images"]:
                front_img = batch["observation"]["images"]["front"]
                if isinstance(front_img, torch.Tensor):
                    front_img = front_img.cpu().numpy()
                    if front_img.ndim == 4:  # Batch dimension
                        front_img = front_img[0]
                groot_batch["observation.images.front"] = [front_img]
            
            if "wrist" in batch["observation"]["images"]:
                wrist_img = batch["observation"]["images"]["wrist"]
                if isinstance(wrist_img, torch.Tensor):
                    wrist_img = wrist_img.cpu().numpy()
                    if wrist_img.ndim == 4:
                        wrist_img = wrist_img[0]
                groot_batch["observation.images.wrist"] = [wrist_img]
        
        # Handle state
        if "observation" in batch and "state" in batch["observation"]:
            state = batch["observation"]["state"]
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
                if state.ndim == 2:
                    state = state[0]
            
            # Split state into modalities (5 joints + 1 gripper for SO-101)
            groot_batch["state.single_arm"] = [state[:5]]
            groot_batch["state.gripper"] = [state[5:]]
        
        # Handle instruction
        if "instruction" in batch:
            groot_batch["annotation.human.task_description"] = [batch["instruction"]]
        
        return groot_batch
    
    def _select_action_dummy(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action using dummy networks (original implementation)."""
        # Move batch to device
        batch = self._batch_to_device(batch)
        
        # Extract observations
        front_img = batch["observation"]["images"]["front"]
        wrist_img = batch["observation"]["images"]["wrist"]
        state = batch["observation"]["state"]
        
        # Ensure batch dimension
        if front_img.dim() == 3:
            front_img = front_img.unsqueeze(0)
            wrist_img = wrist_img.unsqueeze(0)
            state = state.unsqueeze(0)
        
        # Normalize images to [0, 1]
        front_img = front_img.float() / 255.0
        wrist_img = wrist_img.float() / 255.0
        
        # Encode visual features
        front_features = self.vision_encoder(front_img.permute(0, 3, 1, 2))
        wrist_features = self.vision_encoder(wrist_img.permute(0, 3, 1, 2))
        
        # Encode state
        state_features = self.state_encoder(state)
        
        # Combine features
        combined = torch.cat([front_features, wrist_features, state_features], dim=-1)
        features = self.feature_combiner(combined)
        
        # Predict actions
        action_predictions = self.action_predictor(features)
        action_predictions = action_predictions.view(-1, self.config.action_horizon, self.config.action_dim)
        
        # Take first action from horizon
        action = action_predictions[:, 0, :]
        
        # Clip to action space bounds
        action = torch.clamp(action, -1.0, 1.0)
        
        return action
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (computes loss).
        
        Args:
            batch: Training batch with observations and target actions
            
        Returns:
            Dictionary with loss and metrics
        """
        if GROOT_AVAILABLE and hasattr(self, 'groot_policy'):
            # Use actual GR00T model forward pass
            groot_batch = self._convert_batch_to_groot(batch)
            
            # Add target actions to batch
            if "action" in batch:
                target_actions = batch["action"]
                if isinstance(target_actions, torch.Tensor):
                    target_actions = target_actions.cpu().numpy()
                    if target_actions.ndim == 3:
                        target_actions = target_actions[0]  # Remove batch dim
                
                # Split actions into modalities
                groot_batch["action.single_arm"] = target_actions[:, :5]
                groot_batch["action.gripper"] = target_actions[:, 5:]
            
            # Forward through GR00T
            outputs = self.groot_policy.model(groot_batch)
            
            # Extract loss and metrics
            loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
            
            return {
                "loss": loss,
                "metrics": {
                    "loss": loss.item() if hasattr(loss, 'item') else float(loss),
                }
            }
        else:
            # Use dummy implementation
            return self._forward_dummy(batch)
    
    def _forward_dummy(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass using dummy networks."""
        # Extract inputs
        front_img = batch["observation"]["images"]["front"]
        wrist_img = batch["observation"]["images"]["wrist"] 
        state = batch["observation"]["state"]
        target_actions = batch["action"]
        
        # Normalize images
        front_img = front_img.float() / 255.0
        wrist_img = wrist_img.float() / 255.0
        
        # Encode features
        front_features = self.vision_encoder(front_img.permute(0, 3, 1, 2))
        wrist_features = self.vision_encoder(wrist_img.permute(0, 3, 1, 2))
        state_features = self.state_encoder(state)
        
        # Combine and process
        combined = torch.cat([front_features, wrist_features, state_features], dim=-1)
        features = self.feature_combiner(combined)
        
        # Predict actions
        action_predictions = self.action_predictor(features)
        action_predictions = action_predictions.view(-1, self.config.action_horizon, self.config.action_dim)
        
        # Compute loss (MSE for now, would use diffusion loss for actual GR00T)
        # Only compute loss for first n_action_steps
        pred_actions = action_predictions[:, :self.config.n_action_steps, :]
        target_actions = target_actions[:, :self.config.n_action_steps, :]
        
        loss = nn.functional.mse_loss(pred_actions, target_actions)
        
        return {
            "loss": loss,
            "metrics": {
                "mse": loss.item(),
                "action_mean": pred_actions.mean().item(),
                "action_std": pred_actions.std().item(),
            }
        }
    
    def reset(self):
        """Reset internal buffers."""
        self.observation_buffer.clear()
        self.action_buffer.clear()
        
    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively move batch to device."""
        if isinstance(batch, dict):
            return {k: self._batch_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: Optional[GR00TConfig] = None):
        """Load a pretrained GR00T policy."""
        if config is None:
            config = GR00TConfig()
        config.wandb_artifact_path = checkpoint_path
        return cls(config)
    
    def save_pretrained(self, save_path: str):
        """Save policy weights."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }, save_path / "policy.pt")
        
        print(f"Policy saved to {save_path}")


# Integration with actual GR00T model (placeholder)
class GR00TModelWrapper:
    """
    Placeholder for actual GR00T model integration.
    
    In practice, this would:
    1. Load the Isaac-GR00T model architecture
    2. Load fine-tuned weights from WandB
    3. Handle the diffusion action head properly
    4. Implement proper observation preprocessing
    """
    pass