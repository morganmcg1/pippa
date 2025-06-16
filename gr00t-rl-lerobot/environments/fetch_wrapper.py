"""
Wrapper to adapt Gymnasium-Robotics Fetch environment for GR00T/SO-101 compatibility.

This wrapper handles:
1. Observation format conversion (Fetch state → GR00T dual camera + state)
2. Action space mapping (SO-101 6D joints → Fetch 4D Cartesian)
3. Goal conditioning for table cleanup tasks
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any, Optional
import cv2


class FetchToSO101Wrapper(gym.Wrapper):
    """
    Adapts Fetch Pick and Place environment to match SO-101/GR00T interface.
    
    Key adaptations:
    - Converts Fetch observations to GR00T format (dual cameras + state)
    - Maps between SO-101 joint actions and Fetch Cartesian actions
    - Provides language instructions for goal conditioning
    """
    
    def __init__(
        self,
        env_id: str = "FetchPickAndPlace-v3",
        render_mode: str = "rgb_array",
        max_episode_steps: int = 50,
        camera_resolution: Tuple[int, int] = (224, 224),
        instruction: str = "Pick the object and place it in the target location",
    ):
        """
        Initialize the wrapper.
        
        Args:
            env_id: Gymnasium environment ID
            render_mode: Rendering mode for camera observations
            max_episode_steps: Maximum steps per episode
            camera_resolution: Resolution for camera observations (H, W)
            instruction: Language instruction for the task
        """
        # Create base environment
        base_env = gym.make(
            env_id,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        super().__init__(base_env)
        
        self.camera_resolution = camera_resolution
        self.instruction = instruction
        
        # Define observation space to match GR00T expectations
        # Note: GR00T expects dict with images and state
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Dict({
                "images": gym.spaces.Dict({
                    "front": gym.spaces.Box(
                        low=0, high=255, 
                        shape=(*camera_resolution, 3), 
                        dtype=np.uint8
                    ),
                    "wrist": gym.spaces.Box(
                        low=0, high=255,
                        shape=(*camera_resolution, 3),
                        dtype=np.uint8
                    ),
                }),
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(6,),  # SO-101 has 6 joints
                    dtype=np.float32
                ),
            }),
            "instruction": gym.spaces.Text(max_length=200),
            "achieved_goal": base_env.observation_space["achieved_goal"],
            "desired_goal": base_env.observation_space["desired_goal"],
        })
        
        # SO-101 action space: 6 joint positions
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and convert observations."""
        obs, info = self.env.reset(**kwargs)
        gr00t_obs = self._convert_observation(obs)
        return gr00t_obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute action in environment.
        
        Args:
            action: SO-101 joint positions (6D)
            
        Returns:
            Standard Gym step returns with converted observations
        """
        # Convert SO-101 joint action to Fetch Cartesian action
        fetch_action = self._convert_action(action)
        
        # Execute in base environment
        obs, reward, terminated, truncated, info = self.env.step(fetch_action)
        
        # Convert observation to GR00T format
        gr00t_obs = self._convert_observation(obs)
        
        return gr00t_obs, reward, terminated, truncated, info
    
    def _convert_observation(self, fetch_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Fetch observation to GR00T format.
        
        Fetch provides:
        - observation: 25D vector with gripper pos, object pos, velocities
        - achieved_goal: current object position
        - desired_goal: target object position
        
        GR00T expects:
        - observation.images.front: Front camera view
        - observation.images.wrist: Wrist camera view  
        - observation.state: Joint positions
        - instruction: Language instruction
        """
        # Render camera views
        camera_obs = self.env.render()
        if camera_obs is None:
            # Fallback to synthetic images if rendering fails
            camera_obs = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Resize to expected resolution
        front_cam = cv2.resize(camera_obs, self.camera_resolution)
        # Simulate wrist camera with cropped/zoomed view
        h, w = camera_obs.shape[:2]
        crop = camera_obs[h//3:, w//4:3*w//4]
        wrist_cam = cv2.resize(crop, self.camera_resolution)
        
        # Extract approximate joint positions from Fetch observation
        # Fetch obs[0:3] is gripper position, we'll use as proxy for joints
        gripper_pos = fetch_obs["observation"][:3]
        gripper_fingers = fetch_obs["observation"][9:11]
        
        # Create synthetic joint positions (simplified mapping)
        joint_positions = np.array([
            gripper_pos[0],  # Base rotation (from x position)
            gripper_pos[2],  # Shoulder pitch (from z position)
            gripper_pos[1],  # Elbow (from y position)
            0.0,  # Wrist roll (not directly observable)
            0.0,  # Wrist pitch (not directly observable)
            gripper_fingers.mean(),  # Gripper
        ], dtype=np.float32)
        
        return {
            "observation": {
                "images": {
                    "front": front_cam.astype(np.uint8),
                    "wrist": wrist_cam.astype(np.uint8),
                },
                "state": joint_positions,
            },
            "instruction": self.instruction,
            "achieved_goal": fetch_obs["achieved_goal"],
            "desired_goal": fetch_obs["desired_goal"],
        }
    
    def _convert_action(self, so101_action: np.ndarray) -> np.ndarray:
        """
        Convert SO-101 joint action to Fetch Cartesian action.
        
        SO-101 action: 6D joint positions [-1, 1]
        Fetch action: 4D [dx, dy, dz, gripper] [-1, 1]
        
        This is a simplified mapping - in practice, would use proper
        forward kinematics for SO-101.
        """
        # Simple mapping: use first 3 joints as Cartesian displacement
        # Scale down for safety (Fetch expects small displacements)
        dx = so101_action[0] * 0.05  # Base rotation → x displacement
        dy = so101_action[2] * 0.05  # Elbow → y displacement  
        dz = so101_action[1] * 0.05  # Shoulder → z displacement
        gripper = so101_action[5]  # Gripper joint directly
        
        return np.array([dx, dy, dz, gripper], dtype=np.float32)
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Clean up resources."""
        self.env.close()


# Convenience function
def make_fetch_so101_env(**kwargs):
    """Create a Fetch environment wrapped for SO-101/GR00T compatibility."""
    return FetchToSO101Wrapper(**kwargs)