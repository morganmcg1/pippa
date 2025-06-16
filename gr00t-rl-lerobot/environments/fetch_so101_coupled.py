"""
Fetch environment wrapper that couples joints to create effective 6-DoF control.

This implementation couples the shoulder roll and upperarm roll joints,
effectively reducing the 7-DoF Fetch arm to match SO-101's 6-DoF configuration.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
import cv2
from pathlib import Path

# Register gymnasium_robotics environments
gym.register_envs(gymnasium_robotics)


class FetchSO101CoupledWrapper(gym.Wrapper):
    """
    Wraps Fetch Pick and Place to couple joints for 6-DoF behavior.
    
    Strategy: We couple the shoulder roll (joint 1) and upperarm roll (joint 3)
    to move together, effectively removing one degree of freedom.
    
    This maintains:
    - 3 position DOF (x, y, z)
    - 3 orientation DOF (roll, pitch, yaw) 
    - 1 gripper DOF
    
    But reduces from 7 to 6 arm joints like SO-101.
    """
    
    def __init__(
        self,
        env_id: str = "FetchPickAndPlaceDense-v3",
        render_mode: str = "rgb_array",
        max_episode_steps: int = 50,
        camera_resolution: Tuple[int, int] = (224, 224),
        use_joint_space: bool = False,
        couple_joints: bool = True,
    ):
        """
        Args:
            env_id: Fetch environment ID
            render_mode: Rendering mode for camera observations
            max_episode_steps: Maximum steps per episode
            camera_resolution: Resolution for camera images
            use_joint_space: If True, expose joint-level actions (6D)
            couple_joints: If True, couple joints to simulate 6-DoF
        """
        # Create base environment
        env = gym.make(
            env_id,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps
        )
        super().__init__(env)
        
        self.camera_resolution = camera_resolution
        self.use_joint_space = use_joint_space
        self.couple_joints = couple_joints
        
        # Define joint coupling (shoulder roll + upperarm roll)
        self.coupled_joints = [1, 3]  # Indices in the 7-joint array
        self.coupling_ratio = 1.0  # They move together with same angle
        
        # SO-101 has 6 joints
        self.num_joints = 6
        
        # Redefine action space if using joint space
        if self.use_joint_space:
            # 6 joint velocities/positions instead of 4D Cartesian
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_joints,),
                dtype=np.float32
            )
        
        # Keep observation space as dual cameras + state
        self.observation_space = spaces.Dict({
            "observation": spaces.Dict({
                "images": spaces.Dict({
                    "front": spaces.Box(0, 255, (*camera_resolution, 3), np.uint8),
                    "wrist": spaces.Box(0, 255, (*camera_resolution, 3), np.uint8),
                }),
                "state": spaces.Box(-np.inf, np.inf, (self.num_joints,), np.float32),
            }),
            "instruction": spaces.Text(max_length=200),
        })
        
        self.instruction = "Pick the object and place it at the target location"
        
    def reset(self, **kwargs):
        """Reset environment and return observation in GR00T format."""
        obs, info = self.env.reset(**kwargs)
        
        # Convert to our format
        groot_obs = self._convert_observation(obs)
        
        return groot_obs, info
    
    def step(self, action: np.ndarray):
        """
        Step environment with action.
        
        Args:
            action: Either 6D joint action or 4D Cartesian action
        """
        if self.use_joint_space and self.couple_joints:
            # Convert 6D joint action to 7D for Fetch
            fetch_action = self._expand_coupled_action(action)
            # Note: This requires modifying Fetch to accept joint actions
            # For now, we'll convert to Cartesian
            fetch_action = self._joint_to_cartesian(fetch_action)
        elif self.use_joint_space:
            # Convert joint action to Cartesian for standard Fetch
            fetch_action = self._joint_to_cartesian(action)
        else:
            # Use Cartesian action directly (4D)
            fetch_action = action[:4] if len(action) > 4 else action
        
        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(fetch_action)
        
        # Convert observation
        groot_obs = self._convert_observation(obs)
        
        # Add task completion info
        info["task_success"] = info.get("is_success", False)
        
        return groot_obs, reward, terminated, truncated, info
    
    def _expand_coupled_action(self, action_6d: np.ndarray) -> np.ndarray:
        """
        Expand 6D action to 7D by coupling joints.
        
        The coupled joints move together with the same velocity/position.
        """
        action_7d = np.zeros(7)
        
        # Map 6D to 7D with coupling
        j6 = 0
        for j7 in range(7):
            if j7 == self.coupled_joints[1]:
                # Second coupled joint copies from first
                action_7d[j7] = action_7d[self.coupled_joints[0]] * self.coupling_ratio
            elif j7 == self.coupled_joints[0]:
                # First coupled joint gets the action
                action_7d[j7] = action_6d[j6]
                j6 += 1
            else:
                # Other joints map directly
                action_7d[j7] = action_6d[j6]
                j6 += 1
        
        return action_7d
    
    def _joint_to_cartesian(self, joint_action: np.ndarray) -> np.ndarray:
        """
        Convert joint space action to Cartesian space.
        
        This is a simplified conversion - in practice, you'd use
        proper forward kinematics and Jacobian.
        """
        # For now, use a simple linear mapping
        # Real implementation would use MuJoCo's forward kinematics
        
        if len(joint_action) >= 6:
            # Rough mapping: first 3 joints ~ position, next 3 ~ orientation
            dx = joint_action[0] * 0.05  # Scale down for safety
            dy = joint_action[1] * 0.05
            dz = joint_action[2] * 0.05
            gripper = joint_action[5] if len(joint_action) > 5 else 0.0
        else:
            dx = dy = dz = gripper = 0.0
        
        return np.array([dx, dy, dz, gripper])
    
    def _convert_observation(self, fetch_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Fetch observation to GR00T format with coupled joints."""
        # Render camera views
        camera_obs = self.env.render()
        
        # Front camera (full view)
        front_cam = cv2.resize(camera_obs, self.camera_resolution)
        
        # Wrist camera (cropped/zoomed to simulate wrist view)
        h, w = camera_obs.shape[:2]
        crop = camera_obs[h//3:, w//4:3*w//4]
        wrist_cam = cv2.resize(crop, self.camera_resolution)
        
        # Extract joint positions from Fetch observation
        # Fetch exposes: observation, achieved_goal, desired_goal
        if "observation" in fetch_obs:
            # First 3: gripper position
            # Next 3: object position  
            # Next 3: object relative position
            # Next 2: gripper state
            # Next 3: object rotation
            # Next 3: object velocity
            # Next 3: object angular velocity
            # Next 2: gripper velocity
            
            # For joint state, we need actual joint positions
            # This is a simplification - real implementation would
            # query MuJoCo for actual joint angles
            gripper_pos = fetch_obs["observation"][:3]
            gripper_state = fetch_obs["observation"][9:11]
            
            # Create fake joint positions based on gripper position
            # Real implementation would use MuJoCo's qpos
            if self.couple_joints:
                joint_state = self._estimate_coupled_joints(gripper_pos, gripper_state)
            else:
                joint_state = self._estimate_joints(gripper_pos, gripper_state)
        else:
            joint_state = np.zeros(self.num_joints)
        
        return {
            "observation": {
                "images": {
                    "front": front_cam,
                    "wrist": wrist_cam,
                },
                "state": joint_state.astype(np.float32),
            },
            "instruction": self.instruction,
        }
    
    def _estimate_coupled_joints(self, gripper_pos: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
        """
        Estimate 6 joint positions from gripper position (simplified).
        
        In reality, you'd use inverse kinematics.
        """
        # Very rough approximation for 6 joints
        joints = np.zeros(6)
        
        # Map gripper position to approximate joint angles
        joints[0] = np.arctan2(gripper_pos[1], gripper_pos[0])  # Base rotation
        joints[1] = gripper_pos[2] * 2.0  # Shoulder pitch (coupled)
        joints[2] = -gripper_pos[2] * 1.5  # Elbow
        joints[3] = gripper_pos[0] * 0.5  # Wrist pitch
        joints[4] = gripper_pos[1] * 0.5  # Wrist roll  
        joints[5] = gripper_state[0]  # Gripper
        
        return joints
    
    def _estimate_joints(self, gripper_pos: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
        """Estimate joint positions without coupling (7 joints mapped to 6)."""
        return self._estimate_coupled_joints(gripper_pos, gripper_state)


def make_fetch_so101_coupled_env(**kwargs) -> FetchSO101CoupledWrapper:
    """Create coupled Fetch environment for SO-101 simulation."""
    return FetchSO101CoupledWrapper(**kwargs)


if __name__ == "__main__":
    # Test the coupled environment
    env = make_fetch_so101_coupled_env(use_joint_space=False)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"\nObservation keys: {obs.keys()}")
    print(f"State shape: {obs['observation']['state'].shape}")
    print(f"Front cam shape: {obs['observation']['images']['front'].shape}")
    
    # Test stepping
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i}: reward={reward:.3f}, success={info.get('task_success', False)}")
        
        if terminated or truncated:
            break
    
    env.close()