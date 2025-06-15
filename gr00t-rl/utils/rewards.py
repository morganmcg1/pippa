#!/usr/bin/env python3
"""
Reward functions for robot tasks.
Designed to be verifiable for GRPO training.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple


class TaskReward:
    """Base class for task-specific reward functions."""
    
    def __init__(self, sparse: bool = False):
        """
        Args:
            sparse: If True, only give reward on success
        """
        self.sparse = sparse
        
    def compute(
        self, 
        obs: Dict[str, Any], 
        action: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute reward for current state."""
        raise NotImplementedError


class PickAndPlaceReward(TaskReward):
    """Reward for pick and place tasks."""
    
    def __init__(
        self, 
        sparse: bool = False,
        success_threshold: float = 0.05,
        height_bonus: float = 0.1,
        distance_penalty: float = 1.0
    ):
        super().__init__(sparse)
        self.success_threshold = success_threshold
        self.height_bonus = height_bonus
        self.distance_penalty = distance_penalty
        
    def compute(
        self, 
        obs: Dict[str, Any], 
        action: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute pick and place reward.
        
        Rewards:
        - Distance to object (before grasping)
        - Height of object (after grasping)
        - Distance to goal (while carrying)
        - Success bonus (object at goal)
        """
        object_pos = obs.get("_object_position", np.zeros(3))
        goal_pos = obs.get("_goal_position", np.zeros(3))
        gripper_pos = obs.get("_gripper_position", np.zeros(3))
        is_grasped = obs.get("_is_grasped", False)
        
        if self.sparse:
            # Sparse reward: only on success
            distance_to_goal = np.linalg.norm(object_pos - goal_pos)
            return float(distance_to_goal < self.success_threshold)
        
        # Dense reward computation
        reward = 0.0
        
        if not is_grasped:
            # Reward for getting close to object
            distance_to_object = np.linalg.norm(gripper_pos - object_pos)
            reward -= self.distance_penalty * distance_to_object
        else:
            # Reward for lifting object
            object_height = object_pos[2]
            reward += self.height_bonus * object_height
            
            # Reward for moving toward goal
            distance_to_goal = np.linalg.norm(object_pos - goal_pos)
            reward -= self.distance_penalty * distance_to_goal
            
            # Success bonus
            if distance_to_goal < self.success_threshold:
                reward += 10.0
                
        return reward


class PushToGoalReward(TaskReward):
    """Reward for pushing tasks."""
    
    def __init__(
        self,
        sparse: bool = False,
        success_threshold: float = 0.05,
        progress_weight: float = 1.0
    ):
        super().__init__(sparse)
        self.success_threshold = success_threshold
        self.progress_weight = progress_weight
        self.initial_distance = None
        
    def compute(
        self, 
        obs: Dict[str, Any], 
        action: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute push to goal reward.
        
        Rewards:
        - Progress toward goal
        - Contact maintenance
        - Success bonus
        """
        object_pos = obs.get("_object_position", np.zeros(3))
        goal_pos = obs.get("_goal_position", np.zeros(3))
        
        distance_to_goal = np.linalg.norm(object_pos[:2] - goal_pos[:2])  # 2D distance
        
        # Initialize initial distance
        if self.initial_distance is None:
            self.initial_distance = distance_to_goal
            
        if self.sparse:
            # Sparse reward: only on success
            return float(distance_to_goal < self.success_threshold)
        
        # Dense reward computation
        reward = 0.0
        
        # Progress reward
        progress = (self.initial_distance - distance_to_goal) / self.initial_distance
        reward += self.progress_weight * progress
        
        # Penalty for distance (encourages completion)
        reward -= 0.1 * distance_to_goal
        
        # Success bonus
        if distance_to_goal < self.success_threshold:
            reward += 10.0
            
        return reward
    
    def reset(self):
        """Reset reward function state."""
        self.initial_distance = None


class StackBlocksReward(TaskReward):
    """Reward for stacking tasks."""
    
    def __init__(
        self,
        sparse: bool = False,
        num_blocks: int = 3,
        stack_threshold: float = 0.05,
        height_threshold: float = 0.02
    ):
        super().__init__(sparse)
        self.num_blocks = num_blocks
        self.stack_threshold = stack_threshold
        self.height_threshold = height_threshold
        
    def compute(
        self, 
        obs: Dict[str, Any], 
        action: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute stacking reward.
        
        Rewards:
        - Blocks aligned horizontally
        - Blocks stacked vertically
        - Stability bonus
        """
        # Assume block positions are provided as list
        block_positions = obs.get("_block_positions", [])
        
        if len(block_positions) < self.num_blocks:
            return 0.0
            
        if self.sparse:
            # Check if all blocks are stacked
            return float(self._check_stack_complete(block_positions))
        
        # Dense reward computation
        reward = 0.0
        
        # Reward for each correctly placed block
        for i in range(1, len(block_positions)):
            lower_block = block_positions[i-1]
            upper_block = block_positions[i]
            
            # Horizontal alignment
            xy_distance = np.linalg.norm(upper_block[:2] - lower_block[:2])
            if xy_distance < self.stack_threshold:
                reward += 1.0
                
            # Vertical spacing
            height_diff = upper_block[2] - lower_block[2]
            if abs(height_diff - self.height_threshold) < 0.01:
                reward += 0.5
                
        # Bonus for complete stack
        if self._check_stack_complete(block_positions):
            reward += 10.0
            
        return reward
    
    def _check_stack_complete(self, block_positions: list) -> bool:
        """Check if stack is complete."""
        if len(block_positions) < self.num_blocks:
            return False
            
        for i in range(1, len(block_positions)):
            lower_block = block_positions[i-1]
            upper_block = block_positions[i]
            
            # Check horizontal alignment
            xy_distance = np.linalg.norm(upper_block[:2] - lower_block[:2])
            if xy_distance > self.stack_threshold:
                return False
                
            # Check vertical spacing
            height_diff = upper_block[2] - lower_block[2]
            if height_diff < 0:  # Upper block is lower
                return False
                
        return True


class CompositeReward:
    """Combine multiple reward functions."""
    
    def __init__(self, reward_functions: Dict[str, Tuple[TaskReward, float]]):
        """
        Args:
            reward_functions: Dict mapping names to (reward_fn, weight) tuples
        """
        self.reward_functions = reward_functions
        
    def compute(
        self, 
        obs: Dict[str, Any], 
        action: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted sum of rewards.
        
        Returns:
            total_reward: Weighted sum
            reward_components: Individual components
        """
        total_reward = 0.0
        components = {}
        
        for name, (reward_fn, weight) in self.reward_functions.items():
            component_reward = reward_fn.compute(obs, action, info)
            components[name] = component_reward
            total_reward += weight * component_reward
            
        return total_reward, components