#!/usr/bin/env python3
"""
Collect demonstrations from Fetch environment for GR00T fine-tuning.
This script provides multiple collection methods:
1. Random policy (baseline)
2. Scripted pick-and-place policy
3. Pre-trained PPO policy
4. Human teleoperation (keyboard control)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from collections import defaultdict
import cv2
import pickle
from datetime import datetime

# Set up rendering
os.environ['MUJOCO_GL'] = 'osmesa'

sys.path.append(str(Path(__file__).parent.parent))

from environments.fetch_wrapper import FetchGoalWrapper


class FetchDemonstrationCollector:
    """Collect demonstrations from Fetch environment."""
    
    def __init__(
        self,
        env_id: str = "FetchPickAndPlace-v3",
        save_dir: str = "./fetch_demonstrations",
        image_size: Tuple[int, int] = (224, 224),
        fps: int = 25,
        max_episode_steps: int = 50
    ):
        """Initialize demonstration collector."""
        self.env_id = env_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.fps = fps
        self.max_episode_steps = max_episode_steps
        
        # Create environment
        self.env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps)
        self.env = FetchGoalWrapper(
            self.env,
            observation_mode="observation_goal",
            reward_mode="sparse",
            goal_in_observation=True
        )
        
        # Storage for demonstrations
        self.demonstrations = []
        self.episode_count = 0
        
    def collect_random_episodes(self, num_episodes: int = 10):
        """Collect episodes using random actions (baseline)."""
        print(f"\nCollecting {num_episodes} random episodes...")
        
        for ep in range(num_episodes):
            trajectory = self._collect_episode(policy="random")
            if trajectory is not None:
                self.demonstrations.append(trajectory)
                self.episode_count += 1
                print(f"Episode {self.episode_count}: {len(trajectory['observations'])} steps, "
                      f"success: {trajectory['success']}")
    
    def collect_scripted_episodes(self, num_episodes: int = 100):
        """Collect episodes using scripted pick-and-place policy."""
        print(f"\nCollecting {num_episodes} scripted episodes...")
        
        for ep in range(num_episodes):
            trajectory = self._collect_episode(policy="scripted")
            if trajectory is not None:
                self.demonstrations.append(trajectory)
                self.episode_count += 1
                success_rate = sum(d['success'] for d in self.demonstrations) / len(self.demonstrations)
                print(f"Episode {self.episode_count}: {len(trajectory['observations'])} steps, "
                      f"success: {trajectory['success']}, overall success rate: {success_rate:.2%}")
    
    def _collect_episode(self, policy: str = "random") -> Optional[Dict[str, Any]]:
        """Collect a single episode."""
        obs, info = self.env.reset()
        
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'images': [],
            'language': "Pick the cube and place it on the target",
            'success': False,
            'episode_length': 0
        }
        
        done = False
        step_count = 0
        
        while not done and step_count < self.max_episode_steps:
            # Get image
            image = self.env.render()
            image_resized = cv2.resize(image, self.image_size)
            trajectory['images'].append(image_resized)
            
            # Store observation
            trajectory['observations'].append(obs.copy())
            
            # Get action based on policy
            if policy == "random":
                action = self.env.action_space.sample()
            elif policy == "scripted":
                action = self._get_scripted_action(obs, info, step_count)
            else:
                raise ValueError(f"Unknown policy: {policy}")
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            
            obs = next_obs
            step_count += 1
        
        # Check if successful
        trajectory['success'] = info.get('is_success', False)
        trajectory['episode_length'] = step_count
        
        return trajectory
    
    def _get_scripted_action(self, obs: np.ndarray, info: Dict, step: int) -> np.ndarray:
        """
        Simple scripted policy for pick and place.
        Phases:
        1. Move above object
        2. Lower to object
        3. Close gripper
        4. Lift object
        5. Move to goal
        6. Lower to goal height
        7. Open gripper
        """
        # Extract positions from observation
        gripper_pos = obs[:3]  # First 3 values are gripper position
        object_pos = obs[3:6]   # Next 3 are object position
        goal_pos = info.get('desired_goal', obs[-3:])  # Goal position
        
        # Gripper state (fingers)
        gripper_open = obs[9] + obs[10]  # Sum of finger positions
        
        # Action: [dx, dy, dz, gripper_action]
        action = np.zeros(4)
        
        # Determine phase based on step and state
        object_gripped = gripper_open < 0.02 and np.linalg.norm(gripper_pos - object_pos) < 0.05
        
        if not object_gripped:
            # Phase 1-3: Pick up object
            if step < 10:
                # Move above object
                target = object_pos.copy()
                target[2] += 0.1  # 10cm above object
                action[:3] = np.clip(target - gripper_pos, -0.05, 0.05)
                action[3] = 1  # Open gripper
            elif step < 20:
                # Lower to object
                action[:3] = np.clip(object_pos - gripper_pos, -0.05, 0.05)
                action[3] = 1  # Keep open
            else:
                # Close gripper
                action[:3] = 0  # Stay in place
                action[3] = -1  # Close gripper
        else:
            # Phase 4-7: Place object
            if step < 30:
                # Lift object
                target = gripper_pos.copy()
                target[2] = 0.6  # Lift to safe height
                action[:3] = np.clip(target - gripper_pos, -0.05, 0.05)
                action[3] = -1  # Keep closed
            elif step < 35:
                # Move to above goal
                target = goal_pos.copy()
                target[2] = 0.6  # Stay high
                action[:3] = np.clip(target - gripper_pos, -0.05, 0.05)
                action[3] = -1  # Keep closed
            elif step < 40:
                # Lower to goal
                action[:3] = np.clip(goal_pos - gripper_pos, -0.05, 0.05)
                action[3] = -1  # Keep closed
            else:
                # Open gripper
                action[:3] = 0  # Stay in place
                action[3] = 1  # Open gripper
        
        return action
    
    def save_demonstrations(self):
        """Save collected demonstrations in LeRobot-compatible format."""
        if not self.demonstrations:
            print("No demonstrations to save!")
            return
        
        # Create dataset directory structure
        dataset_dir = self.save_dir / f"fetch_demos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "env_id": self.env_id,
            "num_episodes": len(self.demonstrations),
            "fps": self.fps,
            "image_size": self.image_size,
            "success_rate": sum(d['success'] for d in self.demonstrations) / len(self.demonstrations),
            "total_steps": sum(d['episode_length'] for d in self.demonstrations),
            "collection_date": datetime.now().isoformat()
        }
        
        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save episodes
        episodes_dir = dataset_dir / "episodes"
        episodes_dir.mkdir(exist_ok=True)
        
        for i, demo in enumerate(self.demonstrations):
            episode_dir = episodes_dir / f"episode_{i:05d}"
            episode_dir.mkdir(exist_ok=True)
            
            # Save trajectory data
            trajectory_data = {
                'observations': np.array(demo['observations']),
                'actions': np.array(demo['actions']),
                'rewards': np.array(demo['rewards']),
                'language': demo['language'],
                'success': demo['success'],
                'episode_length': demo['episode_length']
            }
            
            with open(episode_dir / "trajectory.pkl", 'wb') as f:
                pickle.dump(trajectory_data, f)
            
            # Save images
            images_dir = episode_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            for j, img in enumerate(demo['images']):
                cv2.imwrite(str(images_dir / f"frame_{j:05d}.png"), 
                           cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        print(f"\nSaved {len(self.demonstrations)} demonstrations to: {dataset_dir}")
        print(f"Success rate: {metadata['success_rate']:.2%}")
        print(f"Total steps: {metadata['total_steps']}")
        
        # Create modality.json for GR00T compatibility
        self._create_modality_json(dataset_dir)
        
        return dataset_dir
    
    def _create_modality_json(self, dataset_dir: Path):
        """Create modality.json for GR00T training."""
        modality = {
            "embodiment": "fetch",
            "version": "1.0",
            "observation": {
                "images": {
                    "front": {
                        "shape": list(self.image_size) + [3],
                        "fps": self.fps,
                        "encoding": "rgb"
                    }
                },
                "state": {
                    "shape": [28],  # Fetch observation dimension
                    "names": [
                        "gripper_pos_x", "gripper_pos_y", "gripper_pos_z",
                        "object_pos_x", "object_pos_y", "object_pos_z",
                        "object_rel_x", "object_rel_y", "object_rel_z",
                        "gripper_finger_right", "gripper_finger_left",
                        "object_rot_x", "object_rot_y", "object_rot_z",
                        "object_vel_x", "object_vel_y", "object_vel_z",
                        "object_angvel_x", "object_angvel_y", "object_angvel_z",
                        "gripper_vel_x", "gripper_vel_y", "gripper_vel_z",
                        "gripper_finger_vel_right", "gripper_finger_vel_left",
                        "goal_x", "goal_y", "goal_z"
                    ]
                }
            },
            "action": {
                "shape": [4],
                "names": ["dx", "dy", "dz", "gripper"],
                "range": [
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0]
                ]
            },
            "language": {
                "type": "instruction",
                "examples": [
                    "Pick the cube and place it on the target",
                    "Move the block to the goal position",
                    "Grasp the object and put it at the target location"
                ]
            }
        }
        
        with open(dataset_dir / "modality.json", 'w') as f:
            json.dump(modality, f, indent=2)
        
        print(f"Created modality.json for GR00T compatibility")


def main():
    """Main function to collect demonstrations."""
    parser = argparse.ArgumentParser(description="Collect Fetch demonstrations for GR00T")
    parser.add_argument("--env-id", type=str, default="FetchPickAndPlace-v3",
                        help="Gymnasium environment ID")
    parser.add_argument("--policy", type=str, default="scripted",
                        choices=["random", "scripted", "mixed"],
                        help="Policy to use for collection")
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument("--save-dir", type=str, default="./fetch_demonstrations",
                        help="Directory to save demonstrations")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224],
                        help="Image size for saving")
    parser.add_argument("--fps", type=int, default=25,
                        help="FPS for data collection")
    parser.add_argument("--max-episode-steps", type=int, default=50,
                        help="Maximum steps per episode")
    args = parser.parse_args()
    
    # Create collector
    collector = FetchDemonstrationCollector(
        env_id=args.env_id,
        save_dir=args.save_dir,
        image_size=tuple(args.image_size),
        fps=args.fps,
        max_episode_steps=args.max_episode_steps
    )
    
    print(f"Collecting demonstrations for {args.env_id}")
    print(f"Policy: {args.policy}")
    print(f"Target episodes: {args.num_episodes}")
    print("-" * 50)
    
    # Collect demonstrations based on policy
    if args.policy == "random":
        collector.collect_random_episodes(args.num_episodes)
    elif args.policy == "scripted":
        collector.collect_scripted_episodes(args.num_episodes)
    elif args.policy == "mixed":
        # Collect some random and some scripted
        collector.collect_random_episodes(args.num_episodes // 4)
        collector.collect_scripted_episodes(3 * args.num_episodes // 4)
    
    # Save demonstrations
    dataset_path = collector.save_demonstrations()
    
    print("\nNext steps:")
    print("1. Convert to LeRobot format if needed")
    print("2. Run GR00T fine-tuning with:")
    print(f"   python train_gr00t_sft.py --dataset-path {dataset_path}")
    print("3. Use fine-tuned model with PPO training")


if __name__ == "__main__":
    main()