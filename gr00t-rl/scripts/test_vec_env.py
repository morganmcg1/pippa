#!/usr/bin/env python3
"""Test vectorized environment"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from environments.vec_isaac_env import DummyVecEnv
from scripts.train_ppo_fetch_with_video_table import make_fetch_env
import numpy as np

# Create env functions
env_fns = [make_fetch_env("FetchReach-v3", i, False, "test") for i in range(2)]

# Create vectorized env
envs = DummyVecEnv(env_fns)

# Reset
obs, info = envs.reset()

print(f"Obs type: {type(obs)}")
print(f"Obs shape: {obs.shape if hasattr(obs, 'shape') else 'no shape'}")
print(f"Is numpy array: {isinstance(obs, np.ndarray)}")

if isinstance(obs, np.ndarray):
    print(f"Obs dtype: {obs.dtype}")
    print(f"First env obs shape: {obs[0].shape}")
    print(f"Sample values: {obs[0][:5]}")
else:
    print(f"Obs content: {obs}")