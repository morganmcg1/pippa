#!/usr/bin/env python3
"""
Minimal test for WandB video table logging
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'

from pathlib import Path
import wandb
from dotenv import load_dotenv

load_dotenv()

# Initialize WandB
run = wandb.init(
    project="pippa",
    entity="wild-ai",
    name="test_video_table_minimal",
    tags=["gr00t-rl", "test", "video-table"]
)

# Create a video table with INCREMENTAL mode
video_table = wandb.Table(
    columns=["global_step", "episode", "video", "episode_return"],
    log_mode="INCREMENTAL"
)

# Find a test video file
video_dirs = sorted(Path("videos").glob("*/"))
test_video = None

for video_dir in video_dirs[-5:]:  # Check last 5 directories
    mp4_files = list(video_dir.glob("*.mp4"))
    if mp4_files:
        test_video = mp4_files[0]
        break

if test_video:
    print(f"Found test video: {test_video}")
    
    # Add a row to the table
    video_obj = wandb.Video(str(test_video), fps=30, format="mp4")
    video_table.add_data(
        1000,  # global_step
        0,     # episode
        video_obj,
        -1.0   # episode_return
    )
    
    # Log the table
    wandb.log({"test_video_table": video_table}, step=1000)
    print("Logged video table with 1 row")
    
    # Add another row
    video_table.add_data(
        2000,  # global_step
        1,     # episode
        video_obj,  # Reuse same video for test
        -0.5   # episode_return
    )
    
    # Log the updated table
    wandb.log({"test_video_table": video_table}, step=2000)
    print("Logged video table with 2 rows")
    
else:
    print("No test videos found!")
    
    # Create a dummy video for testing
    import numpy as np
    import cv2
    
    dummy_frames = []
    for i in range(30):  # 1 second at 30fps
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        dummy_frames.append(frame)
    
    # Write dummy video
    Path("videos/test").mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter("videos/test/dummy.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
    for frame in dummy_frames:
        out.write(frame)
    out.release()
    
    # Log dummy video
    video_obj = wandb.Video("videos/test/dummy.mp4", fps=30, format="mp4")
    video_table.add_data(0, 0, video_obj, 0.0)
    wandb.log({"test_video_table": video_table}, step=0)
    print("Created and logged dummy video")

# Log final table as IMMUTABLE
final_table = wandb.Table(
    columns=video_table.columns,
    data=video_table.data,
    log_mode="IMMUTABLE"
)
wandb.log({"final_test_video_table": final_table})
print(f"Logged final table with {len(video_table.data)} rows")

wandb.finish()