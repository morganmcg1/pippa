#!/usr/bin/env python3
"""
Debug video tracking logic for WandB table
"""

from pathlib import Path

# Simulate the video tracking logic
video_episode_tracker = {}
env0_episode_count = 0

# Add some tracking data
for i in range(5):
    video_episode_tracker[env0_episode_count] = {
        "global_step": 1000 * i,
        "episode_data": {"episode_return": 10.0 * i},
        "actual_episode": i
    }
    env0_episode_count += 1

print(f"Video tracker keys: {list(video_episode_tracker.keys())}")

# Test parsing video filenames
test_filenames = [
    "episode-0.mp4",
    "episode-episode-0.mp4",
    "episode-10.mp4",
    "episode-episode-10.mp4"
]

for filename in test_filenames:
    video_file = Path(filename)
    print(f"\nTesting: {filename}")
    
    try:
        # Handle both "episode-0" and "episode-episode-0" formats
        parts = video_file.stem.split('-')
        episode_num = int(parts[-1])
        print(f"  Parts: {parts}")
        print(f"  Extracted episode: {episode_num}")
        
        if episode_num in video_episode_tracker:
            print(f"  ✓ Found in tracker!")
        else:
            print(f"  ✗ NOT in tracker")
    except (ValueError, IndexError) as e:
        print(f"  Error parsing: {e}")

# Test with real video directory structure
video_dir = Path("videos/FetchReach-v3__test__1__123456")
if video_dir.exists():
    print(f"\nActual video files in {video_dir}:")
    for video_file in sorted(video_dir.glob("*.mp4")):
        print(f"  {video_file.name}")