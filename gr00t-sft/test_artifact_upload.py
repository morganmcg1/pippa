#!/usr/bin/env python3
"""
Minimal test to verify WandB artifact upload is working
"""

import wandb
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.expanduser("~/pippa/.env"))

# Initialize WandB
run = wandb.init(
    project="pippa",
    entity="wild-ai",
    name="artifact_upload_test",
    tags=["test", "artifact-verification"]
)

print(f"WandB run initialized: {run.name}")

try:
    # Create a small test file
    test_dir = Path("./test_artifact_dir")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "test.txt"
    with open(test_file, 'w') as f:
        f.write("This is a test artifact file\n")
    
    # Create artifact
    artifact = wandb.Artifact(
        name="test-artifact",
        type="test",
        description="Testing artifact upload"
    )
    
    print("Adding file to artifact...")
    artifact.add_file(str(test_file))
    
    print("Logging artifact...")
    wandb.run.log_artifact(artifact, aliases=["test", "latest"])
    print("✓ Artifact logged successfully!")
    
    # Clean up
    import shutil
    shutil.rmtree(test_dir)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Ensure wandb finishes
    print("\nFinishing WandB run...")
    wandb.finish()
    print("✓ WandB run finished!")