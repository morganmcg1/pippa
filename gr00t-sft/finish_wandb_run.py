#!/usr/bin/env python3
"""
Script to finish a WandB run that didn't complete properly
"""

import wandb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.expanduser("~/pippa/.env"))

# Resume the run
run = wandb.init(
    project="pippa",
    entity="wild-ai",
    id="yipnk526",
    resume="must"
)

print(f"Resumed run: {run.name} (ID: {run.id})")
print(f"Run state: {run.state}")

# Log a final summary message
wandb.log({
    "final_message": "Run was terminated during artifact upload. Manually finishing.",
    "training_completed": True,
    "final_step": 75
})

# Properly finish the run
wandb.finish()
print("WandB run finished successfully!")