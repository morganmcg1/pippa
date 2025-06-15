#!/usr/bin/env python3
"""
Quick test to verify WandB table logging is working.
"""

import wandb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize WandB
wandb.init(
    project=os.getenv("WANDB_PROJECT", "pippa"),
    entity=os.getenv("WANDB_ENTITY", "wild-ai"),
    name="table_logging_test",
    tags=["test", "tables"],
)

print("Testing WandB table logging...")

# Create a simple table
test_table = wandb.Table(columns=["iteration", "metric", "value"])

# Add some data
test_table.add_data(1, "accuracy", 0.75)
test_table.add_data(1, "loss", 0.25)
test_table.add_data(2, "accuracy", 0.85)
test_table.add_data(2, "loss", 0.15)

print(f"Table has {len(test_table.data)} rows")

# Log the table
wandb.log({"test_results": test_table})
print("Logged test_results table")

# Create another table
generation_table = wandb.Table(columns=["prompt", "generation", "valid"])
generation_table.add_data("Calculate: 5 + 3 = ", "8", True)
generation_table.add_data("Calculate: 12 - 7 = ", "5", True)
generation_table.add_data("Calculate: 8 * 4 = ", "32", True)

# Log it
wandb.log({"generation_samples": generation_table})
print("Logged generation_samples table")

# Log some scalar metrics too
wandb.log({
    "test/metric1": 0.5,
    "test/metric2": 0.8,
})

print("\nFinishing run...")
wandb.finish()

print("Done! Check the run at WandB to see if tables appeared.")