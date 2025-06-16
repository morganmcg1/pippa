#!/usr/bin/env python3
"""
Test WandB table logging to debug why tables aren't appearing.
"""

import wandb
import numpy as np
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_basic_table():
    """Test the most basic table logging."""
    print("Testing basic WandB table logging...")
    
    # Initialize wandb
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="test_table_logging",
        tags=["test", "table-debug"],
    )
    
    print(f"WandB run initialized: {run.url}")
    
    # Test 1: Simple table with immediate logging
    print("\n1. Testing simple table...")
    table = wandb.Table(columns=["step", "value", "description"])
    table.add_data(1, 0.5, "First row")
    table.add_data(2, 0.7, "Second row")
    table.add_data(3, 0.9, "Third row")
    
    # Log without step
    wandb.log({"test_table_no_step": table})
    print("✓ Logged table without step")
    
    # Test 2: Table with step
    print("\n2. Testing table with step...")
    table2 = wandb.Table(columns=["iteration", "accuracy", "loss"])
    table2.add_data(1, 0.85, 0.15)
    table2.add_data(2, 0.90, 0.10)
    
    # Log with step
    wandb.log({"test_table_with_step": table2}, step=10)
    print("✓ Logged table with step=10")
    
    # Test 3: Multiple tables in one log
    print("\n3. Testing multiple tables...")
    table3 = wandb.Table(columns=["task", "reward"])
    table3.add_data("deduction", -0.5)
    table3.add_data("abduction", -0.8)
    table3.add_data("induction", -1.0)
    
    wandb.log({
        "task_rewards": table3,
        "metric": 0.5  # Mix with scalar
    }, step=20)
    print("✓ Logged table with scalar metric")
    
    # Test 4: Incremental table updates
    print("\n4. Testing incremental updates...")
    for i in range(3):
        iter_table = wandb.Table(columns=["iter", "sample", "valid"])
        iter_table.add_data(i, f"sample_{i}", True)
        wandb.log({f"iteration_{i}_samples": iter_table}, step=30 + i)
        print(f"✓ Logged iteration {i} table")
        time.sleep(1)  # Brief pause
    
    # Test 5: Large table
    print("\n5. Testing larger table...")
    large_table = wandb.Table(columns=["idx", "prompt", "completion", "reward"])
    for i in range(20):
        large_table.add_data(
            i,
            f"Calculate: {i} + {i+1} = ?",
            f"{2*i + 1}",
            np.random.random() - 0.5
        )
    wandb.log({"large_sample_table": large_table})
    print("✓ Logged large table with 20 rows")
    
    # Test 6: Table as artifact
    print("\n6. Testing table as artifact...")
    artifact_table = wandb.Table(columns=["test", "result"])
    artifact_table.add_data("artifact_test", "success")
    
    artifact = wandb.Artifact("test_table_artifact", type="dataset")
    artifact.add(artifact_table, "test_data")
    wandb.log_artifact(artifact)
    print("✓ Logged table as artifact")
    
    # Finish
    wandb.finish()
    print("\n✅ All tests complete! Check the run at:", run.url)
    print("Look for tables in the 'Tables' section of the run page.")

if __name__ == "__main__":
    test_basic_table()