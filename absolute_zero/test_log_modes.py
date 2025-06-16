#!/usr/bin/env python3
"""
Test different log_mode settings for WandB tables.
"""

import wandb
import os
from dotenv import load_dotenv

load_dotenv()

def test_log_modes():
    """Test different log modes for tables."""
    
    # Initialize run
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="test_table_log_modes",
        tags=["debug", "tables", "log-modes"],
    )
    
    print(f"Testing at: {run.url}")
    
    # Test 1: IMMUTABLE mode (default)
    print("\n1. Testing IMMUTABLE mode (default)...")
    table1 = wandb.Table(columns=["test", "value"])
    table1.add_data("immutable", 1)
    run.log({"immutable_table": table1})
    print("✓ Logged immutable table")
    
    # Test 2: MUTABLE mode - can be updated
    print("\n2. Testing MUTABLE mode...")
    table2 = wandb.Table(columns=["iteration", "score"], allow_mixed_types=False)
    table2.add_data(1, 0.5)
    run.log({"mutable_table": table2})
    
    # Now update it
    table2.add_data(2, 0.7)
    run.log({"mutable_table": table2})  # This should work with MUTABLE
    print("✓ Logged and updated mutable table")
    
    # Test 3: Create table properly for our use case
    print("\n3. Testing proper table creation for training...")
    # Create a fresh table each time we want to log
    for i in range(3):
        # Create NEW table each time
        iteration_table = wandb.Table(columns=["step", "accuracy", "loss"])
        iteration_table.add_data(i*10, 0.5 + i*0.1, 1.0 - i*0.2)
        run.log({f"training_table_iter_{i}": iteration_table}, commit=True)
    print("✓ Logged separate tables for each iteration")
    
    # Test 4: Single comprehensive table
    print("\n4. Testing comprehensive training table...")
    training_table = wandb.Table(columns=[
        "iteration", "global_step", "task_type", "role",
        "prompt", "generation", "reward", "valid"
    ])
    
    # Add multiple rows
    for i in range(5):
        training_table.add_data(
            i,  # iteration
            i * 100,  # global_step
            "deduction",  # task_type
            "proposer",  # role
            f"Calculate: {i} + {i+1} = ",  # prompt
            f"{i + i + 1}",  # generation
            1.0 if i % 2 == 0 else -1.0,  # reward
            i % 2 == 0  # valid
        )
    
    # Log the complete table
    run.log({"comprehensive_training_table": training_table}, commit=True)
    print("✓ Logged comprehensive table")
    
    # Test 5: Add to summary
    print("\n5. Adding final table to summary...")
    final_table = wandb.Table(columns=["metric", "value"])
    final_table.add_data("final_accuracy", 0.85)
    final_table.add_data("total_iterations", 100)
    run.summary["final_metrics_table"] = final_table
    print("✓ Added final table to summary")
    
    print("\n" + "="*50)
    print("Finishing run...")
    print("="*50)
    
    # Finish the run
    run.finish()
    print(f"\nRun finished! Check for tables at: {run.url}")

if __name__ == "__main__":
    test_log_modes()