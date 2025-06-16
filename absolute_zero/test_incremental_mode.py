#!/usr/bin/env python3
"""
Test INCREMENTAL log_mode for WandB tables.
"""

import wandb
import os
from dotenv import load_dotenv
import time

load_dotenv()

def test_incremental_mode():
    """Test table logging with INCREMENTAL mode."""
    
    # Initialize run
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="test_incremental_tables",
        tags=["debug", "tables", "incremental"],
    )
    
    print(f"Testing at: {run.url}")
    
    # Test 1: Create initial incremental table
    print("\n1. Creating initial INCREMENTAL table...")
    # First time: create and log with initial data
    inc_table = wandb.Table(columns=["iteration", "step", "accuracy"])
    inc_table.add_data(1, 0, 0.5)
    run.log({"training_progress": inc_table}, commit=True)
    print("✓ Logged initial table")
    
    # Test 2: Add more data to same table name
    print("\n2. Adding more rows to incremental table...")
    for i in range(2, 5):
        # Create new table instance each time with new data
        inc_table = wandb.Table(columns=["iteration", "step", "accuracy"])
        inc_table.add_data(i, i*10, 0.5 + i*0.1)
        run.log({"training_progress": inc_table}, commit=True)
        time.sleep(0.5)
    print("✓ Added incremental updates")
    
    # Test 3: Comprehensive training table with incremental updates
    print("\n3. Simulating training with incremental tables...")
    for iteration in range(3):
        # Each iteration logs a new set of samples
        samples_table = wandb.Table(columns=[
            "iteration", "global_step", "task_type", "role",
            "prompt", "generation", "reward", "valid"
        ])
        
        # Add multiple rows for this iteration
        for task_type in ["deduction", "abduction", "induction"]:
            samples_table.add_data(
                iteration,
                iteration * 100,
                task_type,
                "proposer",
                f"Generate {task_type}: ",
                f"Sample output {iteration}",
                1.0 if iteration % 2 == 0 else -1.0,
                True
            )
        
        # Log this iteration's samples
        run.log({"training_samples": samples_table}, step=iteration*100, commit=True)
        
        # Also add to summary with unique name
        run.summary[f"training_samples_iter_{iteration}"] = samples_table
        
        print(f"✓ Logged samples for iteration {iteration}")
        time.sleep(0.5)
    
    # Test 4: Final summary table
    print("\n4. Creating final summary table...")
    final_table = wandb.Table(columns=["metric", "value"])
    final_table.add_data("final_accuracy", 0.85)
    final_table.add_data("total_iterations", 100)
    final_table.add_data("best_iteration", 95)
    run.summary["final_metrics"] = final_table
    print("✓ Added final summary table")
    
    print("\n" + "="*50)
    print("Finishing run...")
    print("="*50)
    
    # Finish the run
    run.finish()
    print(f"\nRun finished! Check for incremental tables at: {run.url}")

if __name__ == "__main__":
    test_incremental_mode()