#!/usr/bin/env python3
"""
Test table logging using wandb.run.log() as per documentation.
"""

import wandb
import os
from dotenv import load_dotenv

load_dotenv()

def test_run_log_tables():
    """Test table logging with run.log() method."""
    
    # Initialize run
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="test_run_log_method",
        tags=["debug", "tables", "run-log"],
    )
    
    print(f"Testing at: {run.url}")
    
    # Test 1: Simple table with run.log()
    print("\n1. Testing run.log() with simple table...")
    table1 = wandb.Table(columns=["test", "value"])
    table1.add_data("method", "run.log")
    table1.add_data("status", "testing")
    run.log({"simple_table": table1})
    print("✓ Logged simple table with run.log()")
    
    # Test 2: With commit=True
    print("\n2. Testing run.log() with commit=True...")
    table2 = wandb.Table(columns=["iteration", "accuracy"])
    table2.add_data(1, 0.5)
    table2.add_data(2, 0.7)
    run.log({"accuracy_table": table2}, commit=True)
    print("✓ Logged with commit=True")
    
    # Test 3: With step parameter
    print("\n3. Testing with step parameter...")
    table3 = wandb.Table(columns=["step", "loss"])
    table3.add_data(10, 1.5)
    table3.add_data(20, 1.2)
    run.log({"loss_table": table3}, step=100, commit=True)
    print("✓ Logged with step=100")
    
    # Test 4: Incremental mode (as per docs)
    print("\n4. Testing incremental table...")
    # First create and log initial table
    inc_table = wandb.Table(columns=["epoch", "metric"])
    inc_table.add_data(1, 0.1)
    run.log({"incremental_table": inc_table})
    
    # Then add more data
    inc_table.add_data(2, 0.2)
    run.log({"incremental_table": inc_table})
    
    inc_table.add_data(3, 0.3)
    run.log({"incremental_table": inc_table})
    print("✓ Logged incremental table updates")
    
    # Test 5: Also try adding to summary
    print("\n5. Adding table to summary...")
    summary_table = wandb.Table(columns=["final", "results"])
    summary_table.add_data("accuracy", 0.95)
    summary_table.add_data("loss", 0.05)
    run.summary["final_results_table"] = summary_table
    print("✓ Added table to summary")
    
    # Log some regular metrics too
    print("\n6. Logging regular metrics...")
    run.log({"test_metric": 42}, commit=True)
    
    print("\n" + "="*50)
    print("Finishing run...")
    print("="*50)
    
    # Finish the run
    run.finish()
    print(f"\nRun finished! Check for tables at: {run.url}")

if __name__ == "__main__":
    test_run_log_tables()