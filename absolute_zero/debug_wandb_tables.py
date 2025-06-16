#!/usr/bin/env python3
"""
Debug why wandb.log() tables don't appear but artifacts do.
"""

import wandb
import os
from dotenv import load_dotenv

load_dotenv()

def test_table_logging_methods():
    """Test different methods of logging tables to understand the issue."""
    
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="debug_table_logging_methods",
        tags=["debug", "tables"],
    )
    
    print(f"Testing table logging at: {run.url}")
    
    # Create a test table
    table = wandb.Table(columns=["iteration", "task", "reward", "valid"])
    table.add_data(1, "deduction", 0.5, True)
    table.add_data(1, "abduction", -0.8, False)
    table.add_data(1, "induction", -1.0, False)
    
    # Test 1: Direct log (what we tried)
    print("\n1. Testing direct wandb.log()...")
    wandb.log({"test_table_direct": table})
    print("✓ Logged with direct method")
    
    # Test 2: Log with explicit commit
    print("\n2. Testing with commit=True...")
    table2 = wandb.Table(columns=["test", "value"])
    table2.add_data("commit_test", 1.0)
    wandb.log({"test_table_commit": table2}, commit=True)
    print("✓ Logged with commit=True")
    
    # Test 3: Log without step
    print("\n3. Testing without step parameter...")
    table3 = wandb.Table(columns=["method", "result"])
    table3.add_data("no_step", "success")
    wandb.log({"test_table_no_step": table3})
    print("✓ Logged without step")
    
    # Test 4: Log at step 0
    print("\n4. Testing at step 0...")
    table4 = wandb.Table(columns=["step_test", "data"])
    table4.add_data("step_0", "testing")
    wandb.log({"test_table_step_0": table4}, step=0)
    print("✓ Logged at step 0")
    
    # Test 5: Use run.log_table() method (alternative API)
    print("\n5. Testing run.log() method...")
    table5 = wandb.Table(columns=["alternative", "api"])
    table5.add_data("run_log", "method")
    run.log({"test_table_run_log": table5})
    print("✓ Logged with run.log()")
    
    # Test 6: Create as artifact (what worked)
    print("\n6. Testing artifact method (known to work)...")
    artifact = wandb.Artifact("test_table_artifact", type="dataset")
    artifact.add(table, "test_table")
    run.log_artifact(artifact)
    print("✓ Logged as artifact")
    
    # Test 7: Use Summary API
    print("\n7. Testing summary API...")
    table7 = wandb.Table(columns=["summary", "test"])
    table7.add_data("summary_api", "value")
    run.summary["test_table_summary"] = table7
    print("✓ Added to summary")
    
    # Test 8: Log table data as custom chart
    print("\n8. Testing custom chart...")
    wandb.log({
        "custom_table_chart": wandb.plot.table(
            columns=["x", "y"],
            data=[[1, 2], [3, 4], [5, 6]]
        )
    })
    print("✓ Logged as custom chart")
    
    # Force sync
    print("\n9. Forcing sync...")
    wandb.log({"sync_marker": 1})
    
    print("\n" + "="*50)
    print("Testing complete!")
    print("Check the Tables section at:", run.url)
    print("Also check the Summary section and Charts")
    print("="*50)
    
    wandb.finish()

if __name__ == "__main__":
    test_table_logging_methods()