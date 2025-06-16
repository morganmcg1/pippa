#!/usr/bin/env python3
"""
Debug table logging with explicit finish and different approaches.
"""

import wandb
import os
from dotenv import load_dotenv
import time
import pandas as pd

load_dotenv()

def test_table_logging_with_finish():
    """Test various table logging approaches with explicit finish."""
    
    # Initialize run
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="debug_table_finish",
        tags=["debug", "tables", "finish-test"],
    )
    
    print(f"Testing at: {run.url}")
    
    # Test 1: Basic table with commit=True
    print("\n1. Basic table with commit=True...")
    table1 = wandb.Table(columns=["test", "value"])
    table1.add_data("basic", 1)
    table1.add_data("test", 2)
    wandb.log({"basic_table": table1}, commit=True)
    print("✓ Logged basic table")
    
    # Test 2: Using run.log instead of wandb.log
    print("\n2. Using run.log()...")
    table2 = wandb.Table(columns=["method", "result"])
    table2.add_data("run_log", "test")
    run.log({"run_log_table": table2}, commit=True)
    print("✓ Logged with run.log")
    
    # Test 3: Log as part of summary
    print("\n3. Adding to summary...")
    table3 = wandb.Table(columns=["summary", "data"])
    table3.add_data("in_summary", "value")
    run.summary["summary_table"] = table3
    print("✓ Added to summary")
    
    # Test 4: Using pandas DataFrame
    print("\n4. Using pandas DataFrame...")
    df = pd.DataFrame({
        "iteration": [1, 2, 3],
        "accuracy": [0.1, 0.2, 0.3],
        "loss": [1.0, 0.8, 0.6]
    })
    pandas_table = wandb.Table(dataframe=df)
    wandb.log({"pandas_table": pandas_table}, commit=True)
    print("✓ Logged pandas table")
    
    # Test 5: Multiple tables in one log
    print("\n5. Multiple tables at once...")
    table5a = wandb.Table(columns=["name", "value"])
    table5a.add_data("table_a", 10)
    table5b = wandb.Table(columns=["name", "value"])
    table5b.add_data("table_b", 20)
    wandb.log({
        "multi_a": table5a,
        "multi_b": table5b,
        "metric": 42
    }, commit=True)
    print("✓ Logged multiple tables")
    
    # Test 6: Log table then update it
    print("\n6. Incremental table updates...")
    incremental_table = wandb.Table(columns=["step", "value"])
    for i in range(3):
        incremental_table.add_data(i, i*10)
        wandb.log({f"incremental_table_step_{i}": incremental_table}, commit=True)
        time.sleep(0.5)
    print("✓ Logged incremental updates")
    
    # Test 7: Create artifact as well (known to work)
    print("\n7. Creating artifact for comparison...")
    artifact = wandb.Artifact("debug_tables", type="dataset")
    comparison_table = wandb.Table(columns=["method", "works"])
    comparison_table.add_data("artifact", "yes")
    comparison_table.add_data("direct_log", "testing")
    artifact.add(comparison_table, "comparison")
    run.log_artifact(artifact)
    print("✓ Created artifact")
    
    # Force a final log with metrics
    print("\n8. Final metrics log...")
    wandb.log({"final_metric": 100}, commit=True)
    
    print("\n" + "="*50)
    print("Finishing run...")
    print("="*50)
    
    # Explicitly finish
    wandb.finish()
    print("\nRun finished! Check for tables at:", f"https://wandb.ai/wild-ai/pippa/runs/{run.id}")
    
    # Wait a moment to ensure upload completes
    time.sleep(2)

if __name__ == "__main__":
    test_table_logging_with_finish()