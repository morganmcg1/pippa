#!/usr/bin/env python3
"""
Investigate why wandb.log() tables don't appear but artifacts do.
Tests various hypotheses about WandB table logging.
"""

import wandb
import os
from dotenv import load_dotenv
import time
import sys

load_dotenv()

def test_wandb_table_issues():
    """Test different scenarios to understand why tables don't appear."""
    
    # Initialize run with specific settings
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="investigate_table_issues",
        tags=["debug", "tables", "investigation"],
        config={"test": "table_investigation"}
    )
    
    print(f"WandB run URL: {run.url}")
    print(f"WandB version: {wandb.__version__}")
    print(f"Python version: {sys.version}")
    
    # Test 1: Check if tables need to be logged with metrics
    print("\n1. Testing if tables need accompanying metrics...")
    table1 = wandb.Table(columns=["test", "value"])
    table1.add_data("with_metric", 1.0)
    wandb.log({
        "table_with_metric": table1,
        "dummy_metric": 1.0  # Add a scalar metric
    })
    print("✓ Logged table with scalar metric")
    
    # Test 2: Check if step matters
    print("\n2. Testing explicit step values...")
    for step in [0, 1, 5, 10]:
        table = wandb.Table(columns=["step", "data"])
        table.add_data(step, f"step_{step}")
        wandb.log({f"table_at_step_{step}": table}, step=step)
        # Also log a metric at same step
        wandb.log({"step_metric": step * 2}, step=step)
    print("✓ Logged tables at different steps")
    
    # Test 3: Check table size limits
    print("\n3. Testing table sizes...")
    # Small table
    small_table = wandb.Table(columns=["size", "data"])
    for i in range(5):
        small_table.add_data("small", i)
    wandb.log({"small_table": small_table})
    
    # Medium table
    medium_table = wandb.Table(columns=["size", "data"])
    for i in range(100):
        medium_table.add_data("medium", i)
    wandb.log({"medium_table": medium_table})
    print("✓ Logged tables of different sizes")
    
    # Test 4: Check table naming conventions
    print("\n4. Testing naming conventions...")
    table4 = wandb.Table(columns=["name_test", "value"])
    table4.add_data("underscores", 1)
    wandb.log({"table_with_underscores": table4})
    
    table5 = wandb.Table(columns=["name_test", "value"])
    table5.add_data("camelCase", 2)
    wandb.log({"tableWithCamelCase": table5})
    
    table6 = wandb.Table(columns=["name_test", "value"])
    table6.add_data("dash", 3)
    wandb.log({"table-with-dash": table6})
    print("✓ Logged tables with different naming")
    
    # Test 5: Check if tables need unique names per step
    print("\n5. Testing table name uniqueness...")
    for i in range(3):
        table = wandb.Table(columns=["iteration", "unique"])
        table.add_data(i, f"iter_{i}")
        wandb.log({f"unique_table_iter_{i}": table}, step=i*10)
    print("✓ Logged tables with unique names")
    
    # Test 6: Test logging multiple tables at once
    print("\n6. Testing multiple tables in one log...")
    table_a = wandb.Table(columns=["table", "data"])
    table_a.add_data("A", "data_a")
    table_b = wandb.Table(columns=["table", "data"])
    table_b.add_data("B", "data_b")
    wandb.log({
        "multi_table_a": table_a,
        "multi_table_b": table_b,
        "multi_metric": 42
    })
    print("✓ Logged multiple tables together")
    
    # Test 7: Check if we need to define custom charts
    print("\n7. Testing with custom chart definition...")
    table7 = wandb.Table(columns=["x", "y", "label"])
    for i in range(10):
        table7.add_data(i, i**2, f"point_{i}")
    
    # Log table
    wandb.log({"custom_chart_table": table7})
    
    # Also try creating a custom chart from it
    wandb.log({
        "custom_scatter": wandb.plot.scatter(
            table7, "x", "y", title="Custom Scatter Plot"
        )
    })
    print("✓ Logged table with custom chart")
    
    # Test 8: Force a sync
    print("\n8. Testing forced sync...")
    table8 = wandb.Table(columns=["sync", "test"])
    table8.add_data("forced", "sync")
    wandb.log({"sync_test_table": table8})
    run.log_code()  # Force code logging
    print("✓ Forced sync")
    
    # Test 9: Log table in summary
    print("\n9. Testing summary table...")
    summary_table = wandb.Table(columns=["summary", "final"])
    summary_table.add_data("summary", "value")
    run.summary["summary_table"] = summary_table
    run.summary.update()
    print("✓ Added table to summary")
    
    # Test 10: Log artifact (known to work)
    print("\n10. Creating artifact for comparison...")
    artifact = wandb.Artifact("investigation_tables", type="dataset")
    comparison_table = wandb.Table(columns=["method", "works"])
    comparison_table.add_data("artifact", "yes")
    comparison_table.add_data("direct_log", "testing")
    artifact.add(comparison_table, "comparison")
    run.log_artifact(artifact)
    print("✓ Created artifact")
    
    # Give time for everything to sync
    print("\n" + "="*60)
    print("Investigation complete!")
    print(f"Check the run at: {run.url}")
    print("\nThings to check in WandB UI:")
    print("1. Tables section - do any appear?")
    print("2. Charts section - custom scatter plot?")
    print("3. Summary section - summary table?")
    print("4. Artifacts section - comparison table?")
    print("5. Logs/System - any errors?")
    print("="*60)
    
    # Final metrics to ensure run has data
    for i in range(5):
        wandb.log({"final_metric": i * 10}, step=100 + i)
        time.sleep(0.1)
    
    wandb.finish()
    print("\nRun finished successfully!")

if __name__ == "__main__":
    test_wandb_table_issues()