#!/usr/bin/env python3
"""
Test if commit=True fixes table visibility in WandB.
"""

import wandb
import os
from dotenv import load_dotenv
import time

load_dotenv()

def test_commit_tables():
    """Test table logging with commit=True."""
    
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="test_commit_true_tables",
        tags=["debug", "tables", "commit-test"],
    )
    
    print(f"Testing at: {run.url}")
    
    # Test 1: Log table without commit (old way)
    print("\n1. Testing without commit...")
    table1 = wandb.Table(columns=["method", "result"])
    table1.add_data("no_commit", "test1")
    wandb.log({"table_no_commit": table1}, step=0)
    
    # Test 2: Log table with commit=False explicitly  
    print("\n2. Testing with commit=False...")
    table2 = wandb.Table(columns=["method", "result"])
    table2.add_data("commit_false", "test2")
    wandb.log({"table_commit_false": table2}, step=1, commit=False)
    
    # Test 3: Log table with commit=True
    print("\n3. Testing with commit=True...")
    table3 = wandb.Table(columns=["method", "result"])
    table3.add_data("commit_true", "test3")
    wandb.log({"table_commit_true": table3}, step=2, commit=True)
    
    # Test 4: Log multiple tables with commit=True
    print("\n4. Testing multiple tables with commit=True...")
    table4a = wandb.Table(columns=["test", "value"])
    table4a.add_data("multi_a", 1)
    table4b = wandb.Table(columns=["test", "value"])
    table4b.add_data("multi_b", 2)
    wandb.log({
        "multi_table_a": table4a,
        "multi_table_b": table4b,
        "dummy_metric": 42
    }, step=3, commit=True)
    
    # Test 5: Sequential commits
    print("\n5. Testing sequential commits...")
    for i in range(5):
        table = wandb.Table(columns=["iteration", "data"])
        table.add_data(i, f"seq_{i}")
        wandb.log({f"sequential_table_{i}": table}, step=10+i, commit=True)
        time.sleep(0.5)  # Small delay between commits
    
    # Test 6: Large table with commit
    print("\n6. Testing large table with commit=True...")
    large_table = wandb.Table(columns=["index", "value", "description"])
    for i in range(100):
        large_table.add_data(i, i**2, f"Row {i} description")
    wandb.log({"large_table_commit": large_table}, step=20, commit=True)
    
    # Also create artifact for comparison
    print("\n7. Creating artifact (known to work)...")
    artifact = wandb.Artifact("commit_test_tables", type="dataset")
    comparison_table = wandb.Table(columns=["test", "status"])
    comparison_table.add_data("artifact", "works")
    comparison_table.add_data("commit_true", "testing")
    artifact.add(comparison_table, "comparison")
    run.log_artifact(artifact)
    
    print("\n" + "="*50)
    print("Test complete!")
    print(f"Check tables at: {run.url}")
    print("Look for:")
    print("- table_no_commit")
    print("- table_commit_false")
    print("- table_commit_true (should appear!)")
    print("- multi_table_a/b")
    print("- sequential_table_0-4")
    print("- large_table_commit")
    print("="*50)
    
    wandb.finish()

if __name__ == "__main__":
    test_commit_tables()