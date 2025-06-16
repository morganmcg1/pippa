#!/usr/bin/env python3
"""
Test consistent table names for INCREMENTAL mode.
"""

import wandb
import os
from dotenv import load_dotenv
import time

load_dotenv()

def test_consistent_table_names():
    """Test table logging with consistent names for INCREMENTAL mode."""
    
    # Initialize run
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="test_consistent_table_names",
        tags=["debug", "tables", "incremental", "consistent-names"],
    )
    
    print(f"Testing at: {run.url}")
    
    # Simulate multiple training iterations
    print("\nSimulating training with consistent table names...")
    
    for iteration in range(3):
        print(f"\n--- Iteration {iteration} ---")
        
        # 1. Log training samples (consistent name: samples/training_samples)
        training_table = wandb.Table(columns=[
            "iteration", "global_step", "task_type", "role",
            "prompt", "generation", "reward", "valid"
        ])
        
        # Add some sample data
        for task in ["deduction", "abduction", "induction"]:
            training_table.add_data(
                iteration, iteration * 100, task, "proposer",
                f"Generate {task}:", f"Output {iteration}", 
                1.0 if iteration % 2 == 0 else -1.0, True
            )
        
        # Log with consistent name for INCREMENTAL mode
        run.log({"samples/training_samples": training_table}, step=iteration*100, commit=True)
        run.summary["samples/training_samples_latest"] = training_table
        print(f"✓ Logged training samples")
        
        # 2. Log proposer samples (consistent name: samples/proposer_samples)
        proposer_table = wandb.Table(columns=[
            "iteration", "epoch", "global_step", "task_type", "role",
            "prompt", "generation", "result", "is_valid_proposal"
        ])
        
        for i in range(2):
            proposer_table.add_data(
                iteration, 1, iteration * 100 + 10, "deduction", "proposer",
                "Calculate: 5 + 3 = ", "8", "correct", True
            )
        
        run.log({"samples/proposer_samples": proposer_table}, step=iteration*100+10, commit=True)
        run.summary["samples/proposer_samples_latest"] = proposer_table
        print(f"✓ Logged proposer samples")
        
        # 3. Log solver samples (consistent name: samples/solver_samples)
        solver_table = wandb.Table(columns=[
            "iteration", "epoch", "global_step", "task_type", "role",
            "prompt", "generation", "result", "is_valid_proposal"
        ])
        
        for i in range(2):
            solver_table.add_data(
                iteration, 1, iteration * 100 + 50, "deduction", "solver",
                "Calculate: 7 + 2 = ", "9", "correct", True
            )
        
        run.log({"samples/solver_samples": solver_table}, step=iteration*100+50, commit=True)
        run.summary["samples/solver_samples_latest"] = solver_table
        print(f"✓ Logged solver samples")
        
        time.sleep(0.5)
    
    # Log some final metrics
    print("\nLogging final metrics...")
    run.log({"final_accuracy": 0.85}, commit=True)
    
    print("\n" + "="*50)
    print("Expected results:")
    print("- samples/training_samples: Should show all iterations")
    print("- samples/proposer_samples: Should show all iterations") 
    print("- samples/solver_samples: Should show all iterations")
    print("- All tables should be in INCREMENTAL mode")
    print("="*50)
    
    # Finish the run
    run.finish()
    print(f"\nRun finished! Check tables at: {run.url}")

if __name__ == "__main__":
    test_consistent_table_names()