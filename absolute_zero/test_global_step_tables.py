#!/usr/bin/env python3
"""
Test correct way to log global_step with tables.
"""

import wandb
import os
from dotenv import load_dotenv
import time

load_dotenv()

def test_global_step_with_tables():
    """Test logging global_step as a metric alongside tables."""
    
    # Initialize run
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="test_global_step_tables",
        tags=["debug", "tables", "global-step"],
    )
    
    print(f"Testing at: {run.url}")
    
    # Simulate training with proper global_step logging
    global_step = 0
    
    for iteration in range(3):
        print(f"\n--- Iteration {iteration} ---")
        
        # Simulate some training steps
        for _ in range(5):
            # Log regular metrics with global_step
            wandb.log({
                "loss": 1.0 - (global_step * 0.01),
                "global_step": global_step
            })
            global_step += 1
        
        # Log training samples table with global_step
        samples_table = wandb.Table(columns=[
            "iteration", "task_type", "prompt", "generation", "reward"
        ])
        
        # Add sample data
        for task in ["deduction", "abduction", "induction"]:
            samples_table.add_data(
                iteration, task, 
                f"Generate {task}:", 
                f"Output for iteration {iteration}",
                1.0 if global_step % 2 == 0 else -1.0
            )
        
        # Log table with global_step as a metric (NOT as step parameter)
        run.log({
            "samples/training_samples": samples_table,
            "global_step": global_step
        }, commit=True)
        
        print(f"✓ Logged training samples at global_step {global_step}")
        
        # Simulate proposer training
        for _ in range(3):
            wandb.log({
                "proposer_loss": 0.5 - (global_step * 0.005),
                "global_step": global_step
            })
            global_step += 1
        
        # Log proposer samples
        proposer_table = wandb.Table(columns=[
            "iteration", "epoch", "task_type", "generation", "valid"
        ])
        proposer_table.add_data(iteration, 1, "deduction", "5 + 3 = 8", True)
        proposer_table.add_data(iteration, 1, "abduction", "? + ? = 10", True)
        
        run.log({
            "samples/proposer_samples": proposer_table,
            "global_step": global_step
        }, commit=True)
        
        print(f"✓ Logged proposer samples at global_step {global_step}")
        
        # Add to summary as well
        run.summary["samples/training_samples_latest"] = samples_table
        run.summary["samples/proposer_samples_latest"] = proposer_table
        
        time.sleep(0.5)
    
    print(f"\nFinal global_step: {global_step}")
    
    # Log final metrics
    run.log({
        "final_accuracy": 0.85,
        "total_iterations": 3,
        "global_step": global_step
    }, commit=True)
    
    print("\n" + "="*50)
    print("Expected results:")
    print("- Tables should appear with proper global_step tracking")
    print("- No step warnings in console")
    print("- global_step should be visible as a metric")
    print("="*50)
    
    # Finish the run
    run.finish()
    print(f"\nRun finished! Check at: {run.url}")

if __name__ == "__main__":
    test_global_step_with_tables()