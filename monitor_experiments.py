#!/usr/bin/env python3
"""
Simple script to monitor GRPO experiments via WandB API.
"""
import wandb
from datetime import datetime
import time

def monitor_latest_runs(entity="wild-ai", project="pippa", limit=5):
    """Check status of latest GRPO experiments."""
    api = wandb.Api()
    
    # Get latest runs with grpo-setup tag
    runs = api.runs(f"{entity}/{project}", 
                    filters={"tags": {"$in": ["grpo-setup"]}},
                    order="-created_at",
                    limit=limit)
    
    print(f"\n{'='*80}")
    print(f"GRPO Experiment Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*80}\n")
    
    for run in runs:
        # Get run details
        name = run.name
        display_name = run.display_name
        state = run.state
        created = run.created_at
        
        # Get metrics
        summary = run.summary
        train_reward = summary.get('train/reward', 'N/A')
        eval_accuracy = summary.get('eval_accuracy', summary.get('arithmetic_eval', 'N/A'))
        final_accuracy = summary.get('final/eval_accuracy', eval_accuracy)
        epoch = summary.get('train/epoch', 'N/A')
        
        # Print run info
        print(f"Run: {display_name} ({name})")
        print(f"  Status: {state}")
        print(f"  Created: {created}")
        print(f"  Epoch: {epoch}")
        print(f"  Training Reward: {train_reward}")
        print(f"  Eval Accuracy: {final_accuracy}")
        print("-" * 40)
    
    # Special check for early stopping
    early_stop_runs = api.runs(f"{entity}/{project}",
                              filters={"display_name": {"$eq": "early_stopping_25epochs"}},
                              limit=1)
    
    if early_stop_runs:
        print("\n*** EARLY STOPPING EXPERIMENT ***")
        run = early_stop_runs[0]
        print(f"Status: {run.state}")
        print(f"Epoch: {run.summary.get('train/epoch', 'N/A')}")
        print(f"Best Accuracy: {run.summary.get('eval/best_accuracy', 'N/A')}")
        print(f"Current Reward: {run.summary.get('train/reward', 'N/A')}")

if __name__ == "__main__":
    # Monitor once
    monitor_latest_runs()
    
    # Optional: continuous monitoring
    # while True:
    #     monitor_latest_runs()
    #     time.sleep(300)  # Check every 5 minutes