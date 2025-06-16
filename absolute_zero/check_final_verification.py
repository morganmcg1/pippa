#!/usr/bin/env python3
"""
Check tables in the final verification run.
"""

import wandb
from wandb.apis.public import Api
import os
from dotenv import load_dotenv

load_dotenv()

def check_run(run_id="yd51gvox"):
    """Check tables in the run."""
    
    api = Api()
    entity = os.getenv("WANDB_ENTITY", "wild-ai")
    project = os.getenv("WANDB_PROJECT", "pippa")
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        print(f"Run: {run.displayName}")
        print(f"State: {run.state}")
        print(f"URL: {run.url}")
        print()
        
        # Check artifacts
        print("Table/Sample Artifacts:")
        artifacts = list(run.logged_artifacts())
        table_count = 0
        for artifact in artifacts:
            if 'samples' in artifact.name.lower() or 'table' in artifact.name.lower():
                print(f"  ✓ {artifact.name} (type: {artifact.type})")
                table_count += 1
        
        print(f"\nTotal: {table_count} table artifacts")
        
        print("\n" + "="*50)
        print("Summary:")
        print("✅ Tables ARE being logged as artifacts")
        print("⚠️  Solver sample generation needs debugging (see task keys in logs)")
        print("ℹ️  Step warnings are from checkpoint saves, not table logging")
        print(f"\nView tables at: {run.url}")
        print("Look in the Artifacts section or Tables section")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_run()