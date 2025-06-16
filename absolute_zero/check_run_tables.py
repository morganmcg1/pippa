#!/usr/bin/env python3
"""
Check if tables were logged to a specific WandB run.
"""

import wandb
from wandb.apis.public import Api
import os
from dotenv import load_dotenv

load_dotenv()

def check_run_tables(run_id="776mk9um"):
    """Check tables in a specific run."""
    
    # Initialize API
    api = Api()
    
    # Get the run
    entity = os.getenv("WANDB_ENTITY", "wild-ai")
    project = os.getenv("WANDB_PROJECT", "pippa")
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        print(f"Run: {run.name} ({run.displayName})")
        print(f"State: {run.state}")
        print(f"URL: {run.url}")
        print()
        
        # Check summary for tables
        print("Summary keys:")
        if hasattr(run, 'summary') and run.summary:
            for key in sorted(run.summary.keys()):
                if 'table' in key.lower() or 'samples' in key.lower():
                    print(f"  - {key}")
        
        # Check logged artifacts
        print("\nLogged artifacts:")
        artifacts = run.logged_artifacts()
        for artifact in artifacts:
            if 'table' in artifact.name.lower() or 'samples' in artifact.name.lower():
                print(f"  - {artifact.name} (type: {artifact.type})")
        
        # Check history for table logs
        print("\nChecking history for table logs...")
        history = run.scan_history()
        table_keys = set()
        for entry in history:
            for key in entry.keys():
                if 'table' in key.lower() or 'samples' in key.lower():
                    table_keys.add(key)
        
        if table_keys:
            print("Table keys found in history:")
            for key in sorted(table_keys):
                print(f"  - {key}")
        else:
            print("No table keys found in history")
            
    except Exception as e:
        print(f"Error accessing run: {e}")

if __name__ == "__main__":
    check_run_tables()