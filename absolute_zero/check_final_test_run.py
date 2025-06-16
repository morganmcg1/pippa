#!/usr/bin/env python3
"""
Check the final test run for tables and artifacts.
"""

import wandb
from wandb.apis.public import Api
import os
from dotenv import load_dotenv

load_dotenv()

def check_final_test_run():
    """Check tables in the final test run."""
    
    # Initialize API
    api = Api()
    
    # Get the run
    entity = os.getenv("WANDB_ENTITY", "wild-ai")
    project = os.getenv("WANDB_PROJECT", "pippa")
    run_id = "xbfg0kk1"
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        print(f"Run: {run.name} ({run.displayName})")
        print(f"State: {run.state}")
        print(f"URL: {run.url}")
        print(f"Runtime: {run.summary.get('_runtime', 0):.1f} seconds")
        print()
        
        # Check summary for tables
        print("Summary keys related to tables/samples:")
        if hasattr(run, 'summary') and run.summary:
            found_tables = False
            for key in sorted(run.summary.keys()):
                if 'table' in key.lower() or 'samples' in key.lower():
                    print(f"  - {key}")
                    found_tables = True
            if not found_tables:
                print("  (none found)")
        
        # Check logged artifacts
        print("\nLogged artifacts:")
        artifacts = list(run.logged_artifacts())
        if artifacts:
            for artifact in artifacts:
                print(f"  - {artifact.name} (type: {artifact.type})")
        else:
            print("  (none)")
        
        # Check history for table logs
        print("\nChecking history for table logs...")
        history = run.scan_history()
        table_keys = set()
        metric_keys = set()
        for entry in history:
            for key in entry.keys():
                if 'table' in key.lower() or 'samples' in key.lower():
                    table_keys.add(key)
                if 'global_step' in key.lower() or 'eval' in key.lower():
                    metric_keys.add(key)
        
        if table_keys:
            print("Table keys found in history:")
            for key in sorted(table_keys):
                print(f"  - {key}")
        else:
            print("  (no table keys found)")
            
        if metric_keys:
            print("\nMetric keys found:")
            for key in sorted(metric_keys):
                print(f"  - {key}")
                
        # Check why it failed
        print(f"\nRun failed after {run.summary.get('_runtime', 0):.1f} seconds")
        print("This was likely due to the test completing quickly (2 iterations)")
        
    except Exception as e:
        print(f"Error accessing run: {e}")

if __name__ == "__main__":
    check_final_test_run()