#!/usr/bin/env python3
"""
Check if tables are logged to run v6iq91jk.
"""

import wandb
from wandb.apis.public import Api
import os
from dotenv import load_dotenv

load_dotenv()

def check_tables():
    """Check tables in the current run."""
    
    # Initialize API
    api = Api()
    
    # Get the run
    entity = os.getenv("WANDB_ENTITY", "wild-ai")
    project = os.getenv("WANDB_PROJECT", "pippa")
    run_id = "v6iq91jk"
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        print(f"Run: {run.displayName}")
        print(f"State: {run.state}")
        print(f"URL: {run.url}")
        print()
        
        # Check summary for tables
        print("Summary keys with 'samples' or 'table':")
        if hasattr(run, 'summary') and run.summary:
            found = False
            for key in sorted(run.summary.keys()):
                if 'samples' in key.lower() or 'table' in key.lower():
                    print(f"  ✓ {key}")
                    found = True
            if not found:
                print("  (none found)")
        
        # Check artifacts
        print("\nArtifacts with 'samples' or 'table':")
        artifacts = list(run.logged_artifacts())
        found_artifacts = False
        for artifact in artifacts:
            if 'samples' in artifact.name.lower() or 'table' in artifact.name.lower():
                print(f"  ✓ {artifact.name} (type: {artifact.type})")
                found_artifacts = True
        if not found_artifacts:
            print("  (none found)")
        
        # Check history
        print("\nChecking history for table logs...")
        history = run.scan_history()
        table_keys = set()
        has_global_step = False
        
        for entry in history:
            if 'global_step' in entry:
                has_global_step = True
            for key in entry.keys():
                if 'samples/' in key or 'table' in key.lower():
                    table_keys.add(key)
        
        if table_keys:
            print("Table keys in history:")
            for key in sorted(table_keys):
                print(f"  ✓ {key}")
        else:
            print("  (no table keys found)")
            
        print(f"\nglobal_step logged: {'✓ Yes' if has_global_step else '✗ No'}")
        
        # Summary
        print("\n" + "="*50)
        if table_keys or found_artifacts:
            print("✅ Tables are being logged!")
            print(f"Check them at: {run.url}")
        else:
            print("⚠️  No tables found yet. The run might still be initializing.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_tables()