#!/usr/bin/env python3
"""Run progressive overfitting experiments from easy to hard tasks."""

import subprocess
import sys
import time

experiments = [
    {
        "name": "1_echo_baseline",
        "task": "echo",
        "batch_size": 32,
        "num_generations": 16,
        "epochs": 30,
        "lr": 5e-4,
        "temperature": 0.3,
        "n_samples": 10
    },
    {
        "name": "2_pattern_completion",
        "task": "pattern",
        "batch_size": 32,
        "num_generations": 16,
        "epochs": 50,
        "lr": 5e-4,
        "temperature": 0.5,
        "n_samples": 10
    },
    {
        "name": "3_simple_math",
        "task": "math",
        "batch_size": 32,
        "num_generations": 16,
        "epochs": 50,
        "lr": 5e-4,
        "temperature": 0.5,
        "n_samples": 15
    },
    {
        "name": "4_word_problems",
        "task": "word_problem",
        "batch_size": 32,
        "num_generations": 16,
        "epochs": 100,
        "lr": 3e-4,
        "temperature": 0.7,
        "n_samples": 20
    },
    {
        "name": "5_echo_max_gpu",
        "task": "echo",
        "batch_size": 64,
        "num_generations": 32,
        "epochs": 20,
        "lr": 1e-3,
        "temperature": 0.3,
        "n_samples": 10
    }
]

def run_experiment(exp):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp['name']}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "train_grpo_progressive_overfit.py",
        "--task", exp["task"],
        "--batch_size", str(exp["batch_size"]),
        "--num_generations", str(exp["num_generations"]),
        "--epochs", str(exp["epochs"]),
        "--lr", str(exp["lr"]),
        "--temperature", str(exp["temperature"]),
        "--n_samples", str(exp["n_samples"])
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: Experiment {exp['name']} failed!")
        return False
    
    print(f"Experiment {exp['name']} completed in {duration:.1f}s")
    return True

def main():
    print("Starting progressive GRPO overfitting experiments")
    print(f"Running {len(experiments)} experiments")
    
    successful = 0
    failed = 0
    
    for exp in experiments:
        if run_experiment(exp):
            successful += 1
        else:
            failed += 1
            print("Stopping due to failure")
            break
        
        # Small pause between experiments
        time.sleep(5)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {successful} successful, {failed} failed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()