#!/usr/bin/env python3
"""
Run multiple overfitting experiments with different configurations
"""

from train_grpo_overfit import main, OverfitExperimentConfig
import time

# Define experiment configurations
experiments = [
    # Baseline: moderate settings
    {
        "name": "baseline",
        "config": OverfitExperimentConfig(
            exp_name="grpo_overfit_baseline",
            num_samples=100,
            total_epochs=20,
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
        )
    },
    
    # Aggressive: small dataset, high LR, many epochs
    {
        "name": "aggressive",
        "config": OverfitExperimentConfig(
            exp_name="grpo_overfit_aggressive",
            num_samples=50,  # Smaller dataset
            total_epochs=50,  # More epochs
            learning_rate=1e-4,  # Higher LR
            per_device_train_batch_size=16,  # Larger batch for better GPU usage
            gradient_accumulation_steps=1,
        )
    },
    
    # Memory optimized: maximize GPU usage
    {
        "name": "memory_optimized",
        "config": OverfitExperimentConfig(
            exp_name="grpo_overfit_memopt",
            num_samples=100,
            total_epochs=30,
            learning_rate=5e-5,
            per_device_train_batch_size=32,  # Much larger batch
            gradient_accumulation_steps=1,
            max_prompt_length=512,  # Longer sequences
            max_completion_length=256,
        )
    },
    
    # Ultra small: extreme overfitting
    {
        "name": "ultra_small",
        "config": OverfitExperimentConfig(
            exp_name="grpo_overfit_ultra",
            num_samples=20,  # Tiny dataset
            total_epochs=100,  # Many epochs
            learning_rate=2e-4,  # Very high LR
            per_device_train_batch_size=4,  # Small batch due to tiny dataset
            gradient_accumulation_steps=1,
            num_generations=8,  # More generations per prompt
        )
    },
]

def run_all_experiments():
    """Run all defined experiments sequentially"""
    print("=" * 80)
    print("GRPO OVERFITTING EXPERIMENTS")
    print("=" * 80)
    print(f"Total experiments to run: {len(experiments)}")
    print()
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {exp['name']}")
        print(f"{'='*80}")
        
        try:
            start_time = time.time()
            main(exp['config'])
            duration = time.time() - start_time
            
            print(f"\n✓ Experiment '{exp['name']}' completed in {duration:.2f} seconds")
            
        except Exception as e:
            print(f"\n✗ Experiment '{exp['name']}' failed with error: {str(e)}")
            continue
        
        # Short pause between experiments
        if i < len(experiments):
            print("\nPausing for 10 seconds before next experiment...")
            time.sleep(10)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    run_all_experiments()