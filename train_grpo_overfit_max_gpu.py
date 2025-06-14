#!/usr/bin/env python3
"""
Maximum GPU utilization experiment for GRPO overfitting
"""

from train_grpo_overfit import main, OverfitExperimentConfig

if __name__ == "__main__":
    # Configuration to maximize GPU usage on H100 (80GB)
    config = OverfitExperimentConfig(
        exp_name="grpo_overfit_max_gpu",
        num_samples=100,
        total_epochs=30,
        learning_rate=1e-4,  # Higher LR for faster convergence
        
        # Maximize batch size for H100
        # Current usage: ~15GB with batch_size=8
        # Target: ~60GB usage (75% of 80GB)
        # Scaling factor: 60/15 = 4x
        per_device_train_batch_size=64,  # 8x larger
        gradient_accumulation_steps=1,
        
        # Longer sequences for more memory usage
        max_prompt_length=512,
        max_completion_length=256,
        
        # More generations for better exploration
        num_generations=8,
        
        # No regularization for faster overfitting
        weight_decay=0.0,
        warmup_ratio=0.01,  # Minimal warmup
    )
    
    print("=" * 80)
    print("MAXIMUM GPU UTILIZATION EXPERIMENT")
    print("=" * 80)
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Sequence lengths: prompt={config.max_prompt_length}, completion={config.max_completion_length}")
    print(f"Generations per prompt: {config.num_generations}")
    print(f"Expected GPU memory usage: ~60-70GB (75-87% of H100)")
    print("=" * 80)
    
    main(config)