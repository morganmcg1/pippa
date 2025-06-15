#!/bin/bash
# Full training run for unified Absolute Zero implementation

cd ~/pippa
source az_venv/bin/activate

echo "Launching Full Absolute Zero Unified Training..."
echo "Configuration:"
echo "- 50 iterations for meaningful curriculum emergence"
echo "- Batch size 48 (divisible by 6 for task allocation)"
echo "- Larger seed buffer (64) for more diversity"
echo "- Standard hyperparameters from GRPO success"

# Important batch size consideration:
# With batch_size=48:
# - 48/6 = 8 samples per task type per role
# - Proposer: 8 × 3 task types = 24 prompts
# - Solver: 8 × 3 task types = 24 prompts
# - Total: 48 prompts per iteration
# - With num_generations=8: 48/8 = 6 effective batch size

python absolute_zero/train_absolute_zero_unified.py \
    --model "Qwen/Qwen2-0.5B-Instruct" \
    --iterations 50 \
    --batch-size 48 \
    --seed-buffer-size 64 \
    --learning-rate 5e-6 \
    --temperature 0.7 \
    --beta 0.1 \
    --name-suffix "full_50iter_batch48"