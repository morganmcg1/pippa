#!/usr/bin/env python3
"""
Quick script to add sample logging to the current Absolute Zero run.
This modifies the existing training script to log samples during training.
"""

import subprocess
import os

# Create a patch for the current training script
patch_content = '''
# Add this import at the top after other imports
from transformers import TrainerCallback

# Add this class before TRRPlusBaselines
class QuickSampleLogger(TrainerCallback):
    """Quick callback to log samples to WandB during training."""
    
    def __init__(self, role, iteration, tokenizer):
        self.role = role
        self.iteration = iteration
        self.tokenizer = tokenizer
        self.logged_steps = set()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step in self.logged_steps:
            return
            
        # Log every 20 steps
        if state.global_step % 20 == 0:
            self.logged_steps.add(state.global_step)
            
            # Log current metrics with role prefix
            import wandb
            log_dict = {
                f"{self.role}/step": state.global_step,
                f"{self.role}/current_iteration": self.iteration,
            }
            
            # Add any relevant metrics from logs
            if 'train/reward' in logs:
                log_dict[f"{self.role}/current_reward"] = logs['train/reward']
            if 'train/loss' in logs:
                log_dict[f"{self.role}/current_loss"] = logs['train/loss']
                
            wandb.log(log_dict)

# Then modify the trainer creation to add callbacks:
# For solver (around line 653):
solver_trainer = GRPOTrainer(
    model=solver_model_path,
    args=solver_config,
    train_dataset=solver_dataset,
    reward_funcs=[solver_reward_function],
    callbacks=[QuickSampleLogger('solver', iteration + 1, trainer.tokenizer)]  # ADD THIS
)

# For proposer (around line 747):
proposer_trainer = GRPOTrainer(
    model=proposer_model_path,
    args=proposer_config,
    train_dataset=proposer_dataset,
    reward_funcs=[proposer_reward_function],
    callbacks=[QuickSampleLogger('proposer', iteration + 1, trainer.tokenizer)]  # ADD THIS
)
'''

print("Patch content to add to train_absolute_zero_baseline.py:")
print("="*60)
print(patch_content)
print("="*60)

# Also create a monitoring script
monitor_script = '''#!/bin/bash
# Monitor the current Absolute Zero run

echo "Monitoring Absolute Zero training..."
echo "=================================="

# SSH to the server and check the tmux session
ssh ubuntu@192.222.52.59 << 'EOF'
echo "Current training status:"
tmux capture-pane -t absolute_zero_improved -p | tail -50 | grep -E "(ITERATION|accuracy|reward|PROPOSER|SOLVER|Iteration.*complete)"

echo ""
echo "GPU usage:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "Process info:"
ps aux | grep -E "(train_absolute_zero|python)" | grep -v grep | head -5
EOF
'''

with open('/Users/morganmcguire/ML/pippa/absolute_zero/monitor_training.sh', 'w') as f:
    f.write(monitor_script)

os.chmod('/Users/morganmcguire/ML/pippa/absolute_zero/monitor_training.sh', 0o755)

print("\nCreated monitor_training.sh script")
print("\nTo monitor the current run:")
print("  ./absolute_zero/monitor_training.sh")
print("\nTo restart with enhanced logging:")
print("  1. Stop current run: ssh ubuntu@192.222.52.59 'tmux send-keys -t absolute_zero_improved C-c'")
print("  2. Launch new version: ssh ubuntu@192.222.52.59 'cd ~/pippa && python absolute_zero/train_absolute_zero_with_callbacks.py'")