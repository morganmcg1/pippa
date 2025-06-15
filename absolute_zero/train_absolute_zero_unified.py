#!/usr/bin/env python3
"""
Absolute Zero unified implementation for arithmetic tasks.
Based on https://arxiv.org/pdf/2505.03335v2

Key features:
1. Single unified model serves as both proposer and solver
2. Seeding phase to populate initial task buffers (no gradients)
3. Three task types: Deduction, Abduction, Induction
4. Joint training with single RL update per iteration
"""

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, TrainerState, TrainerControl
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
from dotenv import load_dotenv
import os
import wandb
import re
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import argparse
import json

# Load environment variables
load_dotenv()

# Enable WandB model checkpointing
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TaskBuffer:
    """Buffer for storing validated tasks of each type."""
    def __init__(self, maxlen: int = 1000):
        self.buffer = deque(maxlen=maxlen)
    
    def add(self, task: Dict[str, Any]):
        self.buffer.append(task)
    
    def sample(self, n: int) -> List[Dict[str, Any]]:
        """Sample n tasks from buffer, with replacement if needed."""
        if len(self.buffer) == 0:
            return []
        if n <= len(self.buffer):
            return random.sample(list(self.buffer), n)
        # Sample with replacement if we need more than available
        return random.choices(list(self.buffer), k=n)
    
    def __len__(self):
        return len(self.buffer)


class TRRPlusBaselines:
    """Task-Relative REINFORCE++ with separate baselines for each task type and role."""
    def __init__(self):
        # 6 baselines: 3 task types × 2 roles
        self.baselines = {
            'proposer_deduction': deque(maxlen=100),
            'proposer_abduction': deque(maxlen=100),
            'proposer_induction': deque(maxlen=100),
            'solver_deduction': deque(maxlen=100),
            'solver_abduction': deque(maxlen=100),
            'solver_induction': deque(maxlen=100),
        }
    
    def update(self, role: str, task_type: str, reward: float):
        """Update baseline for specific role and task type."""
        key = f"{role}_{task_type}"
        if key in self.baselines:
            self.baselines[key].append(reward)
    
    def get_baseline(self, role: str, task_type: str) -> float:
        """Get baseline value for specific role and task type."""
        key = f"{role}_{task_type}"
        if key in self.baselines and len(self.baselines[key]) > 0:
            return np.mean(self.baselines[key])
        return 0.0
    
    def compute_advantage(self, role: str, task_type: str, reward: float) -> float:
        """Compute advantage using task-relative baseline."""
        baseline = self.get_baseline(role, task_type)
        return reward - baseline


class UnifiedAbsoluteZeroTrainer:
    """Unified trainer using single model for both proposer and solver roles."""
    
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model_name = model_name
        
        # Single unified model
        print("Loading unified model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Task buffers for three types
        self.deduction_buffer = TaskBuffer(maxlen=1000)
        self.abduction_buffer = TaskBuffer(maxlen=1000)
        self.induction_buffer = TaskBuffer(maxlen=1000)
        
        # TRR++ baselines
        self.baselines = TRRPlusBaselines()
        
        # Track solver performance for learnability rewards
        self.solver_performance = {
            'deduction': deque(maxlen=100),
            'abduction': deque(maxlen=100),
            'induction': deque(maxlen=100),
        }
    
    def create_deduction_task(self, a: int, op: str, b: int) -> Dict[str, Any]:
        """Create a deduction task: predict output given expression."""
        expression = f"{a} {op} {b}"
        
        # Compute answer
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        elif op == '*':
            answer = a * b
        else:
            return None
        
        return {
            'type': 'deduction',
            'expression': expression,
            'answer': str(answer),
            'a': a,
            'b': b,
            'op': op
        }
    
    def create_abduction_task(self, op: str, result: int) -> Optional[Dict[str, Any]]:
        """Create an abduction task: find inputs given operation and result."""
        # Find valid inputs for the given result
        valid_pairs = []
        
        if op == '+':
            # For addition: a + b = result
            for a in range(0, min(result + 1, 21)):
                b = result - a
                if 0 <= b <= 20:
                    valid_pairs.append((a, b))
        elif op == '-':
            # For subtraction: a - b = result
            for a in range(max(0, result), min(result + 21, 41)):
                b = a - result
                if 0 <= b <= 20:
                    valid_pairs.append((a, b))
        elif op == '*':
            # For multiplication: a * b = result
            if result == 0:
                valid_pairs = [(0, random.randint(1, 10)), (random.randint(1, 10), 0)]
            else:
                for a in range(1, min(result + 1, 21)):
                    if result % a == 0:
                        b = result // a
                        if 0 < b <= 20:
                            valid_pairs.append((a, b))
        
        if not valid_pairs:
            return None
        
        # Pick one valid pair as the canonical answer
        a, b = random.choice(valid_pairs)
        
        return {
            'type': 'abduction',
            'operation': op,
            'result': result,
            'answer_a': a,
            'answer_b': b,
            'valid_pairs': valid_pairs
        }
    
    def create_induction_task(self, rule: str, num_examples: int = 5) -> Dict[str, Any]:
        """Create an induction task: infer pattern from examples."""
        examples = []
        
        for _ in range(num_examples):
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            
            if rule == 'addition':
                output = a + b
                rule_desc = "add the two numbers"
            elif rule == 'subtraction':
                output = a - b
                rule_desc = "subtract the second from the first"
            elif rule == 'multiplication':
                output = a * b
                rule_desc = "multiply the two numbers"
            elif rule == 'sum_plus_one':
                output = a + b + 1
                rule_desc = "add the two numbers and add 1"
            elif rule == 'difference_squared':
                output = (a - b) ** 2
                rule_desc = "square the difference"
            else:
                return None
            
            examples.append({'input': (a, b), 'output': output})
        
        # Split into visible and hidden examples
        visible_examples = examples[:num_examples // 2]
        hidden_examples = examples[num_examples // 2:]
        
        return {
            'type': 'induction',
            'rule': rule,
            'rule_description': rule_desc,
            'visible_examples': visible_examples,
            'hidden_examples': hidden_examples,
            'all_examples': examples
        }
    
    def seed_buffers(self, batch_size: int = 8, buffer_size: int = 32):
        """Populate initial task buffers without taking gradients (seeding phase)."""
        print("\n" + "="*60)
        print("SEEDING PHASE - Populating initial task buffers")
        print("="*60)
        
        # Seed deduction buffer
        print(f"\nSeeding deduction buffer (target: {buffer_size} tasks)...")
        while len(self.deduction_buffer) < buffer_size:
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            op = random.choice(['+', '-', '*'])
            task = self.create_deduction_task(a, op, b)
            if task:
                self.deduction_buffer.add(task)
        print(f"✓ Deduction buffer seeded with {len(self.deduction_buffer)} tasks")
        
        # Seed abduction buffer
        print(f"\nSeeding abduction buffer (target: {buffer_size} tasks)...")
        while len(self.abduction_buffer) < buffer_size:
            result = random.randint(2, 20)
            op = random.choice(['+', '-', '*'])
            task = self.create_abduction_task(op, result)
            if task:
                self.abduction_buffer.add(task)
        print(f"✓ Abduction buffer seeded with {len(self.abduction_buffer)} tasks")
        
        # Seed induction buffer
        print(f"\nSeeding induction buffer (target: {buffer_size} tasks)...")
        rules = ['addition', 'subtraction', 'multiplication', 'sum_plus_one', 'difference_squared']
        while len(self.induction_buffer) < buffer_size:
            rule = random.choice(rules)
            task = self.create_induction_task(rule, num_examples=6)
            if task:
                self.induction_buffer.add(task)
        print(f"✓ Induction buffer seeded with {len(self.induction_buffer)} tasks")
        
        print("\nSeeding phase complete!")
    
    def create_proposer_prompt(self, task_type: str, buffer_samples: List[Dict]) -> str:
        """Create few-shot prompt for proposer to generate new tasks."""
        if task_type == 'deduction':
            prompt = "Generate arithmetic problems:\n"
            for sample in buffer_samples[-4:]:  # 4-shot
                prompt += f"Calculate: {sample['expression']} = \n"
            prompt += "Calculate: "
            
        elif task_type == 'abduction':
            prompt = "Find inputs that produce the result:\n"
            for sample in buffer_samples[-4:]:
                prompt += f"Find: ? {sample['operation']} ? = {sample['result']}\n"
            prompt += "Find: ? "
            
        elif task_type == 'induction':
            prompt = "Create pattern examples:\n"
            for sample in buffer_samples[-2:]:  # 2-shot due to length
                examples = sample['visible_examples'][:2]
                prompt += f"Pattern: ({examples[0]['input'][0]},{examples[0]['input'][1]})→{examples[0]['output']}, "
                prompt += f"({examples[1]['input'][0]},{examples[1]['input'][1]})→{examples[1]['output']}\n"
            prompt += "Pattern: "
        
        return prompt
    
    def create_solver_prompt(self, task: Dict[str, Any], task_type: str) -> str:
        """Create prompt for solver to attempt the task."""
        if task_type == 'deduction':
            return f"Calculate: {task['expression']} = "
            
        elif task_type == 'abduction':
            return f"Find two numbers that when {task['operation']} equals {task['result']}. Answer: "
            
        elif task_type == 'induction':
            prompt = "Given these examples, find the pattern:\n"
            for ex in task['visible_examples']:
                prompt += f"({ex['input'][0]},{ex['input'][1]}) → {ex['output']}\n"
            # Ask to predict one of the hidden examples
            hidden = task['hidden_examples'][0]
            prompt += f"What is ({hidden['input'][0]},{hidden['input'][1]}) → "
            
        return prompt
    
    def parse_proposer_generation(self, generation: str, task_type: str) -> Optional[Dict[str, Any]]:
        """Parse proposer's generation into a task."""
        if task_type == 'deduction':
            # Parse "Calculate: a op b = "
            match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)', generation)
            if match:
                a, op, b = match.groups()
                return self.create_deduction_task(int(a), op, int(b))
                
        elif task_type == 'abduction':
            # Parse "Find: ? op ? = result"
            match = re.search(r'\?\s*([+\-*])\s*\?\s*=\s*(\d+)', generation)
            if match:
                op, result = match.groups()
                return self.create_abduction_task(op, int(result))
                
        elif task_type == 'induction':
            # Parse pattern examples
            matches = re.findall(r'\((\d+),(\d+)\)→(\d+)', generation)
            if len(matches) >= 3:
                # Infer the rule from examples
                examples = [{'input': (int(a), int(b)), 'output': int(o)} for a, b, o in matches]
                
                # Try to detect the rule
                rule = self.infer_rule_from_examples(examples)
                if rule:
                    return self.create_induction_task(rule, num_examples=6)
        
        return None
    
    def infer_rule_from_examples(self, examples: List[Dict]) -> Optional[str]:
        """Try to infer the arithmetic rule from examples."""
        # Check common rules
        rules = {
            'addition': lambda a, b: a + b,
            'subtraction': lambda a, b: a - b,
            'multiplication': lambda a, b: a * b,
            'sum_plus_one': lambda a, b: a + b + 1,
            'difference_squared': lambda a, b: (a - b) ** 2
        }
        
        for rule_name, rule_func in rules.items():
            if all(rule_func(ex['input'][0], ex['input'][1]) == ex['output'] for ex in examples):
                return rule_name
        
        return None
    
    def evaluate_solver_response(self, response: str, task: Dict[str, Any], task_type: str) -> float:
        """Evaluate solver's response and return reward."""
        if task_type == 'deduction':
            # Extract number from response
            match = re.search(r'-?\d+', response)
            if match and match.group() == task['answer']:
                return 1.0
            return -1.0
            
        elif task_type == 'abduction':
            # Check if response contains valid input pair
            numbers = re.findall(r'\d+', response)
            if len(numbers) >= 2:
                a, b = int(numbers[0]), int(numbers[1])
                # Check if this is a valid pair
                if (a, b) in task['valid_pairs'] or (b, a) in task['valid_pairs']:
                    return 1.0
            return -1.0
            
        elif task_type == 'induction':
            # Check if predicted output is correct
            match = re.search(r'-?\d+', response)
            if match:
                predicted = int(match.group())
                expected = task['hidden_examples'][0]['output']
                if predicted == expected:
                    return 1.0
            return -1.0
        
        return -1.0


def create_epoch_sample_logger(role: str, iteration: int, tokenizer, trainer: UnifiedAbsoluteZeroTrainer):
    """Create a callback for logging samples at epoch end."""
    
    class EpochSampleLogger(TrainerCallback):
        """Callback to log sample generations at the end of each epoch."""
        
        def __init__(self):
            self.role = role
            self.iteration = iteration
            self.tokenizer = tokenizer
            self.trainer = trainer
            self.epoch_count = 0
            
        def on_epoch_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs):
            """Log samples at the end of each epoch."""
            self.epoch_count += 1
            model = kwargs.get('model')
            if model is None:
                return
            
            # Create table with consistent name
            table_name = f"{self.role}_samples"
            table = wandb.Table(columns=[
                "iteration", "epoch", "global_step", "task_type", 
                "prompt", "generation", "parsed_task", "is_valid"
            ])
            
            model.eval()
            with torch.no_grad():
                # Log samples for each task type
                for task_type in ['deduction', 'abduction', 'induction']:
                    # Get buffer samples for few-shot prompt
                    if task_type == 'deduction':
                        buffer_samples = self.trainer.deduction_buffer.sample(4)
                    elif task_type == 'abduction':
                        buffer_samples = self.trainer.abduction_buffer.sample(4)
                    else:
                        buffer_samples = self.trainer.induction_buffer.sample(2)
                    
                    if not buffer_samples:
                        continue
                    
                    # Generate sample
                    prompt = self.trainer.create_proposer_prompt(task_type, buffer_samples)
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        temperature=1.0,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    completion = generated[len(prompt):].strip()
                    
                    # Parse and validate
                    parsed_task = self.trainer.parse_proposer_generation(completion, task_type)
                    is_valid = parsed_task is not None
                    
                    table.add_data(
                        self.iteration,
                        self.epoch_count,
                        state.global_step,
                        task_type,
                        prompt[-100:],  # Last 100 chars
                        completion[:100],  # First 100 chars
                        str(parsed_task)[:100] if parsed_task else "PARSE_FAILED",
                        is_valid
                    )
            
            # Log the table
            wandb.log({table_name: table})
            model.train()
    
    return EpochSampleLogger()


def main():
    parser = argparse.ArgumentParser(description="Absolute Zero unified training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct", help="Model name")
    parser.add_argument("--iterations", type=int, default=20, help="Total iterations")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seed-buffer-size", type=int, default=32, help="Initial buffer size")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode")
    parser.add_argument("--name-suffix", type=str, default="", help="Suffix for run name")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize WandB
    run_name = "absolute_zero_unified"
    if args.quick_test:
        run_name = "absolute_zero_unified_quick_test"
    if args.name_suffix:
        run_name += f"_{args.name_suffix}"
    
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=run_name,
        tags=["absolute-zero", "arithmetic", "unified-model", "three-tasks"],
        config={
            "model": args.model,
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "seed_buffer_size": args.seed_buffer_size,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "beta": args.beta,
            "algorithm": "absolute_zero_unified"
        }
    )
    
    print("\n" + "="*60)
    print("ABSOLUTE ZERO UNIFIED IMPLEMENTATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Iterations: {args.iterations}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed buffer size: {args.seed_buffer_size}")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = UnifiedAbsoluteZeroTrainer(args.model, device)
    
    # Phase 1: Seeding (no gradients)
    trainer.seed_buffers(batch_size=args.batch_size, buffer_size=args.seed_buffer_size)
    
    # Phase 2: Joint training
    for iteration in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'='*60}")
        
        # Step 1: Propose phase - generate new tasks
        all_prompts = []
        all_info = []  # (task, task_type, role)
        
        print("\n[PROPOSE PHASE]")
        for task_type in ['deduction', 'abduction', 'induction']:
            # Get buffer for few-shot examples
            if task_type == 'deduction':
                buffer = trainer.deduction_buffer
            elif task_type == 'abduction':
                buffer = trainer.abduction_buffer
            else:
                buffer = trainer.induction_buffer
            
            # Generate new tasks
            buffer_samples = buffer.sample(4)
            if not buffer_samples:
                continue
            
            # Create proposer prompts
            # Each iteration should have equal proposer/solver samples
            # With 3 task types, allocate batch_size/6 per task type per role
            for _ in range(args.batch_size // 6):  # Divide by 6: 3 task types × 2 roles
                prompt = trainer.create_proposer_prompt(task_type, buffer_samples)
                all_prompts.append(prompt)
                all_info.append((None, task_type, 'proposer'))
        
        # Step 2: Solve phase - attempt existing tasks
        print("\n[SOLVE PHASE]")
        for task_type in ['deduction', 'abduction', 'induction']:
            # Get buffer
            if task_type == 'deduction':
                buffer = trainer.deduction_buffer
            elif task_type == 'abduction':
                buffer = trainer.abduction_buffer
            else:
                buffer = trainer.induction_buffer
            
            # Sample tasks to solve
            tasks = buffer.sample(args.batch_size // 6)  # Match proposer allocation
            for task in tasks:
                prompt = trainer.create_solver_prompt(task, task_type)
                all_prompts.append(prompt)
                all_info.append((task, task_type, 'solver'))
        
        if not all_prompts:
            print("No prompts generated, skipping iteration")
            continue
        
        # Step 3: Create dataset
        dataset = Dataset.from_dict({'prompt': all_prompts})
        
        # Step 4: Define combined reward function
        def combined_reward_function(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
            rewards = []
            batch_indices = kwargs.get('batch_indices', [])
            
            for i, completion in enumerate(completions):
                if batch_indices:
                    task, task_type, role = all_info[batch_indices[i]]
                else:
                    task, task_type, role = all_info[i]
                
                prompt = prompts[i] if prompts else all_prompts[i]
                completion_only = completion[len(prompt):].strip()
                
                if role == 'proposer':
                    # Parse and validate the generated task
                    parsed_task = trainer.parse_proposer_generation(completion_only, task_type)
                    
                    if parsed_task:
                        # Add to buffer
                        if task_type == 'deduction':
                            trainer.deduction_buffer.add(parsed_task)
                        elif task_type == 'abduction':
                            trainer.abduction_buffer.add(parsed_task)
                        else:
                            trainer.induction_buffer.add(parsed_task)
                        
                        # Learnability reward (simplified: reward diversity and validity)
                        base_reward = 0.5  # Valid generation
                        
                        # Bonus for appropriate difficulty
                        if task_type == 'deduction' and 5 <= parsed_task['a'] <= 15:
                            base_reward += 0.3
                        elif task_type == 'abduction' and 10 <= parsed_task['result'] <= 30:
                            base_reward += 0.3
                        
                        reward = base_reward
                    else:
                        reward = -1.0  # Invalid generation
                    
                else:  # solver
                    # Evaluate correctness
                    reward = trainer.evaluate_solver_response(completion_only, task, task_type)
                    
                    # Track performance for learnability
                    trainer.solver_performance[task_type].append(reward)
                
                # Update baseline and compute advantage
                trainer.baselines.update(role, task_type, reward)
                advantage = trainer.baselines.compute_advantage(role, task_type, reward)
                rewards.append(reward)  # Return raw reward, GRPO will handle advantage internally
            
            return rewards
        
        # Step 5: Configure training
        # Adjust batch size for gradient accumulation
        per_device_batch = min(args.batch_size // args.gradient_accumulation_steps, len(dataset))
        
        config = GRPOConfig(
            output_dir=f"./absolute_zero_unified_iter_{iteration}",
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=1,
            learning_rate=args.learning_rate,
            logging_steps=1,
            save_steps=1000,
            report_to="wandb",
            remove_unused_columns=False,
            num_generations=4 if args.batch_size <= 16 else (8 if args.batch_size <= 32 else 16),  # Scale with batch size
            temperature=args.temperature,
            max_completion_length=64,
            max_prompt_length=256,
            seed=SEED + iteration,
            beta=args.beta,
            loss_type="grpo",
            dataloader_num_workers=0,
            bf16=True,
            gradient_checkpointing=True,
        )
        
        # Step 6: Single GRPO update
        grpo_trainer = GRPOTrainer(
            model=trainer.model,
            args=config,
            train_dataset=dataset,
            reward_funcs=[combined_reward_function],
            callbacks=[create_epoch_sample_logger('unified', iteration + 1, trainer.tokenizer, trainer)]
        )
        
        print(f"\n[TRAINING] Single RL update with {len(dataset)} samples...")
        grpo_trainer.train()
        
        # Log metrics
        wandb.log({
            "iteration": iteration + 1,
            "buffer_sizes/deduction": len(trainer.deduction_buffer),
            "buffer_sizes/abduction": len(trainer.abduction_buffer),
            "buffer_sizes/induction": len(trainer.induction_buffer),
        })
        
        # Log baseline statistics
        for key, values in trainer.baselines.baselines.items():
            if len(values) > 0:
                role, task_type = key.split('_', 1)
                wandb.log({
                    f"{role}/{task_type}_baseline_mean": np.mean(values),
                    f"{role}/{task_type}_baseline_std": np.std(values),
                })
        
        print(f"\nIteration {iteration + 1} complete!")
        print(f"Buffer sizes - Deduction: {len(trainer.deduction_buffer)}, "
              f"Abduction: {len(trainer.abduction_buffer)}, "
              f"Induction: {len(trainer.induction_buffer)}")
    
    wandb.finish()
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()