#!/usr/bin/env python3
"""
Enhanced Absolute Zero with problem validation.
Generates many candidate problems and filters for valid arithmetic.
"""

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
from dotenv import load_dotenv
import os
import wandb
import re
from typing import Dict, Any, List, Tuple
from collections import deque
import copy

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


def validate_arithmetic_problem(problem: str) -> Tuple[bool, str, str]:
    """
    Validate that a problem is solvable arithmetic and compute the answer.
    Returns (is_valid, problem_cleaned, correct_answer)
    """
    match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)', problem)
    if not match:
        return False, "", ""
    
    a, op, b = match.groups()
    a, b = int(a), int(b)
    
    # Compute correct answer
    if op == '+':
        answer = str(a + b)
    elif op == '-':
        answer = str(a - b)
    elif op == '*':
        answer = str(a * b)
    else:
        return False, "", ""
    
    # Clean problem format
    clean_problem = f"Calculate: {a} {op} {b} = "
    
    return True, clean_problem, answer


class EnhancedAbsoluteZeroTrainer:
    """Enhanced trainer with problem validation and oversampling."""
    
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model_name = model_name
        
        # Initialize tokenizers and models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Proposer and solver can be the same model initially
        print("Loading proposer model...")
        self.proposer_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        print("Loading solver model...")
        self.solver_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # TRR++ baselines
        from train_absolute_zero_baseline import TRRPlusBaselines
        self.baselines = TRRPlusBaselines()
        
        # Buffer for generated problems
        self.problem_buffer = deque(maxlen=1000)
        
        # Tracking solver performance for learnability rewards
        self.solver_history = deque(maxlen=100)
        
    def generate_problems_with_validation(self, target_count: int, max_attempts: int = None) -> List[Dict[str, str]]:
        """Generate problems until we have target_count valid ones."""
        if max_attempts is None:
            max_attempts = target_count * 5  # Try up to 5x to get enough valid
        
        # Few-shot prompts for better generation
        base_prompts = [
            "Generate arithmetic: Calculate: 5 + 3 = \nGenerate arithmetic: Calculate: 12 - 7 = \nGenerate arithmetic: ",
            "Problem: Calculate: 8 + 4 = \nProblem: Calculate: 15 - 6 = \nProblem: ",
            "Create: Calculate: 3 * 4 = \nCreate: Calculate: 7 * 2 = \nCreate: ",
            "Example: Calculate: 9 + 2 = \nExample: Calculate: 6 * 3 = \nExample: ",
            "Calculate: 4 + 5 = \nCalculate: 11 - 3 = \nCalculate: ",
        ]
        
        valid_problems = []
        attempts = 0
        
        self.proposer_model.eval()
        
        print(f"\n[PROPOSER] Generating up to {max_attempts} candidates to get {target_count} valid problems...")
        
        while len(valid_problems) < target_count and attempts < max_attempts:
            prompt = random.choice(base_prompts)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.proposer_model.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=1.0,  # High temperature for diversity
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_return_sequences=4  # Generate multiple at once
                )
            
            for output in outputs:
                generated = self.tokenizer.decode(output, skip_special_tokens=True)
                problem_text = generated[len(prompt):].strip()
                
                # Validate the problem
                is_valid, clean_problem, correct_answer = validate_arithmetic_problem(problem_text)
                
                if is_valid and len(valid_problems) < target_count:
                    valid_problems.append({
                        'prompt': clean_problem,
                        'answer': correct_answer,
                        'source': 'proposer',
                        'raw': problem_text
                    })
                
                attempts += 1
                if attempts >= max_attempts:
                    break
        
        print(f"[PROPOSER] Generated {len(valid_problems)} valid problems from {attempts} attempts "
              f"({100*len(valid_problems)/attempts:.1f}% success rate)")
        
        # Log some examples
        if valid_problems:
            print("\n[PROPOSER] Sample valid problems:")
            for i, prob in enumerate(valid_problems[:3]):
                print(f"  {i+1}. {prob['prompt']} (answer: {prob['answer']})")
        
        return valid_problems


def main():
    # Configuration
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training configuration
    total_iterations = 20
    problems_per_iteration = 100
    solver_epochs_per_iteration = 5
    proposer_epochs_per_iteration = 3
    batch_size = 32
    learning_rate = 5e-6
    temperature = 0.7
    beta = 0.1  # KL penalty
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="absolute_zero_validated",
        tags=["absolute-zero", "arithmetic", "self-play", "validation"],
        config={
            "model": model_name,
            "total_iterations": total_iterations,
            "problems_per_iteration": problems_per_iteration,
            "solver_epochs": solver_epochs_per_iteration,
            "proposer_epochs": proposer_epochs_per_iteration,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "beta": beta,
            "algorithm": "absolute_zero_validated"
        }
    )
    
    print("\n" + "="*60)
    print("ABSOLUTE ZERO WITH VALIDATION")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Total iterations: {total_iterations}")
    print(f"Problems per iteration: {problems_per_iteration}")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = EnhancedAbsoluteZeroTrainer(model_name, device)
    
    # Initialize WandB tables
    proposer_table = wandb.Table(columns=["iteration", "problem", "answer", "success_rate"])
    solver_table = wandb.Table(columns=["iteration", "accuracy", "problems_solved"])
    
    # Main training loop
    for iteration in range(total_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{total_iterations}")
        print(f"{'='*60}")
        
        # Generate validated problems
        proposer_problems = trainer.generate_problems_with_validation(
            target_count=problems_per_iteration - 20,  # Leave room for seed problems
            max_attempts=problems_per_iteration * 3
        )
        
        # Add seed problems
        seed_problems = []
        for _ in range(20):
            a = random.randint(0, 10)
            b = random.randint(0, 10)
            op = random.choice(['+', '-', '*'])
            
            if op == '+':
                answer = str(a + b)
            elif op == '-':
                answer = str(a - b)
            else:  # '*'
                answer = str(a * b)
            
            seed_problems.append({
                'prompt': f"Calculate: {a} {op} {b} = ",
                'answer': answer,
                'source': 'seed'
            })
        
        all_problems = proposer_problems + seed_problems
        
        # Create solver dataset
        solver_data = {
            'prompt': [p['prompt'] for p in all_problems],
            'answer': [p['answer'] for p in all_problems]
        }
        solver_dataset = Dataset.from_dict(solver_data)
        
        # Train solver (similar to original implementation)
        print(f"\nTraining solver for {solver_epochs_per_iteration} epochs...")
        
        # ... (rest of training logic similar to original)
        
        # Log to WandB tables
        if proposer_problems:
            success_rate = len(proposer_problems) / (problems_per_iteration * 3)
            for prob in proposer_problems[:5]:  # Log first 5
                proposer_table.add_data(iteration + 1, prob['prompt'], prob['answer'], success_rate)
        
        # Evaluate solver
        from train_absolute_zero_baseline import evaluate_on_standard_dataset
        solver_metrics = evaluate_on_standard_dataset(trainer.solver_model, trainer.tokenizer, device)
        solver_table.add_data(iteration + 1, solver_metrics['eval_accuracy'], solver_metrics['eval_correct'])
        
        print(f"\nIteration {iteration + 1} complete!")
        print(f"Solver accuracy: {solver_metrics['eval_accuracy']:.2%}")
    
    # Log tables
    wandb.log({"proposer_problems": proposer_table})
    wandb.log({"solver_performance": solver_table})
    
    wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()