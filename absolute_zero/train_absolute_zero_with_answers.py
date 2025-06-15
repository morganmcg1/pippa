#!/usr/bin/env python3
"""
Absolute Zero where proposer generates problems WITH answers.
This tests the proposer's actual arithmetic understanding.
"""

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
from dotenv import load_dotenv
import os
import wandb
import re
from typing import Dict, Any, List, Tuple
from collections import deque

# Load environment variables
load_dotenv()

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def parse_problem_with_answer(text: str) -> Tuple[str, str, bool]:
    """
    Parse a problem that includes an answer.
    Expected format: "Calculate: 5 + 3 = 8"
    Returns: (problem_without_answer, given_answer, is_valid)
    """
    # Look for pattern: Calculate: X op Y = Z
    match = re.search(r'Calculate:\s*(\d+)\s*([+\-*])\s*(\d+)\s*=\s*(\d+)', text)
    if match:
        a, op, b, given_answer = match.groups()
        problem = f"Calculate: {a} {op} {b} = "
        return problem, given_answer, True
    
    # Fallback: look for X op Y = Z
    match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)\s*=\s*(\d+)', text)
    if match:
        a, op, b, given_answer = match.groups()
        problem = f"Calculate: {a} {op} {b} = "
        return problem, given_answer, True
    
    return "", "", False


def verify_arithmetic(problem: str, given_answer: str) -> Tuple[bool, str]:
    """
    Verify if the given answer is correct for the problem.
    Returns: (is_correct, correct_answer)
    """
    match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)', problem)
    if not match:
        return False, ""
    
    a, op, b = match.groups()
    a, b = int(a), int(b)
    
    if op == '+':
        correct = str(a + b)
    elif op == '-':
        correct = str(a - b)
    elif op == '*':
        correct = str(a * b)
    else:
        return False, ""
    
    return given_answer == correct, correct


class ProposerWithAnswers:
    """Proposer that generates complete problems with answers."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate_problems_with_answers(self, num_problems: int) -> List[Dict[str, Any]]:
        """Generate problems with answers, validating correctness."""
        
        # Few-shot prompts showing problems WITH answers
        base_prompts = [
            "Generate arithmetic with answer: Calculate: 5 + 3 = 8\nGenerate arithmetic with answer: Calculate: 12 - 7 = 5\nGenerate arithmetic with answer: ",
            "Problem: Calculate: 8 + 4 = 12\nProblem: Calculate: 15 - 6 = 9\nProblem: ",
            "Example: Calculate: 3 * 4 = 12\nExample: Calculate: 7 * 2 = 14\nExample: ",
            "Math: Calculate: 9 + 2 = 11\nMath: Calculate: 6 * 3 = 18\nMath: ",
        ]
        
        problems = []
        attempts = 0
        max_attempts = num_problems * 5
        
        self.model.eval()
        
        print(f"\n[PROPOSER] Generating {num_problems} problems with answers...")
        
        while len(problems) < num_problems and attempts < max_attempts:
            prompt = random.choice(base_prompts)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=0.7,  # Lower temp for more accurate answers
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_return_sequences=4
                )
            
            for output in outputs:
                generated = self.tokenizer.decode(output, skip_special_tokens=True)
                completion = generated[len(prompt):].strip()
                
                # Parse problem with answer
                problem, given_answer, is_valid = parse_problem_with_answer(completion)
                
                if is_valid:
                    # Verify the answer is correct
                    is_correct, correct_answer = verify_arithmetic(problem, given_answer)
                    
                    problems.append({
                        'prompt': problem,
                        'given_answer': given_answer,
                        'correct_answer': correct_answer,
                        'is_correct': is_correct,
                        'source': 'proposer',
                        'raw': completion
                    })
                    
                    if len(problems) >= num_problems:
                        break
                
                attempts += 1
        
        # Calculate statistics
        correct_count = sum(1 for p in problems if p['is_correct'])
        accuracy = correct_count / len(problems) if problems else 0
        
        print(f"[PROPOSER] Generated {len(problems)} problems from {attempts} attempts")
        print(f"[PROPOSER] Arithmetic accuracy: {accuracy:.1%} ({correct_count}/{len(problems)} correct)")
        
        # Log examples
        if problems:
            print("\n[PROPOSER] Sample problems:")
            for i, p in enumerate(problems[:3]):
                status = "✓" if p['is_correct'] else "✗"
                print(f"  {i+1}. {p['prompt']}{p['given_answer']} {status}")
                if not p['is_correct']:
                    print(f"      (correct answer: {p['correct_answer']})")
        
        return problems


def compute_proposer_rewards(problems: List[Dict[str, Any]], solver_performance: float) -> List[float]:
    """
    Compute rewards for proposer based on:
    1. Arithmetic correctness of generated answers
    2. Problem difficulty appropriateness
    3. Solver performance on the problems
    """
    rewards = []
    
    for problem in problems:
        reward = 0.0
        
        # 1. Correctness reward (most important)
        if problem['is_correct']:
            reward += 1.0
        else:
            reward -= 0.5  # Penalty for wrong answers
        
        # 2. Solver performance bonus
        # If solver is doing well on these problems, they're appropriate
        reward += solver_performance * 0.5
        
        # 3. Source bonus (only for proposer-generated)
        if problem['source'] == 'proposer':
            reward += 0.1
        else:
            reward = 0.0  # No reward for seed problems
        
        rewards.append(reward)
    
    return rewards


def main():
    """Main training loop with answer-aware proposer."""
    
    # Configuration
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    proposer_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    solver_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Initialize components
    proposer = ProposerWithAnswers(proposer_model, tokenizer, device)
    
    # Training configuration
    iterations = 20
    problems_per_iter = 100
    
    print("\n" + "="*60)
    print("ABSOLUTE ZERO WITH ANSWER GENERATION")
    print("="*60)
    print("Proposer must generate both problems AND correct answers")
    print("="*60 + "\n")
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="absolute_zero_with_answers",
        tags=["absolute-zero", "arithmetic", "answer-aware"],
        config={
            "model": model_name,
            "iterations": iterations,
            "problems_per_iter": problems_per_iter,
            "proposer_generates_answers": True
        }
    )
    
    # Training loop
    for iteration in range(iterations):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration + 1}/{iterations}")
        print(f"{'='*50}")
        
        # Generate problems with answers
        problems = proposer.generate_problems_with_answers(problems_per_iter)
        
        # Log proposer accuracy
        proposer_accuracy = sum(1 for p in problems if p['is_correct']) / len(problems)
        wandb.log({
            "proposer/arithmetic_accuracy": proposer_accuracy,
            "proposer/problems_generated": len(problems),
            "iteration": iteration + 1
        })
        
        # Create solver dataset (only from correct problems)
        correct_problems = [p for p in problems if p['is_correct']]
        if len(correct_problems) < 10:
            print(f"[WARNING] Only {len(correct_problems)} correct problems, adding seed problems")
            # Add seed problems...
        
        # Train solver on correct problems
        solver_data = {
            'prompt': [p['prompt'] for p in correct_problems],
            'answer': [p['correct_answer'] for p in correct_problems]
        }
        solver_dataset = Dataset.from_dict(solver_data)
        
        # ... (continue with solver training)
        
        # Compute proposer rewards
        solver_accuracy = 0.5  # Placeholder - get from actual evaluation
        proposer_rewards = compute_proposer_rewards(problems, solver_accuracy)
        
        # Train proposer with rewards based on correctness
        # ... (proposer training code)
        
    wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()