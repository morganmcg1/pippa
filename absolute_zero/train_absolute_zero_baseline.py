#!/usr/bin/env python3
"""
Absolute Zero baseline implementation for arithmetic tasks.
Based on https://arxiv.org/pdf/2505.03335v2 and our successful GRPO experiments.

Key components:
1. Proposer: Generates new arithmetic problems
2. Solver: Solves the generated problems  
3. Learnability reward: Proposer rewarded for problems that help solver improve
4. TRR++ algorithm: Task-Relative REINFORCE++ with 6 separate baselines
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


class TRRPlusBaselines:
    """Task-Relative REINFORCE++ with 6 separate baselines."""
    def __init__(self, num_baselines=6):
        # 6 baselines: proposer/solver × easy/medium/hard
        self.baselines = {
            'proposer_easy': deque(maxlen=100),
            'proposer_medium': deque(maxlen=100),
            'proposer_hard': deque(maxlen=100),
            'solver_easy': deque(maxlen=100),
            'solver_medium': deque(maxlen=100),
            'solver_hard': deque(maxlen=100),
        }
        
    def update(self, role: str, difficulty: str, reward: float):
        """Update baseline for specific role and difficulty."""
        key = f"{role}_{difficulty}"
        if key in self.baselines:
            self.baselines[key].append(reward)
    
    def get_baseline(self, role: str, difficulty: str) -> float:
        """Get baseline value for specific role and difficulty."""
        key = f"{role}_{difficulty}"
        if key in self.baselines and len(self.baselines[key]) > 0:
            return np.mean(self.baselines[key])
        return 0.0
    
    def compute_advantage(self, role: str, difficulty: str, reward: float) -> float:
        """Compute advantage using task-relative baseline."""
        baseline = self.get_baseline(role, difficulty)
        # Update baseline with new reward
        self.update(role, difficulty, reward)
        return reward - baseline


def classify_problem_difficulty(problem: str, answer: str) -> str:
    """Classify arithmetic problem difficulty based on numbers and operations."""
    # Extract numbers from problem
    numbers = re.findall(r'\d+', problem)
    if not numbers:
        return 'easy'
    
    nums = [int(n) for n in numbers]
    max_num = max(nums)
    
    # Check operation type
    has_multiply = '*' in problem
    has_subtract = '-' in problem and answer.startswith('-')
    
    # Classification logic
    if max_num <= 5 and not has_multiply:
        return 'easy'
    elif max_num <= 10 or (max_num <= 20 and not has_multiply):
        return 'medium'
    else:
        return 'hard'


def extract_answer(text: str, prompt: str) -> str:
    """Extract the answer from model output."""
    completion = text[len(prompt):].strip()
    match = re.match(r'^-?\d+', completion)
    if match:
        return match.group(0)
    tokens = completion.split()
    if tokens:
        return tokens[0]
    return completion


def evaluate_on_standard_dataset(model, tokenizer, device) -> Dict[str, float]:
    """Evaluate model on the standardized arithmetic evaluation dataset."""
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    
    correct_total = 0
    model.eval()
    
    with torch.no_grad():
        for sample in eval_dataset:
            prompt = sample['prompt']
            expected = sample['answer']
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            completion = response[len(prompt):].strip()
            match = re.match(r'^-?\d+', completion)
            predicted = match.group(0) if match else completion.split()[0] if completion else ""
            
            if predicted == expected:
                correct_total += 1
    
    return {
        'eval_accuracy': correct_total / len(eval_dataset),
        'eval_correct': correct_total,
        'eval_total': len(eval_dataset)
    }


class AbsoluteZeroTrainer:
    """Main trainer for Absolute Zero algorithm."""
    
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
        self.baselines = TRRPlusBaselines()
        
        # Buffer for generated problems
        self.problem_buffer = deque(maxlen=1000)
        
        # Tracking solver performance for learnability rewards
        self.solver_history = deque(maxlen=100)
        
    def generate_problems(self, num_problems: int) -> List[Dict[str, str]]:
        """Use proposer to generate new arithmetic problems."""
        problems = []
        
        # Prompt engineering for problem generation
        base_prompts = [
            "Generate a simple arithmetic problem: ",
            "Create a math problem with addition or subtraction: ",
            "Write an arithmetic problem with multiplication: ",
            "Create a challenging arithmetic problem: ",
        ]
        
        self.proposer_model.eval()
        
        for i in range(num_problems):
            prompt = random.choice(base_prompts)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.proposer_model.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=1.0,  # High temperature for diversity
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            problem_text = generated[len(prompt):].strip()
            
            # Parse the generated problem
            # Expected format: "Calculate: X op Y = "
            if "Calculate:" in problem_text:
                problems.append({
                    'prompt': problem_text.split('=')[0].strip() + ' = ',
                    'source': 'proposer'
                })
            else:
                # Try to extract a valid problem
                match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)', problem_text)
                if match:
                    a, op, b = match.groups()
                    problems.append({
                        'prompt': f"Calculate: {a} {op} {b} = ",
                        'source': 'proposer'
                    })
        
        return problems
    
    def solve_problems(self, problems: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Use solver to solve generated problems and compute answers."""
        results = []
        self.solver_model.eval()
        
        for problem in problems:
            prompt = problem['prompt']
            
            # Generate solution
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.solver_model.generate(
                    **inputs,
                    max_new_tokens=16,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            solver_answer = extract_answer(response, prompt)
            
            # Compute the correct answer
            match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)', prompt)
            if match:
                a, op, b = match.groups()
                a, b = int(a), int(b)
                
                if op == '+':
                    correct_answer = str(a + b)
                elif op == '-':
                    correct_answer = str(a - b)
                elif op == '*':
                    correct_answer = str(a * b)
                else:
                    correct_answer = None
                
                if correct_answer:
                    is_correct = solver_answer == correct_answer
                    difficulty = classify_problem_difficulty(prompt, correct_answer)
                    
                    results.append({
                        'prompt': prompt,
                        'solver_answer': solver_answer,
                        'correct_answer': correct_answer,
                        'is_correct': is_correct,
                        'difficulty': difficulty,
                        'source': problem['source']
                    })
        
        return results
    
    def compute_learnability_reward(self, results: List[Dict[str, Any]]) -> float:
        """
        Compute learnability reward for proposer based on solver performance.
        Reward is based on whether the problems helped the solver improve.
        """
        if not results:
            return 0.0
        
        # Current solver accuracy on proposed problems
        current_accuracy = sum(r['is_correct'] for r in results) / len(results)
        self.solver_history.append(current_accuracy)
        
        # Compute improvement (if we have history)
        if len(self.solver_history) > 10:
            recent_avg = np.mean(list(self.solver_history)[-10:])
            previous_avg = np.mean(list(self.solver_history)[-20:-10])
            improvement = recent_avg - previous_avg
        else:
            improvement = current_accuracy - 0.25  # Assume 25% baseline
        
        # Learnability reward components
        # 1. Improvement bonus
        improvement_reward = improvement * 2.0
        
        # 2. Optimal difficulty bonus (problems should be neither too easy nor too hard)
        difficulty_distribution = {'easy': 0, 'medium': 0, 'hard': 0}
        for r in results:
            difficulty_distribution[r['difficulty']] += 1
        
        total = len(results)
        optimal_distribution = {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}
        
        difficulty_penalty = 0
        for diff in ['easy', 'medium', 'hard']:
            actual_ratio = difficulty_distribution[diff] / total
            expected_ratio = optimal_distribution[diff]
            difficulty_penalty += abs(actual_ratio - expected_ratio)
        
        difficulty_reward = 1.0 - difficulty_penalty
        
        # 3. Diversity bonus (reward variety in problems)
        unique_patterns = len(set(r['prompt'].split()[2] for r in results if len(r['prompt'].split()) > 2))
        diversity_reward = min(unique_patterns / 3.0, 1.0)  # Reward for using different operations
        
        # Combined learnability reward
        learnability_reward = (
            0.5 * improvement_reward +
            0.3 * difficulty_reward +
            0.2 * diversity_reward
        )
        
        return learnability_reward
    
    def create_training_datasets(self, num_samples: int) -> Tuple[Dataset, Dataset]:
        """Create training datasets for both proposer and solver."""
        # Generate problems using proposer
        print(f"Generating {num_samples} problems using proposer...")
        generated_problems = self.generate_problems(num_samples)
        
        # Add some seed problems to ensure initial learning
        seed_problems = []
        for _ in range(min(20, num_samples // 5)):
            a = random.randint(0, 10)
            b = random.randint(0, 10)
            op = random.choice(['+', '-', '*'])
            seed_problems.append({
                'prompt': f"Calculate: {a} {op} {b} = ",
                'source': 'seed'
            })
        
        all_problems = generated_problems + seed_problems
        
        # Solve problems to get answers and compute rewards
        print("Solving generated problems...")
        results = self.solve_problems(all_problems)
        
        # Compute learnability reward for proposer
        learnability_reward = self.compute_learnability_reward(results)
        print(f"Learnability reward: {learnability_reward:.3f}")
        
        # Create solver dataset
        solver_data = {
            'prompt': [r['prompt'] for r in results],
            'answer': [r['correct_answer'] for r in results],
            'difficulty': [r['difficulty'] for r in results]
        }
        solver_dataset = Dataset.from_dict(solver_data)
        
        # Create proposer dataset (problems that resulted in good learning)
        # For now, we'll use a simple approach: reward proposer for all generated problems
        # weighted by learnability
        proposer_prompts = [
            f"Generate an arithmetic problem that helps learning: Calculate: {r['prompt'].split('Calculate: ')[1]}"
            for r in results if r['source'] == 'proposer'
        ]
        
        proposer_data = {
            'prompt': proposer_prompts,
            'learnability_reward': [learnability_reward] * len(proposer_prompts)
        }
        proposer_dataset = Dataset.from_dict(proposer_data)
        
        return proposer_dataset, solver_dataset


def main():
    # Configuration
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training configuration (based on GRPO success)
    total_iterations = 20
    samples_per_iteration = 100
    solver_epochs_per_iteration = 5
    proposer_epochs_per_iteration = 3
    batch_size = 32  # Smaller due to two models in memory
    learning_rate = 5e-6
    temperature = 0.7
    beta = 0.1  # KL penalty
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="absolute_zero_baseline",
        tags=["absolute-zero", "arithmetic", "self-play"],
        config={
            "model": model_name,
            "total_iterations": total_iterations,
            "samples_per_iteration": samples_per_iteration,
            "solver_epochs": solver_epochs_per_iteration,
            "proposer_epochs": proposer_epochs_per_iteration,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "beta": beta,
            "algorithm": "absolute_zero",
            "baselines": "TRR++_6_baselines"
        }
    )
    
    print("\\n" + "="*60)
    print("ABSOLUTE ZERO BASELINE IMPLEMENTATION")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Total iterations: {total_iterations}")
    print(f"Samples per iteration: {samples_per_iteration}")
    print("="*60 + "\\n")
    
    # Initialize trainer
    trainer = AbsoluteZeroTrainer(model_name, device)
    
    # Main training loop
    for iteration in range(total_iterations):
        print(f"\\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{total_iterations}")
        print(f"{'='*60}")
        
        # Create datasets for this iteration
        proposer_dataset, solver_dataset = trainer.create_training_datasets(samples_per_iteration)
        
        # Train solver
        print(f"\\nTraining solver for {solver_epochs_per_iteration} epochs...")
        
        # Define solver reward function
        def solver_reward_function(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
            rewards = []
            batch_indices = kwargs.get('batch_indices', [])
            
            if prompts is None:
                prompts = kwargs.get('prompt', [])
            
            for i, (completion, prompt) in enumerate(zip(completions, prompts)):
                extracted = extract_answer(completion, prompt)
                
                # Get the correct answer from dataset
                if batch_indices:
                    expected = solver_dataset[batch_indices[i]]['answer']
                else:
                    # Fallback: compute answer from prompt
                    match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)', prompt)
                    if match:
                        a, op, b = match.groups()
                        a, b = int(a), int(b)
                        if op == '+':
                            expected = str(a + b)
                        elif op == '-':
                            expected = str(a - b)
                        elif op == '*':
                            expected = str(a * b)
                        else:
                            expected = ""
                    else:
                        expected = ""
                
                # Binary reward
                reward = 1.0 if extracted == expected else -1.0
                rewards.append(reward)
                
                # Update baselines
                difficulty = solver_dataset[batch_indices[i]]['difficulty'] if batch_indices else 'medium'
                trainer.baselines.update('solver', difficulty, reward)
            
            return rewards
        
        # Configure solver training
        solver_config = GRPOConfig(
            output_dir=f"./absolute_zero_solver_iter_{iteration}",
            per_device_train_batch_size=batch_size,
            num_train_epochs=solver_epochs_per_iteration,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=1000,
            report_to="wandb",
            remove_unused_columns=False,
            num_generations=16,
            temperature=temperature,
            max_completion_length=16,
            max_prompt_length=128,
            seed=SEED + iteration,
            beta=beta,
            loss_type="grpo",
            dataloader_num_workers=0,
            bf16=True,
            gradient_checkpointing=True,
        )
        
        # For the first iteration, use model name; for subsequent iterations, we need to save and reload
        if iteration == 0:
            solver_model_path = model_name
        else:
            # Save current solver model
            solver_model_path = f"./absolute_zero_solver_iter_{iteration-1}"
            trainer.solver_model.save_pretrained(solver_model_path)
            trainer.tokenizer.save_pretrained(solver_model_path)
        
        # Create custom GRPO trainer for solver
        solver_trainer = GRPOTrainer(
            model=solver_model_path,
            args=solver_config,
            train_dataset=solver_dataset,
            reward_funcs=[solver_reward_function]
        )
        
        # Train solver
        solver_trainer.train()
        
        # Update solver model
        trainer.solver_model = solver_trainer.model
        trainer.tokenizer = solver_trainer.tokenizer
        
        # Evaluate solver on standard dataset
        print("\\nEvaluating solver on standard dataset...")
        solver_metrics = evaluate_on_standard_dataset(trainer.solver_model, trainer.tokenizer, device)
        print(f"Solver accuracy: {solver_metrics['eval_accuracy']:.2%}")
        
        wandb.log({
            "iteration": iteration + 1,
            "solver/eval_accuracy": solver_metrics['eval_accuracy'],
            "solver/eval_correct": solver_metrics['eval_correct'],
        })
        
        # Train proposer (if we have enough data)
        if len(proposer_dataset) > 0:
            print(f"\\nTraining proposer for {proposer_epochs_per_iteration} epochs...")
            
            # Define proposer reward function based on learnability
            def proposer_reward_function(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
                # For proposer, the reward is based on learnability
                # This is simplified - in practice, we'd track which problems helped solver improve
                rewards = []
                batch_indices = kwargs.get('batch_indices', [])
                
                for i in range(len(completions)):
                    if batch_indices:
                        reward = proposer_dataset[batch_indices[i]]['learnability_reward']
                    else:
                        reward = 0.0  # Default reward
                    
                    rewards.append(reward)
                    
                    # Update baselines (assume medium difficulty for now)
                    trainer.baselines.update('proposer', 'medium', reward)
                
                return rewards
            
            # Configure proposer training
            proposer_config = GRPOConfig(
                output_dir=f"./absolute_zero_proposer_iter_{iteration}",
                per_device_train_batch_size=min(batch_size, len(proposer_dataset)),
                num_train_epochs=proposer_epochs_per_iteration,
                learning_rate=learning_rate,
                logging_steps=10,
                save_steps=1000,
                report_to="wandb",
                remove_unused_columns=False,
                num_generations=8,  # Fewer generations for proposer
                temperature=1.0,  # Higher temperature for diversity
                max_completion_length=64,
                max_prompt_length=128,
                seed=SEED + iteration + 1000,
                beta=beta,
                loss_type="grpo",
                dataloader_num_workers=0,
                bf16=True,
                gradient_checkpointing=True,
            )
            
            # For the first iteration, use model name; for subsequent iterations, we need to save and reload
            if iteration == 0:
                proposer_model_path = model_name
            else:
                # Save current proposer model
                proposer_model_path = f"./absolute_zero_proposer_iter_{iteration-1}"
                trainer.proposer_model.save_pretrained(proposer_model_path)
                trainer.tokenizer.save_pretrained(proposer_model_path)
            
            # Create custom GRPO trainer for proposer
            proposer_trainer = GRPOTrainer(
                model=proposer_model_path,
                args=proposer_config,
                train_dataset=proposer_dataset,
                reward_funcs=[proposer_reward_function]
            )
            
            # Train proposer
            proposer_trainer.train()
            
            # Update proposer model
            trainer.proposer_model = proposer_trainer.model
        
        # Log baseline statistics
        baseline_stats = {}
        for key, values in trainer.baselines.baselines.items():
            if len(values) > 0:
                baseline_stats[f"baseline/{key}_mean"] = np.mean(values)
                baseline_stats[f"baseline/{key}_std"] = np.std(values)
        
        wandb.log(baseline_stats)
        
        print(f"\\nIteration {iteration + 1} complete!")
        print(f"Solver accuracy: {solver_metrics['eval_accuracy']:.2%}")
        print(f"Baselines updated: {len(baseline_stats)} values")
    
    # Final evaluation
    print("\\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    final_metrics = evaluate_on_standard_dataset(trainer.solver_model, trainer.tokenizer, device)
    print(f"Final solver accuracy: {final_metrics['eval_accuracy']:.2%}")
    
    wandb.log({
        "final/eval_accuracy": final_metrics['eval_accuracy'],
        "final/eval_correct": final_metrics['eval_correct'],
        "final/eval_total": final_metrics['eval_total'],
    })
    
    # Save models
    print("\\nSaving models...")
    trainer.solver_model.save_pretrained("./absolute_zero_solver_final")
    trainer.proposer_model.save_pretrained("./absolute_zero_proposer_final")
    trainer.tokenizer.save_pretrained("./absolute_zero_tokenizer")
    
    wandb.finish()
    print("\\nTraining complete!")


if __name__ == "__main__":
    main()