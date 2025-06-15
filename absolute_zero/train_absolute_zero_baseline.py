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
        
    def generate_problems(self, num_problems: int) -> Tuple[List[Dict[str, str]], List[Dict]]:
        """Use proposer to generate new arithmetic problems."""
        problems = []
        raw_generations = []  # Track all raw outputs for debugging
        
        # Prompt engineering for problem generation
        base_prompts = [
            "Generate a simple arithmetic problem in the format 'Calculate: X + Y = ': ",
            "Create a math problem with addition or subtraction in the format 'Calculate: X - Y = ': ",
            "Write an arithmetic problem with multiplication in the format 'Calculate: X * Y = ': ",
            "Create a challenging arithmetic problem in the format 'Calculate: X op Y = ' where op is +, -, or *: ",
            "Calculate: ",  # Simple prompt that might work better
        ]
        
        self.proposer_model.eval()
        
        print(f"\n[PROPOSER] Generating {num_problems} problems...")
        
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
            
            # Log raw generation
            raw_generations.append({
                'prompt_type': prompt.strip(),
                'raw_output': problem_text[:100],  # Truncate for logging
                'parsed': False
            })
            
            # Parse the generated problem
            # Expected format: "Calculate: X op Y = "
            if "Calculate:" in problem_text:
                parsed_problem = problem_text.split('=')[0].strip() + ' = '
                problems.append({
                    'prompt': parsed_problem,
                    'source': 'proposer',
                    'raw': problem_text
                })
                raw_generations[-1]['parsed'] = True
                raw_generations[-1]['final_problem'] = parsed_problem
            else:
                # Try to extract a valid problem
                match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)', problem_text)
                if match:
                    a, op, b = match.groups()
                    parsed_problem = f"Calculate: {a} {op} {b} = "
                    problems.append({
                        'prompt': parsed_problem,
                        'source': 'proposer',
                        'raw': problem_text
                    })
                    raw_generations[-1]['parsed'] = True
                    raw_generations[-1]['final_problem'] = parsed_problem
        
        # Print debug info
        parsed_count = sum(1 for g in raw_generations if g['parsed'])
        print(f"[PROPOSER] Successfully parsed {parsed_count}/{num_problems} problems")
        
        # Sample some raw outputs for debugging
        print("\n[PROPOSER] Sample raw outputs:")
        for i, gen in enumerate(raw_generations[:5]):
            print(f"  {i+1}. Prompt: '{gen['prompt_type']}'")
            print(f"     Raw: '{gen['raw_output']}'")
            if gen['parsed']:
                print(f"     Parsed: '{gen.get('final_problem', 'N/A')}'")
            else:
                print(f"     Parsed: FAILED")
        
        return problems, raw_generations
    
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
    
    def compute_learnability_rewards(self, results: List[Dict[str, Any]]) -> List[float]:
        """
        Compute per-problem learnability rewards for proposer based on solver performance.
        Returns a list of rewards, one for each problem.
        """
        if not results:
            return []
        
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
        
        # Base improvement reward (applies to all problems)
        base_improvement_reward = improvement * 2.0
        
        # Compute per-problem rewards
        rewards = []
        for result in results:
            # Start with base reward
            reward = base_improvement_reward
            
            # 1. Correctness bonus - reward problems the solver got right
            if result['is_correct']:
                reward += 0.5
            else:
                reward -= 0.2
            
            # 2. Difficulty-appropriate bonus
            difficulty = result['difficulty']
            if difficulty == 'medium':
                reward += 0.3  # Medium problems are ideal
            elif difficulty == 'easy':
                reward += 0.1  # Easy problems are okay
            else:  # hard
                reward += 0.2  # Hard problems are good if solver can handle them
            
            # 3. Source bonus - only reward proposer-generated problems
            if result['source'] == 'proposer':
                reward += 0.2
            else:
                reward = 0.0  # Seed problems get no reward for proposer
            
            rewards.append(reward)
        
        # Log aggregate statistics
        proposer_rewards = [r for r, res in zip(rewards, results) if res['source'] == 'proposer']
        if proposer_rewards:
            print(f"[PROPOSER] Average reward: {np.mean(proposer_rewards):.3f}")
            print(f"[PROPOSER] Reward range: [{min(proposer_rewards):.3f}, {max(proposer_rewards):.3f}]")
        
        return rewards
    
    def create_training_datasets(self, num_samples: int) -> Tuple[Dataset, Dataset, List[Dict], List[Dict]]:
        """Create training datasets for both proposer and solver."""
        # Generate problems using proposer
        print(f"Generating {num_samples} problems using proposer...")
        generated_problems, raw_generations = self.generate_problems(num_samples)
        
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
        
        # Compute per-problem learnability rewards for proposer
        learnability_rewards = self.compute_learnability_rewards(results)
        print(f"Computed {len(learnability_rewards)} learnability rewards")
        
        # Create solver dataset
        solver_data = {
            'prompt': [r['prompt'] for r in results],
            'answer': [r['correct_answer'] for r in results],
            'difficulty': [r['difficulty'] for r in results]
        }
        solver_dataset = Dataset.from_dict(solver_data)
        
        # Create proposer dataset (problems that resulted in good learning)
        # The proposer should learn to generate problems, not complete them
        # We need to train it on the generation prompts, not the problems themselves
        proposer_prompts = []
        proposer_completions = []
        
        # Use the same prompts we used for generation
        generation_prompts = [
            "Generate a simple arithmetic problem in the format 'Calculate: X + Y = ': ",
            "Create a math problem with addition or subtraction in the format 'Calculate: X - Y = ': ",
            "Write an arithmetic problem with multiplication in the format 'Calculate: X * Y = ': ",
            "Create a challenging arithmetic problem in the format 'Calculate: X op Y = ' where op is +, -, or *: ",
            "Calculate: ",
        ]
        
        # Create training data where the proposer learns to complete prompts with good problems
        proposer_rewards_list = []
        for i, r in enumerate(results):
            if r['source'] == 'proposer':  # Train on all proposer problems, not just correct ones
                # Create multiple training examples with different prompts
                prompt = random.choice(generation_prompts)
                completion = r['prompt']  # The actual problem like "Calculate: 5 + 3 = "
                
                proposer_prompts.append(prompt)
                proposer_completions.append(completion)
                proposer_rewards_list.append(learnability_rewards[i])
        
        # If no valid proposer problems, create some training data from seed problems
        if len(proposer_prompts) == 0:
            print("[WARNING] No valid proposer problems found, using seed problems for training")
            for i, r in enumerate(results[:10]):  # Use first 10 results
                if r['source'] == 'seed':
                    prompt = random.choice(generation_prompts)
                    completion = r['prompt']
                    proposer_prompts.append(prompt)
                    proposer_completions.append(completion)
                    # Give seed problems a small positive reward to bootstrap learning
                    proposer_rewards_list.append(0.1)
        
        # Create dataset with proper format for GRPO
        # The 'prompt' field should contain the full text (prompt + completion)
        # for the model to learn the generation pattern
        proposer_data = {
            'prompt': proposer_prompts,
            'completion': proposer_completions,  # Store completions separately
            'learnability_reward': proposer_rewards_list  # Use individual rewards
        }
        proposer_dataset = Dataset.from_dict(proposer_data)
        
        return proposer_dataset, solver_dataset, raw_generations, results


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
    
    # Initialize WandB tables for tracking
    proposer_table = wandb.Table(columns=["iteration", "prompt_type", "raw_output", "parsed_problem", "parsed_success"])
    solver_table = wandb.Table(columns=["iteration", "problem", "solver_answer", "correct_answer", "is_correct", "difficulty"])
    
    # Main training loop
    for iteration in range(total_iterations):
        print(f"\\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{total_iterations}")
        print(f"{'='*60}")
        
        # Create datasets for this iteration
        proposer_dataset, solver_dataset, raw_generations, solver_results = trainer.create_training_datasets(samples_per_iteration)
        
        # Log proposer generations to WandB table
        for gen in raw_generations[:20]:  # Log first 20 for visibility
            proposer_table.add_data(
                iteration + 1,
                gen['prompt_type'],
                gen['raw_output'],
                gen.get('final_problem', 'N/A'),
                gen['parsed']
            )
        
        # Log solver results to WandB table
        for result in solver_results[:20]:  # Log first 20 for visibility
            solver_table.add_data(
                iteration + 1,
                result['prompt'],
                result['solver_answer'],
                result['correct_answer'],
                result['is_correct'],
                result['difficulty']
            )
        
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
                rewards = []
                batch_indices = kwargs.get('batch_indices', [])
                
                if prompts is None:
                    prompts = kwargs.get('prompt', [])
                
                for i, (completion, prompt) in enumerate(zip(completions, prompts)):
                    if batch_indices and i < len(batch_indices):
                        # Get the pre-computed reward from the dataset
                        reward = proposer_dataset[batch_indices[i]]['learnability_reward']
                    else:
                        # Default reward for generation - check if it's a valid problem
                        # Extract the generated problem
                        generated = completion.strip()
                        
                        # Check if it's a valid arithmetic problem
                        import re
                        match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)', generated)
                        if match:
                            reward = 0.1  # Small positive reward for valid format
                        else:
                            reward = -0.5  # Negative reward for invalid format
                    
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
    
    # Log WandB tables
    wandb.log({"proposer_generations": proposer_table})
    wandb.log({"solver_results": solver_table})
    
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