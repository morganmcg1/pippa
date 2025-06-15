#!/usr/bin/env python3
"""
Script to generate and log sample outputs from saved checkpoints.
Run this periodically to see what the models are generating.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import re
import os
import argparse
from datetime import datetime

def generate_proposer_samples(model, tokenizer, device, num_samples=10):
    """Generate sample problems from the proposer."""
    model.eval()
    
    prompts = [
        "Generate arithmetic: Calculate: 5 + 3 = \nGenerate arithmetic: Calculate: 12 - 7 = \nGenerate arithmetic: ",
        "Problem: Calculate: 8 + 4 = \nProblem: Calculate: 15 - 6 = \nProblem: ",
        "Create: Calculate: 3 * 4 = \nCreate: Calculate: 7 * 2 = \nCreate: ",
        "Example: Calculate: 9 + 2 = \nExample: Calculate: 6 * 3 = \nExample: ",
        "Calculate: 4 + 5 = \nCalculate: 11 - 3 = \nCalculate: ",
    ]
    
    samples = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(prompts))):
            prompt = prompts[i % len(prompts)]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=2  # Generate 2 samples per prompt
            )
            
            for output in outputs:
                generated = tokenizer.decode(output, skip_special_tokens=True)
                completion = generated[len(prompt):].strip()
                
                # Try to parse the problem
                match = re.search(r'Calculate:\s*(\d+)\s*([+\-*])\s*(\d+)\s*=', completion)
                if not match:
                    match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)\s*=', completion)
                
                parsed = match.group(0) if match else "PARSE_FAILED"
                
                samples.append({
                    'prompt_template': prompt.split('\n')[-1][:30] + "...",
                    'raw_generation': completion[:100],
                    'parsed_problem': parsed,
                    'valid': parsed != "PARSE_FAILED"
                })
    
    return samples


def generate_solver_samples(model, tokenizer, device, num_samples=10):
    """Generate sample solutions from the solver."""
    model.eval()
    
    test_problems = [
        "Calculate: 5 + 3 = ",
        "Calculate: 12 - 7 = ",
        "Calculate: 8 * 4 = ",
        "Calculate: 15 + 6 = ",
        "Calculate: 9 - 2 = ",
        "Calculate: 3 * 7 = ",
        "Calculate: 20 - 15 = ",
        "Calculate: 4 + 9 = ",
        "Calculate: 6 * 5 = ",
        "Calculate: 18 - 9 = ",
    ]
    
    samples = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_problems))):
            prompt = test_problems[i]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated[len(prompt):].strip()
            
            # Extract answer
            match = re.match(r'^-?\d+', completion)
            predicted = match.group(0) if match else completion.split()[0] if completion else ""
            
            # Calculate correct answer
            problem_match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)', prompt)
            if problem_match:
                a, op, b = problem_match.groups()
                a, b = int(a), int(b)
                if op == '+':
                    correct = str(a + b)
                elif op == '-':
                    correct = str(a - b)
                elif op == '*':
                    correct = str(a * b)
                else:
                    correct = "?"
            else:
                correct = "?"
            
            is_correct = predicted == correct
            
            samples.append({
                'problem': prompt,
                'predicted_answer': predicted,
                'correct_answer': correct,
                'is_correct': is_correct,
                'raw_output': completion[:50]
            })
    
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=None, 
                        help='Iteration number to load (default: latest)')
    parser.add_argument('--run-id', type=str, default='iwpb33bn',
                        help='WandB run ID')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        id=args.run_id,
        resume="allow"
    )
    
    # Find the latest checkpoint
    if args.iteration is None:
        # Look for the latest iteration
        import glob
        solver_dirs = glob.glob("./absolute_zero_solver_iter_*")
        proposer_dirs = glob.glob("./absolute_zero_proposer_iter_*")
        
        if solver_dirs:
            solver_iters = [int(d.split('_')[-1]) for d in solver_dirs]
            args.iteration = max(solver_iters)
            print(f"Found latest iteration: {args.iteration}")
        else:
            print("No checkpoints found!")
            return
    
    # Load models
    print(f"\nLoading models from iteration {args.iteration}...")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load solver
    solver_path = f"./absolute_zero_solver_iter_{args.iteration}"
    if os.path.exists(solver_path):
        print(f"Loading solver from {solver_path}")
        solver_model = AutoModelForCausalLM.from_pretrained(solver_path).to(device)
    else:
        print(f"Solver checkpoint not found at {solver_path}")
        solver_model = None
    
    # Load proposer
    proposer_path = f"./absolute_zero_proposer_iter_{args.iteration}"
    if os.path.exists(proposer_path):
        print(f"Loading proposer from {proposer_path}")
        proposer_model = AutoModelForCausalLM.from_pretrained(proposer_path).to(device)
    else:
        print(f"Proposer checkpoint not found at {proposer_path}")
        proposer_model = None
    
    # Generate samples
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if proposer_model:
        print("\nGenerating proposer samples...")
        proposer_samples = generate_proposer_samples(proposer_model, tokenizer, device)
        
        # Create WandB table
        proposer_table = wandb.Table(columns=[
            "prompt_template", "raw_generation", "parsed_problem", "valid"
        ])
        
        for sample in proposer_samples:
            proposer_table.add_data(
                sample['prompt_template'],
                sample['raw_generation'],
                sample['parsed_problem'],
                sample['valid']
            )
        
        wandb.log({f"proposer_samples_iter_{args.iteration}_{timestamp}": proposer_table})
        
        # Print summary
        valid_count = sum(1 for s in proposer_samples if s['valid'])
        print(f"Proposer: {valid_count}/{len(proposer_samples)} valid problems generated")
        print("Sample outputs:")
        for i, sample in enumerate(proposer_samples[:3]):
            print(f"  {i+1}. {sample['parsed_problem']} (Valid: {sample['valid']})")
    
    if solver_model:
        print("\nGenerating solver samples...")
        solver_samples = generate_solver_samples(solver_model, tokenizer, device)
        
        # Create WandB table
        solver_table = wandb.Table(columns=[
            "problem", "predicted_answer", "correct_answer", "is_correct", "raw_output"
        ])
        
        for sample in solver_samples:
            solver_table.add_data(
                sample['problem'],
                sample['predicted_answer'],
                sample['correct_answer'],
                sample['is_correct'],
                sample['raw_output']
            )
        
        wandb.log({f"solver_samples_iter_{args.iteration}_{timestamp}": solver_table})
        
        # Print summary
        correct_count = sum(1 for s in solver_samples if s['is_correct'])
        accuracy = correct_count / len(solver_samples)
        print(f"Solver: {correct_count}/{len(solver_samples)} correct ({accuracy:.1%} accuracy)")
        print("Sample outputs:")
        for i, sample in enumerate(solver_samples[:5]):
            status = "✓" if sample['is_correct'] else "✗"
            print(f"  {i+1}. {sample['problem']}{sample['predicted_answer']} {status} (correct: {sample['correct_answer']})")
    
    print(f"\nSamples logged to WandB run: {args.run_id}")
    print(f"View at: https://wandb.ai/wild-ai/pippa/runs/{args.run_id}")
    
    wandb.finish()


if __name__ == "__main__":
    main()