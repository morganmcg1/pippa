#!/usr/bin/env python3
"""
Re-evaluate completed GRPO models on the standardized arithmetic evaluation dataset.
This script:
1. Finds the best completed runs with model artifacts
2. Downloads and evaluates each model
3. Resumes the original WandB run
4. Logs the new "arithmetic_eval" metric
5. Finishes the run

This ensures all models are compared fairly on the same test set.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
import re
from dotenv import load_dotenv
import os
from typing import Dict, Any, List

load_dotenv()

def extract_answer(text: str, prompt: str) -> str:
    """Extract the answer from model output"""
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
    print("\nEvaluating on standardized dataset (morgan/arithmetic_eval)...")
    
    # Load standardized evaluation dataset
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    print(f"Loaded {len(eval_dataset)} evaluation samples")
    
    # Initialize results tracking
    results_by_difficulty = {}
    results_by_operation = {}
    correct_total = 0
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(eval_dataset):
            if i % 50 == 0 and i > 0:
                print(f"Progress: {i}/{len(eval_dataset)}")
            
            prompt = sample['prompt']
            expected = sample['answer']
            difficulty = sample['difficulty']
            operation = sample['metadata']['operation'] if sample['metadata'] else None
            
            # Generate model answer
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer
            predicted = extract_answer(response, prompt)
            
            # Check correctness
            is_correct = predicted == expected
            if is_correct:
                correct_total += 1
            
            # Track by difficulty
            if difficulty not in results_by_difficulty:
                results_by_difficulty[difficulty] = {'correct': 0, 'total': 0}
            results_by_difficulty[difficulty]['total'] += 1
            if is_correct:
                results_by_difficulty[difficulty]['correct'] += 1
            
            # Track by operation
            if operation:
                if operation not in results_by_operation:
                    results_by_operation[operation] = {'correct': 0, 'total': 0}
                results_by_operation[operation]['total'] += 1
                if is_correct:
                    results_by_operation[operation]['correct'] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct_total / len(eval_dataset)
    
    # Display results
    print(f"\nOverall accuracy: {overall_accuracy:.2%} ({correct_total}/{len(eval_dataset)})")
    
    print("\nAccuracy by difficulty:")
    for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        if diff in results_by_difficulty:
            stats = results_by_difficulty[diff]
            acc = stats['correct'] / stats['total']
            print(f"  {diff}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nAccuracy by operation:")
    for op in ['+', '-', '*', '/']:
        if op in results_by_operation:
            stats = results_by_operation[op]
            acc = stats['correct'] / stats['total']
            print(f"  {op}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    return overall_accuracy

def reevaluate_run(run_id: str, run_name: str, model_artifact_name: str = None):
    """Re-evaluate a specific run and update its metrics."""
    print(f"\n{'='*60}")
    print(f"Processing run: {run_name} (ID: {run_id})")
    print(f"{'='*60}")
    
    # Resume the original run
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        id=run_id,
        resume="must"
    )
    
    try:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        if model_artifact_name:
            # Download model from artifact
            print(f"Loading model from artifact: {model_artifact_name}")
            artifact = wandb.use_artifact(model_artifact_name, type='model')
            artifact_dir = artifact.download()
            model_path = artifact_dir
        else:
            # Try to use the base model name from config
            model_path = wandb.config.get("model", "Qwen/Qwen2-0.5B-Instruct")
            print(f"Using model: {model_path}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Evaluate on standard dataset
        arithmetic_eval_score = evaluate_on_standard_dataset(model, tokenizer, device)
        
        # Log as summary metric
        wandb.run.summary["arithmetic_eval"] = arithmetic_eval_score
        wandb.run.summary["arithmetic_eval_note"] = "Standardized evaluation on morgan/arithmetic_eval dataset"
        
        # Also log as regular metric
        wandb.log({
            "arithmetic_eval": arithmetic_eval_score,
            "evaluation_dataset": "morgan/arithmetic_eval",
            "evaluation_samples": 200
        })
        
        print(f"\n✅ Successfully updated run with arithmetic_eval: {arithmetic_eval_score:.2%}")
        
    except Exception as e:
        print(f"❌ Error processing run {run_name}: {e}")
    finally:
        wandb.finish()

def main():
    print("Re-evaluating GRPO Models with Standardized Dataset")
    print("="*60)
    
    # Top runs to re-evaluate (based on previous reported performance)
    runs_to_evaluate = [
        # Format: (run_id, display_name, artifact_name_if_known)
        ("s8snd2a0", "Mixed Small Numbers (60.7%)", "wild-ai/pippa/model-s8snd2a0:latest"),
        ("afj0flv3", "Mixed Dataset Full Diversity (54.7%)", "wild-ai/pippa/model-afj0flv3:latest"),
        ("ipnl8nmm", "Smaller Numbers Full Diversity (45%)", "wild-ai/pippa/model-ipnl8nmm:latest"),
        ("icel5tvz", "Ultra Diversity 64gen (50%)", "wild-ai/pippa/model-icel5tvz:latest"),
        ("js57wrfi", "Extended Training", "wild-ai/pippa/model-js57wrfi:latest"),
        ("2ng01uci", "Expanded Dataset 500", "wild-ai/pippa/model-2ng01uci:latest"),
        ("pm8cy3ri", "Higher LR", "wild-ai/pippa/model-pm8cy3ri:latest"),
        ("kfobm3if", "Higher Beta", "wild-ai/pippa/model-kfobm3if:latest"),
        ("au3eohjq", "Long 100 epochs", "wild-ai/pippa/model-au3eohjq:latest"),
        ("3jjl84p4", "Fixed Completions", "wild-ai/pippa/model-3jjl84p4:latest"),
    ]
    
    print(f"\nFound {len(runs_to_evaluate)} runs to re-evaluate")
    
    # Process each run
    summary_results = []
    for run_id, display_name, artifact_name in runs_to_evaluate:
        try:
            reevaluate_run(run_id, display_name, artifact_name)
            # Note: We'll collect results after the run is updated in WandB
        except Exception as e:
            print(f"Failed to process {display_name}: {e}")
    
    print("\n" + "="*60)
    print("Re-evaluation complete!")
    print("Check WandB for updated metrics: https://wandb.ai/wild-ai/pippa")
    print("\nKey metric: 'arithmetic_eval' - standardized evaluation on 200 problems")
    print("Base model baseline: ~30%")

if __name__ == "__main__":
    main()