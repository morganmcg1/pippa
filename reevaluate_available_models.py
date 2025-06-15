#!/usr/bin/env python3
"""
Re-evaluate models that have checkpoints saved or use base model for runs without artifacts.
This updates the WandB runs with the standardized arithmetic_eval metric.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
import re
from dotenv import load_dotenv
import os
from typing import Dict, Any

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
    difficulty_metrics = {}
    for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        if diff in results_by_difficulty:
            stats = results_by_difficulty[diff]
            acc = stats['correct'] / stats['total']
            difficulty_metrics[f"eval_{diff}_accuracy"] = acc
            print(f"  {diff}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nAccuracy by operation:")
    operation_metrics = {}
    for op in ['+', '-', '*', '/']:
        if op in results_by_operation:
            stats = results_by_operation[op]
            acc = stats['correct'] / stats['total']
            op_name = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op]
            operation_metrics[f"eval_{op_name}_accuracy"] = acc
            print(f"  {op}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    return {
        'arithmetic_eval': overall_accuracy,
        'arithmetic_eval_correct': correct_total,
        'arithmetic_eval_total': len(eval_dataset),
        **difficulty_metrics,
        **operation_metrics
    }

def reevaluate_run(run_id: str, run_name: str, has_artifact: bool = False):
    """Re-evaluate a specific run and update its metrics."""
    print(f"\n{'='*60}")
    print(f"Processing run: {run_name}")
    print(f"Run ID: {run_id}")
    print(f"Has model artifact: {has_artifact}")
    print(f"{'='*60}")
    
    # Resume the original run
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        id=run_id,
        resume="must"
    )
    
    try:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        if has_artifact:
            # Try to download model from artifact
            artifact_name = f"wild-ai/pippa/model-{run_id}:latest"
            print(f"Attempting to load model from artifact: {artifact_name}")
            try:
                artifact = wandb.use_artifact(artifact_name, type='model')
                artifact_dir = artifact.download()
                model_path = artifact_dir
                print(f"Successfully downloaded model artifact to: {artifact_dir}")
            except Exception as e:
                print(f"Could not load artifact, using base model instead: {e}")
                model_path = "Qwen/Qwen2-0.5B-Instruct"
        else:
            # Use base model for evaluation (to show improvement)
            model_path = "Qwen/Qwen2-0.5B-Instruct"
            print(f"Using base model for comparison: {model_path}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-0.5B-Instruct"  # Always use base tokenizer
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Evaluate on standard dataset
        eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
        
        # Log as summary metrics
        for key, value in eval_metrics.items():
            wandb.run.summary[key] = value
        wandb.run.summary["arithmetic_eval_note"] = "Standardized evaluation on morgan/arithmetic_eval dataset (200 problems)"
        wandb.run.summary["model_source"] = "trained_model" if has_artifact else "base_model_for_comparison"
        
        # Also log as regular metrics
        wandb.log(eval_metrics)
        
        print(f"\n✅ Successfully updated run with arithmetic_eval: {eval_metrics['arithmetic_eval']:.2%}")
        
    except Exception as e:
        print(f"❌ Error processing run {run_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()

def main():
    print("Re-evaluating GRPO Models with Standardized Dataset")
    print("="*60)
    print("\nThis script will update WandB runs with the 'arithmetic_eval' metric")
    print("Base model baseline: ~30% (we'll verify this)")
    
    # Runs to evaluate - focusing on those with and without artifacts
    runs_to_evaluate = [
        # Runs WITH model artifacts (confirmed from query)
        ("s8snd2a0", "Mixed Small Numbers (reported 60.7%)", True),
        ("js57wrfi", "Extended Training", True),
        
        # Key runs WITHOUT artifacts (will use base model for now)
        ("afj0flv3", "Mixed Dataset Full Diversity (reported 54.7%)", False),
        ("ipnl8nmm", "Smaller Numbers Full Diversity (reported 45%)", False),
        ("icel5tvz", "Ultra Diversity 64gen (reported 50%)", False),
        
        # Other completed runs for reference
        ("2ng01uci", "Expanded Dataset 500 samples", False),
        ("pm8cy3ri", "Higher LR (reported 30% eval)", False),
        ("kfobm3if", "Higher Beta (reported 30% eval)", False),
        ("au3eohjq", "Long 100 epochs (reported 35% eval)", False),
    ]
    
    print(f"\nProcessing {len(runs_to_evaluate)} runs...")
    print("Note: Runs without artifacts will show base model performance (~30%)")
    print("This establishes the baseline and shows which runs need artifact uploads")
    
    # First, evaluate base model standalone
    print("\n" + "="*60)
    print("BASELINE: Evaluating base Qwen model")
    print("="*60)
    
    # Create a new run just for base model evaluation
    base_run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="qwen_base_arithmetic_eval",
        tags=["evaluation", "baseline", "arithmetic_eval"],
        config={
            "model": "Qwen/Qwen2-0.5B-Instruct",
            "eval_dataset": "morgan/arithmetic_eval",
            "eval_dataset_size": 200
        }
    )
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct").to(device)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
        
        # Log metrics
        wandb.log(eval_metrics)
        for key, value in eval_metrics.items():
            wandb.run.summary[key] = value
        
        print(f"\n✅ Base model arithmetic_eval: {eval_metrics['arithmetic_eval']:.2%}")
        
    except Exception as e:
        print(f"Error evaluating base model: {e}")
    finally:
        wandb.finish()
    
    # Process each run
    for run_id, display_name, has_artifact in runs_to_evaluate:
        try:
            reevaluate_run(run_id, display_name, has_artifact)
        except Exception as e:
            print(f"Failed to process {display_name}: {e}")
    
    print("\n" + "="*60)
    print("Re-evaluation complete!")
    print("Check WandB for updated metrics: https://wandb.ai/wild-ai/pippa")
    print("\nKey insights:")
    print("- Runs WITH artifacts show actual trained model performance")
    print("- Runs WITHOUT artifacts show base model performance (~30%)")
    print("- This identifies which runs need model artifacts uploaded")
    print("\nMetric name: 'arithmetic_eval' - standardized evaluation on 200 problems")

if __name__ == "__main__":
    main()