#!/usr/bin/env python3
"""GRPO training with verifiable reward tasks."""

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import wandb
import argparse
from typing import List, Tuple, Dict
import time
import os
from dotenv import load_dotenv
import random
import re

def evaluate_on_standard_dataset(model, tokenizer, device) -> Dict[str, float]:
    """
    Evaluate model on the standardized arithmetic evaluation dataset.
    This ensures fair comparison across all experiments.
    """
    print("\n" + "="*60)
    print("STANDARDIZED EVALUATION (morgan/arithmetic_eval)")
    print("="*60)
    
    # Load standardized evaluation dataset
    from datasets import load_dataset
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
            
            # Extract answer (reuse existing extract_answer function if available)
            completion = response[len(prompt):].strip()
            match = re.match(r'^-?\d+', completion)
            predicted = match.group(0) if match else completion.split()[0] if completion else ""
            
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
    difficulty_accs = {}
    for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        if diff in results_by_difficulty:
            stats = results_by_difficulty[diff]
            acc = stats['correct'] / stats['total']
            difficulty_accs[f"eval_{diff}_accuracy"] = acc
            print(f"  {diff}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("\nAccuracy by operation:")
    operation_accs = {}
    for op in ['+', '-', '*', '/']:
        if op in results_by_operation:
            stats = results_by_operation[op]
            acc = stats['correct'] / stats['total']
            op_name = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op]
            operation_accs[f"eval_{op_name}_accuracy"] = acc
            print(f"  {op}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("="*60)
    
    # Return all metrics for WandB logging
    return {
        'arithmetic_eval': overall_accuracy,  # Primary metric
        'arithmetic_eval_correct': correct_total,
        'arithmetic_eval_total': len(eval_dataset),
        **difficulty_accs,
        **operation_accs
    }


# Load environment variables
load_dotenv()

def create_arithmetic_dataset(n_samples: int = 100) -> List[Dict[str, str]]:
    """Simple arithmetic problems with verifiable answers."""
    prompts = []
    for _ in range(n_samples):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        op = random.choice(['+', '-', '*'])
        
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else:  # multiplication
            answer = a * b
            
        prompts.append({
            "prompt": f"Calculate: {a} {op} {b} = ",
            "expected": str(answer)
        })
    return prompts

def create_counting_dataset(n_samples: int = 100) -> List[Dict[str, str]]:
    """Count words/letters in strings - verifiable."""
    prompts = []
    sentences = [
        "the cat", "a big dog", "hello world", "python code", "machine learning",
        "red car", "blue sky", "green tree", "fast runner", "slow walker",
        "happy day", "sad night", "warm sun", "cold moon", "bright star"
    ]
    
    for i in range(n_samples):
        sentence = sentences[i % len(sentences)]
        task_type = random.choice(["words", "letters"])
        
        if task_type == "words":
            answer = len(sentence.split())
            prompt = f"How many words in '{sentence}'? Answer with just the number: "
        else:
            answer = len([c for c in sentence if c.isalpha()])
            prompt = f"How many letters in '{sentence}'? Answer with just the number: "
            
        prompts.append({
            "prompt": prompt,
            "expected": str(answer)
        })
    return prompts

def create_comparison_dataset(n_samples: int = 100) -> List[Dict[str, str]]:
    """Simple comparisons with yes/no answers."""
    prompts = []
    for _ in range(n_samples):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        op = random.choice(['>', '<', '=='])
        
        if op == '>':
            answer = "yes" if a > b else "no"
            prompt = f"Is {a} greater than {b}? Answer yes or no: "
        elif op == '<':
            answer = "yes" if a < b else "no"
            prompt = f"Is {a} less than {b}? Answer yes or no: "
        else:
            answer = "yes" if a == b else "no"
            prompt = f"Is {a} equal to {b}? Answer yes or no: "
            
        prompts.append({
            "prompt": prompt,
            "expected": answer
        })
    return prompts

def create_binary_conversion_dataset(n_samples: int = 100) -> List[Dict[str, str]]:
    """Convert small numbers to binary - verifiable."""
    prompts = []
    for i in range(n_samples):
        num = random.randint(0, 15)  # Keep small for simplicity
        answer = bin(num)[2:]  # Remove '0b' prefix
        
        prompts.append({
            "prompt": f"Convert {num} to binary: ",
            "expected": answer
        })
    return prompts

def create_dataset_for_task(task: str, n_samples: int = 100) -> Tuple[Dataset, callable, Dict]:
    """Create dataset and reward function for specified task."""
    
    if task == "arithmetic":
        data = create_arithmetic_dataset(n_samples)
    elif task == "counting":
        data = create_counting_dataset(n_samples)
    elif task == "comparison":
        data = create_comparison_dataset(n_samples)
    elif task == "binary":
        data = create_binary_conversion_dataset(n_samples)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create dataset
    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in data])
    
    # Create reward function
    expected_answers = {d["prompt"]: d["expected"] for d in data}
    
    def extract_answer(completion: str, prompt: str) -> str:
        """Extract the answer from completion."""
        # Remove the prompt from completion
        answer = completion[len(prompt):].strip()
        
        # For arithmetic, counting, and binary - extract first number/word
        if task in ["arithmetic", "counting", "binary"]:
            # Match first number or alphanumeric sequence
            match = re.search(r'^[\d]+', answer)
            if match:
                return match.group()
        elif task == "comparison":
            # Look for yes/no
            answer_lower = answer.lower()
            if answer_lower.startswith("yes"):
                return "yes"
            elif answer_lower.startswith("no"):
                return "no"
                
        return answer
    
    def reward_function(completions, prompts=None, **kwargs):
        """Verifiable reward function."""
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            answer = extract_answer(completion, prompt)
            expected = expected_answers.get(prompt, "")
            
            # Binary reward: correct (1.0) or incorrect (-1.0)
            if answer == expected:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
                
        return rewards
    
    return dataset, reward_function, expected_answers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="arithmetic",
                        choices=["arithmetic", "counting", "comparison", "binary"],
                        help="Task type with verifiable rewards")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_completion_length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)  # KL penalty
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Initialize wandb
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"grpo_verifiable_{args.task}_b{args.batch_size}_g{args.num_generations}",
        config=vars(args),
        tags=["grpo-setup", "verifiable-rewards", args.task]
    )
    
    # Create dataset and reward function
    train_dataset, reward_fn, expected_answers = create_dataset_for_task(args.task, args.n_samples)
    
    # Log dataset info
    print(f"\n{'='*60}")
    print(f"VERIFIABLE GRPO TRAINING")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Model: {args.model_name}")
    print(f"\nSample prompts and answers:")
    for i in range(min(10, len(train_dataset))):
        prompt = train_dataset[i]['prompt']
        expected = expected_answers[prompt]
        print(f"  [{i}] {prompt} → {expected}")
    print(f"{'='*60}\n")
    
    # Create GRPO config
    config = GRPOConfig(
        output_dir=f"./grpo_verifiable_{args.task}_{args.seed}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        max_prompt_length=128,
        beta=args.beta,  # KL penalty for stability
        loss_type="grpo",  # Standard GRPO
        push_to_hub=False,
        report_to=["wandb"],
        logging_steps=1,
        save_steps=100,
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        log_completions=False,
        wandb_log_unique_prompts=True,  # Always log unique prompts for better visibility
        # Optimization
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=args.model_name,
        args=config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn]
    )
    
    # Train
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    trainer.train()
    
    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    
    # Test final performance
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    # Load model for testing
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Test on sample problems
    test_samples = 20
    correct = 0
    
    with torch.no_grad():
        for i in range(min(test_samples, len(train_dataset))):
            prompt = train_dataset[i]['prompt']
            expected = expected_answers[prompt]
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_completion_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = extract_answer(completion, prompt)
            
            is_correct = answer == expected
            if is_correct:
                correct += 1
                
            print(f"[{i}] Prompt: {prompt}")
            print(f"     Expected: {expected}")
            print(f"     Generated: {completion}")
            print(f"     Extracted: {answer} {'✓' if is_correct else '✗'}")
            print()
    
    accuracy = 100 * correct / test_samples
    print(f"{'='*60}")
    print(f"Final Accuracy: {correct}/{test_samples} ({accuracy:.1f}%)")
    print(f"{'='*60}")
    
    wandb.log({"final_accuracy": accuracy})
    

    # Evaluate on standardized dataset
    print("\nEvaluating on standardized arithmetic dataset...")
    standard_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
    
    # Log standardized metrics
    if 'wandb' in globals() and wandb.run is not None:
        wandb.log(standard_eval_metrics)
        wandb.log({
            "arithmetic_eval": standard_eval_metrics['arithmetic_eval']  # Primary metric
        })
    
    print(f"\n🎯 Standardized evaluation accuracy: {standard_eval_metrics['arithmetic_eval']:.2%}")

wandb.finish()
    
    print("Done!")

if __name__ == "__main__":
    main()