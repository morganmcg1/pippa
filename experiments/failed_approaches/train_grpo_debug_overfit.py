#!/usr/bin/env python3
"""Debug version of GRPO training with extensive logging."""

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

# Load environment variables
load_dotenv()

def create_echo_dataset(n_samples: int = 50) -> List[Dict[str, str]]:
    """Task 1: Simple echo - model should repeat the word."""
    prompts = []
    words = ["hello", "world", "cat", "dog", "sun", "moon", "happy", "sad", "big", "small", 
             "yes", "no", "good", "bad", "hot", "cold", "up", "down", "left", "right"]
    
    for i in range(n_samples):
        word = words[i % len(words)]
        # Try instruction format that might work better
        prompts.append({
            "prompt": f"Instruction: Say the word '{word}'.\nResponse:",
            "expected": word
        })
    return prompts

def create_pattern_dataset(n_samples: int = 50) -> List[Dict[str, str]]:
    """Task 2: Pattern completion - complete simple sequences."""
    patterns = [
        ("A B C", "D"),
        ("1 2 3", "4"),
        ("X Y", "Z"),
        ("cat dog", "mouse"),
        ("red blue", "green"),
        ("up down", "left"),
        ("yes no", "maybe"),
        ("hot cold", "warm"),
        ("big small", "medium"),
        ("fast slow", "quick"),
        ("in out", "around"),
        ("start stop", "pause"),
        ("open close", "lock"),
        ("day night", "dawn"),
        ("black white", "gray")
    ]
    
    prompts = []
    for i in range(n_samples):
        pattern, answer = patterns[i % len(patterns)]
        prompts.append({
            "prompt": f"Instruction: Complete the pattern: {pattern}\nResponse:",
            "expected": answer
        })
    return prompts

def create_dataset_for_task(task: str, n_samples: int = 50) -> Tuple[Dataset, callable, Dict]:
    """Create dataset and reward function for specified task."""
    
    if task == "echo":
        data = create_echo_dataset(n_samples)
    elif task == "pattern":
        data = create_pattern_dataset(n_samples)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create dataset
    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in data])
    
    # Create reward function
    expected_answers = {d["prompt"]: d["expected"] for d in data}
    
    # Track debug info
    debug_info = {
        "total_calls": 0,
        "samples_seen": []
    }
    
    def reward_function(completions, prompts=None, **kwargs):
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        debug_info["total_calls"] += 1
        
        rewards = []
        for i, completion in enumerate(completions):
            # Get the corresponding prompt
            prompt = prompts[i] if i < len(prompts) else ""
            # Extract just the answer part (after the prompt)
            answer = completion[len(prompt):].strip().lower()
            expected = expected_answers.get(prompt, "").lower()
            
            # Store debug info
            if debug_info["total_calls"] <= 5 and len(debug_info["samples_seen"]) < 20:
                debug_info["samples_seen"].append({
                    "prompt": prompt,
                    "completion": completion,
                    "answer_extracted": answer,
                    "expected": expected,
                    "full_response": completion
                })
            
            # Reward structure:
            # +2.0 for exact match
            # +1.0 for starting with expected
            # -0.5 for wrong answer
            if answer == expected:
                rewards.append(2.0)
            elif answer.startswith(expected):
                rewards.append(1.0)
            else:
                rewards.append(-0.5)
                
        # Print debug info periodically
        if debug_info["total_calls"] % 10 == 1:
            print(f"\n{'='*60}")
            print(f"REWARD FUNCTION DEBUG - Call #{debug_info['total_calls']}")
            print(f"{'='*60}")
            print(f"Batch size: {len(completions)}")
            print(f"Sample outputs (showing first 3):")
            for j in range(min(3, len(completions))):
                prompt = prompts[j] if j < len(prompts) else ""
                completion = completions[j]
                answer = completion[len(prompt):].strip().lower()
                expected = expected_answers.get(prompt, "").lower()
                reward = rewards[j]
                print(f"\n  [{j}] Prompt: '{prompt}'")
                print(f"      Expected: '{expected}'")
                print(f"      Full completion: '{completion}'")
                print(f"      Answer extracted: '{answer}'")
                print(f"      Reward: {reward}")
            print(f"{'='*60}\n")
            
        return rewards
    
    return dataset, reward_function, expected_answers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="echo",
                        choices=["echo", "pattern"],
                        help="Task difficulty level")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=2048)  # 4x increase
    parser.add_argument("--num_generations", type=int, default=1024)  # 4x increase
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_completion_length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)  # No accumulation
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.2)
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Initialize wandb
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"grpo_debug_{args.task}_b{args.batch_size}_g{args.num_generations}_lr{args.lr}",
        config=vars(args)
    )
    
    # Create dataset and reward function
    train_dataset, reward_fn, expected_answers = create_dataset_for_task(args.task, args.n_samples)
    
    # Log dataset info
    print(f"\n{'='*60}")
    print(f"DATASET INFORMATION")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num generations: {args.num_generations}")
    print(f"Learning rate: {args.lr}")
    print(f"Temperature: {args.temperature}")
    print(f"Max completion length: {args.max_completion_length}")
    print(f"\nALL TRAINING PROMPTS AND EXPECTED ANSWERS:")
    for i in range(len(train_dataset)):
        prompt = train_dataset[i]['prompt']
        expected = expected_answers[prompt]
        print(f"  [{i:2d}] '{prompt}' → '{expected}'")
    print(f"{'='*60}\n")
    
    # Create GRPO config
    config = GRPOConfig(
        output_dir=f"./grpo_debug_{args.task}_{args.seed}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        max_prompt_length=64,
        beta=0.0,  # Dr GRPO - no KL penalty
        loss_type="dr_grpo",
        num_iterations=args.num_iterations,
        epsilon=args.epsilon,
        push_to_hub=False,
        report_to=["wandb"],
        logging_steps=1,
        save_steps=50,
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        log_completions=False,
        mask_truncated_completions=True,
        # Additional optimization settings
        warmup_steps=20,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=args.model_name,
        args=config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn]
    )
    
    # Log GPU usage before training
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name()}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    trainer.train()
    
    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    
    # Test final performance
    print(f"\n{'='*60}")
    print("FINAL TEST ON TRAINING PROMPTS")
    print(f"{'='*60}")
    
    # Load model and tokenizer for testing
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = min(20, len(train_dataset))
        for i in range(total):
            prompt = train_dataset[i]['prompt']
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate multiple times to see variation
            print(f"\n[{i}] Prompt: '{prompt}'")
            print(f"    Expected: '{expected_answers[prompt]}'")
            print(f"    Generated outputs:")
            
            got_correct = False
            for j in range(3):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_completion_length,
                    temperature=0.1 if j == 0 else 0.5,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = response[len(prompt):].strip()
                
                is_correct = answer.lower() == expected_answers[prompt].lower()
                if is_correct:
                    got_correct = True
                
                print(f"      [{j}] Full: '{response}' | Answer: '{answer}' {'✓' if is_correct else '✗'}")
            
            if got_correct:
                correct += 1
        
        print(f"\n{'='*60}")
        print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"{'='*60}")
    
    wandb.finish()
    print("Done!")

if __name__ == "__main__":
    main()