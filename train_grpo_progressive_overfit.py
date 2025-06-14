#!/usr/bin/env python3
"""Progressive overfitting experiments to verify GRPO training pipeline."""

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import wandb
import argparse
from typing import List, Tuple, Dict
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_echo_dataset(n_samples: int = 20) -> List[Dict[str, str]]:
    """Task 1: Simple echo - model should repeat the word."""
    prompts = []
    words = ["hello", "world", "cat", "dog", "sun", "moon", "happy", "sad", "big", "small"]
    
    for i in range(n_samples):
        word = words[i % len(words)]
        prompts.append({
            "prompt": f"Say {word}",
            "expected": word
        })
    return prompts

def create_pattern_dataset(n_samples: int = 20) -> List[Dict[str, str]]:
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
        ("fast slow", "quick")
    ]
    
    prompts = []
    for i in range(n_samples):
        pattern, answer = patterns[i % len(patterns)]
        prompts.append({
            "prompt": f"Complete the pattern: {pattern}",
            "expected": answer
        })
    return prompts

def create_simple_math_dataset(n_samples: int = 20) -> List[Dict[str, str]]:
    """Task 3: Simple single-digit addition."""
    prompts = []
    problems = [
        ("2 + 2", "4"),
        ("1 + 1", "2"),
        ("3 + 3", "6"),
        ("4 + 1", "5"),
        ("2 + 3", "5"),
        ("5 + 2", "7"),
        ("1 + 4", "5"),
        ("3 + 4", "7"),
        ("6 + 1", "7"),
        ("2 + 5", "7")
    ]
    
    for i in range(n_samples):
        problem, answer = problems[i % len(problems)]
        prompts.append({
            "prompt": f"Calculate: {problem} =",
            "expected": answer
        })
    return prompts

def create_word_problem_dataset(n_samples: int = 20) -> List[Dict[str, str]]:
    """Task 4: Simple word problems."""
    problems = [
        ("If you have 2 apples and get 2 more, how many apples do you have?", "4"),
        ("There are 3 cats. 2 more cats come. How many cats are there?", "5"),
        ("You have 5 cookies and eat 2. How many cookies are left?", "3"),
        ("A box has 4 toys. You add 1 more toy. How many toys in the box?", "5"),
        ("There are 6 birds. 3 fly away. How many birds remain?", "3")
    ]
    
    prompts = []
    for i in range(n_samples):
        problem, answer = problems[i % len(problems)]
        prompts.append({
            "prompt": problem,
            "expected": answer
        })
    return prompts

def create_dataset_for_task(task: str, n_samples: int = 20) -> Tuple[Dataset, callable]:
    """Create dataset and reward function for specified task."""
    
    if task == "echo":
        data = create_echo_dataset(n_samples)
    elif task == "pattern":
        data = create_pattern_dataset(n_samples)
    elif task == "math":
        data = create_simple_math_dataset(n_samples)
    elif task == "word_problem":
        data = create_word_problem_dataset(n_samples)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create dataset
    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in data])
    
    # Create reward function
    expected_answers = {d["prompt"]: d["expected"] for d in data}
    
    def reward_function(completions, prompts=None, **kwargs):
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        rewards = []
        for i, completion in enumerate(completions):
            # Get the corresponding prompt
            prompt = prompts[i] if i < len(prompts) else ""
            # Extract just the answer part (after the prompt)
            answer = completion[len(prompt):].strip().lower()
            expected = expected_answers.get(prompt, "").lower()
            
            # Check if answer starts with expected (allowing for extra tokens)
            if answer.startswith(expected):
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        return rewards
    
    return dataset, reward_function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="echo",
                        choices=["echo", "pattern", "math", "word_problem"],
                        help="Task difficulty level")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_completion_length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    
    # Initialize wandb
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"grpo_overfit_{args.task}_e{args.epochs}_b{args.batch_size}_g{args.num_generations}",
        config=vars(args)
    )
    
    # Create dataset and reward function
    train_dataset, reward_fn = create_dataset_for_task(args.task, args.n_samples)
    
    # Log sample data
    print(f"\nTask: {args.task}")
    print(f"Sample prompts:")
    for i in range(min(3, len(train_dataset))):
        print(f"  {i+1}. {train_dataset[i]['prompt']}")
    
    # Create GRPO config
    config = GRPOConfig(
        output_dir=f"./grpo_overfit_{args.task}_{args.seed}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        max_prompt_length=128,
        beta=0.0,  # Dr GRPO
        loss_type="dr_grpo",
        num_iterations=1,
        push_to_hub=False,
        report_to=["wandb"],
        logging_steps=1,
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        log_completions=True,
        mask_truncated_completions=True,
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=args.model_name,
        args=config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn]  # Use list format like original
    )
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    trainer.train()
    
    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    
    # Test final performance
    print("\nFinal test on training prompts:")
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
        for i in range(min(5, len(train_dataset))):
            prompt = train_dataset[i]['prompt']
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_completion_length,
                temperature=0.1,  # Low temp for testing
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  Prompt: {prompt}")
            print(f"  Response: {response}")
            print()
    
    wandb.finish()
    print("Done!")

if __name__ == "__main__":
    main()