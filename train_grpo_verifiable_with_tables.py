#!/usr/bin/env python3
"""GRPO training with verifiable rewards and WandB Table logging."""

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

class LoggingGRPOTrainer(GRPOTrainer):
    """Custom GRPO trainer that logs generation samples to WandB Tables."""
    
    def __init__(self, expected_answers, extract_answer_fn, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_answers = expected_answers
        self.extract_answer_fn = extract_answer_fn
        self.task_name = task_name
        self.generation_table = wandb.Table(columns=["epoch", "prompt", "completion", "extracted_answer", "expected", "reward", "is_correct"])
        self.samples_logged = 0
        
    def log(self, logs: Dict[str, float]) -> None:
        """Override log method to add generation samples."""
        super().log(logs)
        
        # Log generation samples every 10 steps
        if self.state.global_step % 10 == 0 and hasattr(self, '_current_batch_data'):
            batch_data = self._current_batch_data
            
            # Sample up to 5 generations to log
            num_samples = min(5, len(batch_data['prompts']))
            
            for i in range(num_samples):
                prompt = batch_data['prompts'][i]
                completion = batch_data['completions'][i]
                reward = batch_data['rewards'][i]
                expected = self.expected_answers.get(prompt, "")
                extracted = self.extract_answer_fn(completion, prompt)
                is_correct = extracted == expected
                
                self.generation_table.add_data(
                    self.state.epoch,
                    prompt,
                    completion,
                    extracted,
                    expected,
                    reward,
                    is_correct
                )
                self.samples_logged += 1
            
            # Log table every 50 samples
            if self.samples_logged >= 50:
                wandb.log({f"generation_samples/{self.task_name}": self.generation_table})
                self.samples_logged = 0
                # Reset table for next batch
                self.generation_table = wandb.Table(columns=["epoch", "prompt", "completion", "extracted_answer", "expected", "reward", "is_correct"])
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override to capture batch data for logging."""
        # Store current batch data
        if 'prompt' in inputs:
            self._current_batch_data = {
                'prompts': inputs['prompt'],
                'completions': self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True),
                'rewards': inputs.get('rewards', [])
            }
        
        return super().compute_loss(model, inputs, return_outputs)

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
        name=f"grpo_verifiable_{args.task}_b{args.batch_size}_g{args.num_generations}_tables",
        config=vars(args),
        tags=["grpo-setup", "verifiable-rewards", args.task, "with-tables"]
    )
    
    # Create dataset and reward function
    train_dataset, reward_fn, expected_answers = create_dataset_for_task(args.task, args.n_samples)
    
    # Extract answer function for the task
    def extract_answer_fn(completion: str, prompt: str) -> str:
        answer = completion[len(prompt):].strip()
        if args.task in ["arithmetic", "counting", "binary"]:
            match = re.search(r'^[\d]+', answer)
            if match:
                return match.group()
        elif args.task == "comparison":
            answer_lower = answer.lower()
            if answer_lower.startswith("yes"):
                return "yes"
            elif answer_lower.startswith("no"):
                return "no"
        return answer
    
    # Log dataset info
    print(f"\n{'='*60}")
    print(f"VERIFIABLE GRPO TRAINING WITH WANDB TABLES")
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
        log_completions=True,  # Enable completion logging
        # Optimization
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )
    
    # Create custom trainer with table logging
    trainer = LoggingGRPOTrainer(
        expected_answers=expected_answers,
        extract_answer_fn=extract_answer_fn,
        task_name=args.task,
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
    
    # Log final generation table
    if trainer.samples_logged > 0:
        wandb.log({f"generation_samples/{args.task}_final": trainer.generation_table})
    
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
    
    # Test on sample problems and log to final table
    test_samples = 20
    correct = 0
    final_eval_table = wandb.Table(columns=["prompt", "expected", "generated", "extracted", "is_correct"])
    
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
            answer = extract_answer_fn(completion, prompt)
            
            is_correct = answer == expected
            if is_correct:
                correct += 1
                
            final_eval_table.add_data(prompt, expected, completion, answer, is_correct)
            
            print(f"[{i}] Prompt: {prompt}")
            print(f"     Expected: {expected}")
            print(f"     Generated: {completion}")
            print(f"     Extracted: {answer} {'✓' if is_correct else '✗'}")
            print()
    
    accuracy = 100 * correct / test_samples
    print(f"{'='*60}")
    print(f"Final Accuracy: {correct}/{test_samples} ({accuracy:.1f}%)")
    print(f"{'='*60}")
    
    wandb.log({
        "final_accuracy": accuracy,
        "final_evaluation": final_eval_table
    })
    
    wandb.finish()
    
    print("Done!")

if __name__ == "__main__":
    main()