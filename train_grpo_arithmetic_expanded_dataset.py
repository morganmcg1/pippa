#!/usr/bin/env python3
"""GRPO arithmetic training with expanded dataset (500 samples)."""

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import wandb
import os
from dotenv import load_dotenv
import random
import re
from typing import Dict, Any

# Load environment variables
load_dotenv()

class GRPOTrainerFixed(GRPOTrainer):
    """GRPOTrainer with fixed completion logging."""
    
    def log(self, logs: Dict[str, Any], start_time: float = None) -> None:
        """Override log method to fix completion printing."""
        # Call parent log method but catch the error
        try:
            super().log(logs, start_time)
        except AttributeError as e:
            if "add_section" in str(e):
                # If it's the table error, print completions manually
                if self.args.log_completions and self.state.global_step % self.args.logging_steps == 0:
                    self._print_completions_simple()
                # Still log the metrics
                if hasattr(self, '_wandb'):
                    self._wandb.log(logs, step=self.state.global_step)
            else:
                raise
    
    def _print_completions_simple(self):
        """Simple completion printing without rich tables."""
        if not hasattr(self, '_recent_completions'):
            return
            
        print("\n" + "="*60)
        print(f"Step {self.state.global_step} - Sample Completions")
        print("="*60)
        
        # Show up to 3 samples
        num_samples = min(3, len(self._recent_completions))
        for i in range(num_samples):
            if i < len(self._recent_completions):
                sample = self._recent_completions[i]
                print(f"\nSample {i+1}:")
                print(f"Prompt: {sample.get('prompt', 'N/A')}")
                print(f"Completion: {sample.get('completion', 'N/A')}")
                print(f"Reward: {sample.get('reward', 'N/A')}")
        
        print("="*60 + "\n")
        
        # Clear stored completions
        self._recent_completions = []
    
    def _generate_completions(self, prompts, **generation_kwargs):
        """Override to capture completions for logging."""
        completions = super()._generate_completions(prompts, **generation_kwargs)
        
        # Store completions for our simple logging
        if not hasattr(self, '_recent_completions'):
            self._recent_completions = []
            
        # Decode and store a few samples
        if hasattr(self, 'tokenizer'):
            for i in range(min(3, len(completions))):
                if i < len(prompts):
                    completion_text = self.tokenizer.decode(completions[i], skip_special_tokens=True)
                    self._recent_completions.append({
                        "prompt": prompts[i],
                        "completion": completion_text,
                        "reward": None  # Will be updated later
                    })
        
        return completions

def create_expanded_arithmetic_dataset(n_samples: int = 500):
    """Expanded arithmetic dataset with more diversity."""
    prompts = []
    seen = set()  # Ensure uniqueness
    
    while len(prompts) < n_samples:
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        op = random.choice(['+', '-', '*'])
        
        # Create unique identifier
        problem_id = f"{a}{op}{b}"
        if problem_id in seen:
            continue
        seen.add(problem_id)
        
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

def main():
    # Configuration with expanded dataset
    n_samples = 500  # 5x larger dataset
    batch_size = 64
    num_generations = 16
    lr = 5e-6
    temperature = 0.7
    epochs = 100  # More epochs for larger dataset
    seed = 999
    beta = 0.1  # Standard KL penalty
    
    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Initialize wandb
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"grpo_arithmetic_expanded_{n_samples}_samples",
        config={
            "task": "arithmetic_expanded_dataset",
            "n_samples": n_samples,
            "batch_size": batch_size,
            "num_generations": num_generations,
            "learning_rate": lr,
            "temperature": temperature,
            "epochs": epochs,
            "seed": seed,
            "beta": beta,
            "model": "Qwen/Qwen2-0.5B-Instruct",
            "log_completions": True,
            "wandb_log_unique_prompts": True
        },
        tags=["grpo-setup", "overfit", "arithmetic", "expanded-dataset", "break-barrier"]
    )
    
    # Create dataset
    data = create_expanded_arithmetic_dataset(n_samples)
    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in data])
    expected_answers = {d["prompt"]: d["expected"] for d in data}
    
    # Print dataset info
    print(f"\n{'='*60}")
    print(f"ARITHMETIC GRPO WITH EXPANDED DATASET")
    print(f"{'='*60}")
    print(f"Dataset size: {len(dataset)} unique problems (5x expansion)")
    print(f"Batch size: {batch_size}")
    print(f"Generations per prompt: {num_generations}")
    print(f"Learning rate: {lr}")
    print(f"Temperature: {temperature}")
    print(f"Beta (KL penalty): {beta}")
    print(f"Epochs: {epochs}")
    print(f"\nSample problems:")
    for i in range(min(10, len(data))):
        print(f"  [{i}] {data[i]['prompt']} → {data[i]['expected']}")
    print(f"\nDataset statistics:")
    print(f"  Total unique problems: {len(data)}")
    print(f"  Addition problems: {sum(1 for d in data if '+' in d['prompt'])}")
    print(f"  Subtraction problems: {sum(1 for d in data if '-' in d['prompt'] and '*' not in d['prompt'])}")
    print(f"  Multiplication problems: {sum(1 for d in data if '*' in d['prompt'])}")
    print(f"{'='*60}\n")
    
    # Create reward function
    def reward_function(completions, prompts=None, **kwargs):
        """Binary reward for exact match."""
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            # Extract answer
            answer = completion[len(prompt):].strip()
            # Match first number (including negative)
            match = re.search(r'^-?\d+', answer)
            if match:
                extracted = match.group()
            else:
                extracted = answer.split()[0] if answer else ""
            
            expected = expected_answers.get(prompt, "")
            
            # Binary reward
            if extracted == expected:
                rewards.append(1.0)  # Correct
            else:
                rewards.append(-1.0)  # Incorrect
                
        return rewards
    
    # GRPO config
    config = GRPOConfig(
        output_dir=f"./grpo_arithmetic_expanded_{seed}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=16,
        max_prompt_length=128,
        beta=beta,
        loss_type="grpo",
        push_to_hub=False,
        report_to=["wandb"],
        logging_steps=1,  # Log every step
        save_steps=500,
        seed=seed,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        log_completions=True,
        wandb_log_unique_prompts=True,
        # Optimization parameters
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",
        scale_rewards=True,
    )
    
    # Create trainer with fixed logging
    trainer = GRPOTrainerFixed(
        model="Qwen/Qwen2-0.5B-Instruct",
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_function],
    )
    
    # Train
    print(f"Starting training with {n_samples} samples...")
    trainer.train()
    
    print("\nTraining completed!")
    
    # Test final performance on a separate test set
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = trainer.model
    tokenizer = trainer.tokenizer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")
    
    # Create a test set with different random seed
    random.seed(seed + 1000)
    test_data = create_expanded_arithmetic_dataset(50)  # 50 test problems
    
    correct = 0
    eval_table = wandb.Table(columns=["prompt", "expected", "generated", "extracted", "is_correct"])
    
    with torch.no_grad():
        for i in range(len(test_data)):
            prompt = test_data[i]['prompt']
            expected = test_data[i]['expected']
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = completion[len(prompt):].strip()
            match = re.search(r'^-?\d+', answer)
            extracted = match.group() if match else answer.split()[0] if answer else ""
            
            is_correct = extracted == expected
            if is_correct:
                correct += 1
            
            eval_table.add_data(prompt, expected, answer, extracted, is_correct)
                
            if i < 20:  # Print first 20
                print(f"[{i}] {prompt} → {expected}")
                print(f"     Generated: {answer}")
                print(f"     Extracted: {extracted} {'✓' if is_correct else '✗'}")
    
    accuracy = 100 * correct / len(test_data)
    print(f"\nFinal Test Accuracy: {correct}/{len(test_data)} ({accuracy:.1f}%)")
    print(f"{'='*60}")
    
    wandb.log({
        "final_test_accuracy": accuracy,
        "final_evaluation": eval_table
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()