#!/usr/bin/env python3
"""GRPO arithmetic training with partial credit reward function."""

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

def create_simple_arithmetic_dataset(n_samples: int = 100):
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
            "expected": str(answer),
            "a": a,
            "b": b,
            "op": op,
            "answer_int": answer
        })
    return prompts

def infer_operation(a: int, b: int, extracted: str) -> str:
    """Try to infer what operation the model performed."""
    try:
        result = int(extracted)
        if result == a + b:
            return "+"
        elif result == a - b:
            return "-"
        elif result == a * b:
            return "*"
        else:
            return "unknown"
    except:
        return "invalid"

def main():
    # Configuration with partial rewards
    n_samples = 100
    batch_size = 64
    num_generations = 16
    lr = 5e-6
    temperature = 0.7
    epochs = 100  # More epochs to leverage partial rewards
    seed = 888
    beta = 0.1  # Standard KL penalty
    
    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Initialize wandb
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"grpo_arithmetic_partial_rewards",
        config={
            "task": "arithmetic_partial_rewards",
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
            "wandb_log_unique_prompts": True,
            "reward_scheme": {
                "correct": 1.0,
                "off_by_1": 0.5,
                "off_by_2_5": 0.25,
                "wrong_operation": -0.5,
                "garbage": -1.0
            }
        },
        tags=["grpo-setup", "overfit", "arithmetic", "partial-rewards", "break-barrier"]
    )
    
    # Create dataset
    data = create_simple_arithmetic_dataset(n_samples)
    dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in data])
    problem_data = {d["prompt"]: d for d in data}
    
    # Print dataset info
    print(f"\n{'='*60}")
    print(f"ARITHMETIC GRPO WITH PARTIAL CREDIT REWARDS")
    print(f"{'='*60}")
    print(f"Dataset size: {len(dataset)} problems")
    print(f"Batch size: {batch_size}")
    print(f"Generations per prompt: {num_generations}")
    print(f"Learning rate: {lr}")
    print(f"Temperature: {temperature}")
    print(f"Beta (KL penalty): {beta}")
    print(f"Epochs: {epochs}")
    print(f"\nReward Scheme:")
    print(f"  Correct answer: +1.0")
    print(f"  Off by 1: +0.5")
    print(f"  Off by 2-5: +0.25")
    print(f"  Wrong operation: -0.5")
    print(f"  Garbage output: -1.0")
    print(f"\nSample problems:")
    for i in range(min(10, len(data))):
        print(f"  [{i}] {data[i]['prompt']} → {data[i]['expected']}")
    print(f"{'='*60}\n")
    
    # Create partial credit reward function
    def reward_function(completions, prompts=None, **kwargs):
        """Partial credit reward function."""
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            problem = problem_data.get(prompt, {})
            
            # Extract answer
            answer = completion[len(prompt):].strip()
            # Match first number (including negative)
            match = re.search(r'^-?\d+', answer)
            if match:
                extracted = match.group()
            else:
                extracted = answer.split()[0] if answer else ""
            
            expected = problem.get("expected", "")
            expected_int = problem.get("answer_int", 0)
            
            # Determine reward based on answer quality
            if extracted == expected:
                # Exact match
                rewards.append(1.0)
            else:
                try:
                    extracted_int = int(extracted)
                    diff = abs(extracted_int - expected_int)
                    
                    if diff == 1:
                        # Off by 1 - close!
                        rewards.append(0.5)
                    elif 2 <= diff <= 5:
                        # Off by 2-5 - somewhat close
                        rewards.append(0.25)
                    else:
                        # Check if wrong operation
                        a = problem.get("a", 0)
                        b = problem.get("b", 0)
                        op = problem.get("op", "")
                        inferred_op = infer_operation(a, b, extracted)
                        
                        if inferred_op in ['+', '-', '*'] and inferred_op != op:
                            # Wrong operation but valid
                            rewards.append(-0.5)
                        else:
                            # Too far off
                            rewards.append(-1.0)
                except:
                    # Can't parse as number - garbage output
                    rewards.append(-1.0)
                
        return rewards
    
    # GRPO config
    config = GRPOConfig(
        output_dir=f"./grpo_arithmetic_partial_{seed}",
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
    print(f"Starting training with partial credit rewards...")
    trainer.train()
    
    print("\nTraining completed!")
    
    # Test final performance
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = trainer.model
    tokenizer = trainer.tokenizer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    correct = 0
    off_by_1 = 0
    off_by_2_5 = 0
    wrong_op = 0
    test_samples = 50
    eval_table = wandb.Table(columns=["prompt", "expected", "generated", "extracted", "reward_type"])
    
    with torch.no_grad():
        for i in range(min(test_samples, len(data))):
            prompt = data[i]['prompt']
            expected = data[i]['expected']
            expected_int = data[i]['answer_int']
            a = data[i]['a']
            b = data[i]['b']
            op = data[i]['op']
            
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
            
            # Categorize result
            reward_type = "garbage"
            if extracted == expected:
                correct += 1
                reward_type = "correct"
            else:
                try:
                    extracted_int = int(extracted)
                    diff = abs(extracted_int - expected_int)
                    if diff == 1:
                        off_by_1 += 1
                        reward_type = "off_by_1"
                    elif 2 <= diff <= 5:
                        off_by_2_5 += 1
                        reward_type = "off_by_2_5"
                    else:
                        inferred_op = infer_operation(a, b, extracted)
                        if inferred_op in ['+', '-', '*'] and inferred_op != op:
                            wrong_op += 1
                            reward_type = f"wrong_op ({inferred_op})"
                except:
                    pass
            
            eval_table.add_data(prompt, expected, answer, extracted, reward_type)
                
            if i < 20:  # Print first 20
                print(f"[{i}] {prompt} → {expected}")
                print(f"     Generated: {answer}")
                print(f"     Extracted: {extracted} ({reward_type})")
    
    accuracy = 100 * correct / test_samples
    print(f"\nFinal Results:")
    print(f"  Correct: {correct}/{test_samples} ({100*correct/test_samples:.1f}%)")
    print(f"  Off by 1: {off_by_1}/{test_samples} ({100*off_by_1/test_samples:.1f}%)")
    print(f"  Off by 2-5: {off_by_2_5}/{test_samples} ({100*off_by_2_5/test_samples:.1f}%)")
    print(f"  Wrong operation: {wrong_op}/{test_samples} ({100*wrong_op/test_samples:.1f}%)")
    print(f"  Garbage: {test_samples-correct-off_by_1-off_by_2_5-wrong_op}/{test_samples}")
    print(f"{'='*60}")
    
    wandb.log({
        "final_accuracy": accuracy,
        "final_off_by_1": 100 * off_by_1 / test_samples,
        "final_off_by_2_5": 100 * off_by_2_5 / test_samples,
        "final_wrong_op": 100 * wrong_op / test_samples,
        "final_evaluation": eval_table
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()