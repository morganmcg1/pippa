#!/usr/bin/env python3
"""
GRPO Ultra-High Diversity Experiment - 64 Generations
Based on the successful mixed dataset script that achieved 54.7% accuracy
Target: Break 60% final accuracy barrier with 4x more generation diversity
"""

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
from dotenv import load_dotenv
import os
import wandb
import re
from typing import Dict, Any, List
import logging

# Load environment variables
load_dotenv()

# Force seed for reproducibility
SEED = 7890
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Custom GRPOTrainer to fix log_completions issue
class GRPOTrainerFixed(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_completions = []
    
    def _generate_completions(self, prompts, **generation_kwargs):
        completions = super()._generate_completions(prompts, **generation_kwargs)
        self._last_completions = completions
        return completions
    
    def _print_completions_simple(self):
        if hasattr(self, '_last_completions') and self._last_completions:
            print("\n=== Sample Completions ===")
            for i, (prompt, completion) in enumerate(zip(self.state.current_batch['prompt'][:3], 
                                                         self._last_completions[:3])):
                print(f"\nPrompt {i+1}: {prompt}")
                print(f"Completion {i+1}: {completion}")
            print("=" * 50 + "\n")
    
    def log(self, logs: Dict[str, Any], start_time: float = None) -> None:
        try:
            super().log(logs, start_time)
        except AttributeError as e:
            if "add_section" in str(e):
                self._print_completions_simple()
                if hasattr(self, '_wandb'):
                    self._wandb.log(logs, step=self.state.global_step)
            else:
                raise

def create_mixed_dataset(n_samples=150):
    """Create a mixed dataset with arithmetic, counting, and comparison tasks"""
    prompts = []
    answers = []
    task_types = []
    
    # Task distribution: 50% arithmetic, 25% counting, 25% comparison
    n_arithmetic = n_samples // 2
    n_counting = n_samples // 4
    n_comparison = n_samples - n_arithmetic - n_counting
    
    # Arithmetic tasks
    operations = ['+', '-', '*']
    for _ in range(n_arithmetic):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        op = random.choice(operations)
        
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:  # *
            result = a * b
        
        prompt = f"Calculate: {a} {op} {b} = "
        prompts.append(prompt)
        answers.append(str(result))
        task_types.append("arithmetic")
    
    # Counting tasks (easier)
    for _ in range(n_counting):
        count = random.randint(1, 10)
        word = random.choice(["apple", "cat", "dog", "star", "book"])
        words = " ".join([word] * count)
        
        prompt = f"Count the words: {words}. Answer: "
        prompts.append(prompt)
        answers.append(str(count))
        task_types.append("counting")
    
    # Comparison tasks (easier)
    for _ in range(n_comparison):
        a = random.randint(0, 20)
        b = random.randint(0, 20)
        
        if a > b:
            answer = "yes"
        else:
            answer = "no"
        
        prompt = f"Is {a} greater than {b}? Answer yes or no: "
        prompts.append(prompt)
        answers.append(answer)
        task_types.append("comparison")
    
    # Shuffle to mix task types
    combined = list(zip(prompts, answers, task_types))
    random.shuffle(combined)
    prompts, answers, task_types = zip(*combined)
    
    return Dataset.from_dict({
        "prompt": list(prompts),
        "answer": list(answers),
        "task_type": list(task_types)
    })

def extract_answer(text: str, prompt: str, task_type: str) -> str:
    """Extract the answer from model output based on task type"""
    completion = text[len(prompt):].strip()
    
    if task_type == "comparison":
        # Look for yes/no
        completion_lower = completion.lower()
        if "yes" in completion_lower:
            return "yes"
        elif "no" in completion_lower:
            return "no"
    
    # For arithmetic and counting, extract number
    match = re.match(r'^-?\d+', completion)
    if match:
        return match.group(0)
    
    # Fallback
    tokens = completion.split()
    if tokens:
        return tokens[0].lower() if task_type == "comparison" else tokens[0]
    
    return completion

def reward_function(samples: List[str], prompts: List[str], answers: List[str], 
                   task_types: List[str], **kwargs) -> List[float]:
    """Binary reward with task-specific extraction"""
    rewards = []
    
    for sample, prompt, expected, task_type in zip(samples, prompts, answers, task_types):
        extracted = extract_answer(sample, prompt, task_type)
        
        if extracted == expected:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    
    return rewards

def main():
    # Model configuration
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # Training configuration - ULTRA HIGH DIVERSITY
    n_samples = 150
    batch_size = 960  # 64 generations × 15 prompts
    num_generations = 64  # 4x more than successful baseline!
    learning_rate = 5e-6
    temperature = 0.7
    epochs = 50
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="grpo_arithmetic_ultra_diversity_64gen",
        tags=["grpo-setup", "arithmetic", "ultra-diversity", "mixed-tasks", "64-generations", "break-60"],
        config={
            "model": model_name,
            "n_samples": n_samples,
            "batch_size": batch_size,
            "num_generations": num_generations,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "epochs": epochs,
            "seed": SEED,
            "beta": 0.1,
            "experiment": "ultra_high_diversity",
            "task_distribution": "50% arithmetic, 25% counting, 25% comparison",
            "effective_batch_size": batch_size,
            "note": "64 generations for maximum diversity - 4x baseline"
        }
    )
    
    print("Preparing training configuration...")
    print(f"\n{'='*60}")
    print("ULTRA-HIGH DIVERSITY EXPERIMENT - 64 GENERATIONS")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size} (64 gens × 15 prompts)")
    print(f"Generations: {num_generations} (4x increase from baseline)")
    print(f"Dataset: Mixed tasks for better generalization")
    print(f"Target: Break 60% final accuracy barrier")
    print(f"{'='*60}\n")
    
    print("Creating mixed dataset...")
    dataset = create_mixed_dataset(n_samples)
    
    # Count task types
    task_counts = {}
    for task_type in dataset['task_type']:
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    print(f"\nDataset composition:")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} samples ({count/n_samples*100:.1f}%)")
    
    print("\nSample problems:")
    for i in range(9):  # Show 3 of each type
        print(f"[{i}] [{dataset['task_type'][i]}] {dataset['prompt'][i]} → {dataset['answer'][i]}")
    
    # Prepare dataset
    def prepare_dataset(sample):
        return {
            "prompt": sample["prompt"],
            "answer": sample["answer"],
            "task_type": sample["task_type"]
        }
    
    dataset = dataset.map(prepare_dataset)
    
    # Create reward function wrapper (CRITICAL: exact structure from working script)
    def reward_wrapper(completions, prompts=None, **kwargs):
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        # Get indices from kwargs - this is how GRPO passes the batch indices
        batch_indices = kwargs.get('batch_indices', None)
        
        if batch_indices is not None:
            answers = [dataset[idx]["answer"] for idx in batch_indices]
            task_types = [dataset[idx]["task_type"] for idx in batch_indices]
        else:
            # Fallback: try to match prompts to dataset
            prompt_to_data = {d["prompt"]: (d["answer"], d["task_type"]) for d in dataset}
            answers = []
            task_types = []
            for p in prompts:
                if p in prompt_to_data:
                    ans, tt = prompt_to_data[p]
                    answers.append(ans)
                    task_types.append(tt)
                else:
                    answers.append("")
                    task_types.append("arithmetic")
        
        return reward_function(completions, prompts, answers, task_types, **kwargs)
    
    # GRPO configuration with ULTRA HIGH DIVERSITY
    config = GRPOConfig(
        output_dir="./grpo-ultra-diversity-64gen",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="no",
        report_to="wandb",
        remove_unused_columns=False,
        log_completions=True,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=16,
        max_prompt_length=128,
        seed=SEED,
        # GRPO specific
        beta=0.1,
        loss_type="grpo",
        # Additional
        dataloader_num_workers=0,
        wandb_log_unique_prompts=True,
        bf16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",
        scale_rewards=True,
    )
    
    print("\nGRPO Configuration:")
    print(f"Ultra-high generation diversity: {num_generations} generations")
    print(f"Beta (KL penalty): {config.beta}")
    print(f"Batch size: {batch_size} (supports 64 generations)")
    print(f"Learning rate: {learning_rate}")
    
    # Initialize trainer
    trainer = GRPOTrainerFixed(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_wrapper],
    )
    
    print("\nStarting training with ULTRA-HIGH diversity...")
    print("This will be slower but should achieve superior accuracy!")
    trainer.train()
    
    # Evaluate final performance by task type
    print("\nEvaluating final model...")
    results_by_type = {}
    
    # Get model and tokenizer from trainer
    model = trainer.model
    tokenizer = trainer.tokenizer
    
    model.eval()
    with torch.no_grad():
        for task_type in task_counts.keys():
            correct = 0
            total = 0
            
            for i in range(len(dataset)):
                if dataset[i]["task_type"] != task_type:
                    continue
                
                prompt = dataset[i]["prompt"]
                expected = dataset[i]["answer"]
                
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                extracted = extract_answer(response, prompt, task_type)
                
                if extracted == expected:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            results_by_type[task_type] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy
            }
            print(f"{task_type}: {accuracy:.2%} ({correct}/{total})")
    
    # Overall accuracy
    total_correct = sum(r["correct"] for r in results_by_type.values())
    total_samples = sum(r["total"] for r in results_by_type.values())
    overall_accuracy = total_correct / total_samples
    
    print(f"\nOverall accuracy: {overall_accuracy:.2%} ({total_correct}/{total_samples})")
    
    # Log final metrics
    wandb.log({
        "final_accuracy": overall_accuracy,
        "final_accuracy_arithmetic": results_by_type.get("arithmetic", {}).get("accuracy", 0),
        "final_accuracy_counting": results_by_type.get("counting", {}).get("accuracy", 0),
        "final_accuracy_comparison": results_by_type.get("comparison", {}).get("accuracy", 0),
        "final_correct": total_correct,
        "final_total": total_samples
    })
    
    wandb.finish()
    print("\nTraining completed!")
    print(f"Final accuracy: {overall_accuracy:.2%}")
    print("Did we break 60%? " + ("YES! 🎉" if overall_accuracy > 0.6 else "Not yet, but close!"))

if __name__ == "__main__":
    main()