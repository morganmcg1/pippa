#!/usr/bin/env python3
"""
GRPO training with dynamic curriculum learning.
Adaptively increases difficulty based on model performance.
"""

import torch
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
from dotenv import load_dotenv
import os
import wandb
import re
from typing import Dict, Any, List, Tuple
from collections import deque

# Load environment variables
load_dotenv()

# Enable WandB model checkpointing
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class DynamicCurriculumDataset:
    """Dataset that dynamically adjusts difficulty based on performance."""
    
    def __init__(self, base_size=130, window_size=10):
        self.base_size = base_size
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        self.current_difficulty = 1  # Start with easiest
        self.difficulty_ranges = {
            1: (0, 5),   # Very easy
            2: (0, 7),   # Easy
            3: (0, 10),  # Medium
            4: (5, 15),  # Hard
            5: (10, 20), # Very hard
        }
        self.regenerate_dataset()
    
    def regenerate_dataset(self):
        """Create new dataset based on current difficulty."""
        min_num, max_num = self.difficulty_ranges[self.current_difficulty]
        
        prompts = []
        answers = []
        operations = ['+', '-', '*']
        
        for _ in range(self.base_size):
            a = random.randint(min_num, max_num)
            b = random.randint(min_num, max_num)
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
        
        self.dataset = Dataset.from_dict({
            "prompt": prompts,
            "answer": answers
        })
        
        # Apply prepare function
        self.dataset = self.dataset.map(lambda x: {"prompt": x["prompt"], "answer": x["answer"]})
        
        print(f"\nGenerated dataset for difficulty {self.current_difficulty}: "
              f"numbers {min_num}-{max_num}")
    
    def update_performance(self, reward: float):
        """Update performance tracking and adjust difficulty if needed."""
        self.performance_history.append(reward)
        
        if len(self.performance_history) >= self.window_size:
            avg_performance = np.mean(self.performance_history)
            
            # Increase difficulty if doing well
            if avg_performance > 0.7 and self.current_difficulty < 5:
                self.current_difficulty += 1
                self.performance_history.clear()
                self.regenerate_dataset()
                print(f"\n🎯 Increasing difficulty to level {self.current_difficulty}!")
                wandb.log({"curriculum/difficulty_level": self.current_difficulty})
            
            # Decrease difficulty if struggling
            elif avg_performance < 0.3 and self.current_difficulty > 1:
                self.current_difficulty -= 1
                self.performance_history.clear()
                self.regenerate_dataset()
                print(f"\n📉 Decreasing difficulty to level {self.current_difficulty}")
                wandb.log({"curriculum/difficulty_level": self.current_difficulty})
    
    def get_dataset(self):
        return self.dataset

# Custom GRPOTrainer with dynamic curriculum
class GRPOTrainerDynamicCurriculum(GRPOTrainer):
    def __init__(self, *args, curriculum_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_dataset = curriculum_dataset
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
                if hasattr(self, '_wandb') and self._wandb:
                    self._wandb.log(logs, step=self.state.global_step)
            else:
                raise
        
        # Update curriculum based on performance
        if 'train/reward' in logs and self.curriculum_dataset:
            self.curriculum_dataset.update_performance(logs['train/reward'])
    
    def get_train_dataloader(self):
        """Override to use dynamic dataset."""
        if self.curriculum_dataset:
            self.train_dataset = self.curriculum_dataset.get_dataset()
        return super().get_train_dataloader()

def evaluate_on_standard_dataset(model, tokenizer, device) -> Dict[str, float]:
    """Evaluate model on the standardized arithmetic evaluation dataset."""
    eval_dataset = load_dataset("morgan/arithmetic_eval", split="test")
    
    results_by_difficulty = {}
    results_by_operation = {}
    correct_total = 0
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(eval_dataset):
            prompt = sample['prompt']
            expected = sample['answer']
            difficulty = sample['difficulty']
            operation = sample['metadata']['operation'] if sample['metadata'] else None
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            completion = response[len(prompt):].strip()
            match = re.match(r'^-?\d+', completion)
            predicted = match.group(0) if match else completion.split()[0] if completion else ""
            
            is_correct = predicted == expected
            if is_correct:
                correct_total += 1
            
            if difficulty not in results_by_difficulty:
                results_by_difficulty[difficulty] = {'correct': 0, 'total': 0}
            results_by_difficulty[difficulty]['total'] += 1
            if is_correct:
                results_by_difficulty[difficulty]['correct'] += 1
            
            if operation:
                if operation not in results_by_operation:
                    results_by_operation[operation] = {'correct': 0, 'total': 0}
                results_by_operation[operation]['total'] += 1
                if is_correct:
                    results_by_operation[operation]['correct'] += 1
    
    overall_accuracy = correct_total / len(eval_dataset)
    
    difficulty_accs = {}
    for diff in ['very_easy', 'easy', 'medium', 'hard', 'very_hard']:
        if diff in results_by_difficulty:
            stats = results_by_difficulty[diff]
            acc = stats['correct'] / stats['total']
            difficulty_accs[f"eval_{diff}_accuracy"] = acc
    
    operation_accs = {}
    for op in ['+', '-', '*', '/']:
        if op in results_by_operation:
            stats = results_by_operation[op]
            acc = stats['correct'] / stats['total']
            op_name = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}[op]
            operation_accs[f"eval_{op_name}_accuracy"] = acc
    
    return {
        'eval_accuracy': overall_accuracy,
        'eval_correct': correct_total,
        'eval_total': len(eval_dataset),
        **difficulty_accs,
        **operation_accs
    }

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

def rich_reward_function(samples: List[str], prompts: List[str], answers: List[str], **kwargs) -> List[float]:
    """Rich reward function with partial credit."""
    rewards = []
    
    for sample, prompt, expected in zip(samples, prompts, answers):
        extracted = extract_answer(sample, prompt)
        
        try:
            predicted_num = int(extracted)
            expected_num = int(expected)
            
            distance = abs(predicted_num - expected_num)
            
            if distance == 0:
                reward = 1.0
            elif distance == 1:
                reward = 0.7
            elif distance == 2:
                reward = 0.4
            elif distance <= 5:
                reward = 0.1
            elif distance <= 10:
                reward = -0.2
            else:
                reward = -0.5
                
        except (ValueError, AttributeError):
            reward = -1.0
        
        rewards.append(reward)
    
    return rewards

def main():
    # Model configuration
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # Training configuration
    n_samples = 130
    batch_size = 256
    num_generations = 16
    learning_rate = 5e-6
    temperature = 0.7
    epochs = 100  # More epochs for curriculum
    
    # Initialize dynamic curriculum dataset
    curriculum_dataset = DynamicCurriculumDataset(base_size=n_samples, window_size=10)
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="grpo_dynamic_curriculum",
        tags=["grpo-setup", "standardized-eval", "rich-rewards", "dynamic-curriculum"],
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
            "eval_dataset": "morgan/arithmetic_eval",
            "curriculum_type": "dynamic",
            "window_size": 10,
            "promotion_threshold": 0.7,
            "demotion_threshold": 0.3
        }
    )
    
    print("\n" + "="*60)
    print("DYNAMIC CURRICULUM LEARNING EXPERIMENT")
    print("="*60)
    print("Starting at difficulty 1 (numbers 0-5)")
    print("Will automatically adjust based on performance:")
    print("- Promote to harder level if avg reward > 0.7")
    print("- Demote to easier level if avg reward < 0.3")
    print("- Window size: 10 batches for averaging")
    print("="*60 + "\n")
    
    # Create reward function wrapper that updates with dataset
    def reward_wrapper(completions, prompts=None, **kwargs):
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        batch_indices = kwargs.get('batch_indices', None)
        dataset = curriculum_dataset.get_dataset()
        
        if batch_indices is not None:
            answers = [dataset[idx]["answer"] for idx in batch_indices]
        else:
            prompt_to_answer = {d["prompt"]: d["answer"] for d in dataset}
            answers = [prompt_to_answer.get(p, "") for p in prompts]
        
        return rich_reward_function(completions, prompts, answers, **kwargs)
    
    # GRPO configuration
    config = GRPOConfig(
        output_dir="./grpo-dynamic-curriculum",
        run_name="grpo_dynamic_curriculum",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
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
    
    # Initialize trainer with dynamic curriculum
    trainer = GRPOTrainerDynamicCurriculum(
        model=model_name,
        args=config,
        train_dataset=curriculum_dataset.get_dataset(),
        reward_funcs=[reward_wrapper],
        curriculum_dataset=curriculum_dataset
    )
    
    print("Starting training with dynamic curriculum...")
    trainer.train()
    
    print("\nTraining completed!")
    
    # Final evaluation
    model = trainer.model
    tokenizer = trainer.tokenizer
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    
    print("\nRunning final standardized evaluation...")
    final_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
    
    wandb.log({
        "final/eval_accuracy": final_eval_metrics['eval_accuracy'],
        "final/difficulty_reached": curriculum_dataset.current_difficulty,
        **{f"final/{k}": v for k, v in final_eval_metrics.items()}
    })
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS - Dynamic Curriculum")
    print("="*60)
    print(f"Final standardized eval accuracy: {final_eval_metrics['eval_accuracy']:.2%}")
    print(f"Final difficulty level reached: {curriculum_dataset.current_difficulty}/5")
    print(f"Previous rich rewards best: 75.0%")
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    main()