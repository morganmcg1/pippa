#!/usr/bin/env python3
"""
Hybrid approach: First supervised fine-tuning, then GRPO.
This should give the model a better starting point for GRPO training.
"""

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import GRPOConfig, GRPOTrainer
import random
import numpy as np
from dotenv import load_dotenv
import os
import wandb
import re
from typing import Dict, Any, List

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
                if hasattr(self, '_wandb') and self._wandb:
                    self._wandb.log(logs, step=self.state.global_step)
            else:
                raise

def create_supervised_dataset(n_samples=500):
    """Create larger dataset for supervised training with correct answers."""
    prompts = []
    texts = []
    
    operations = ['+', '-', '*']
    for _ in range(n_samples):
        a = random.randint(0, 10)
        b = random.randint(0, 10)
        op = random.choice(operations)
        
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:  # *
            result = a * b
        
        prompt = f"Calculate: {a} {op} {b} = "
        text = f"{prompt}{result}"
        
        prompts.append(prompt)
        texts.append(text)
    
    return Dataset.from_dict({
        "text": texts,
        "prompt": prompts,
        "answer": [t.split("= ")[-1] for t in texts]
    })

def create_grpo_dataset(n_samples=130):
    """Create arithmetic dataset for GRPO fine-tuning."""
    prompts = []
    answers = []
    
    operations = ['+', '-', '*']
    for _ in range(n_samples):
        a = random.randint(0, 10)
        b = random.randint(0, 10)
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
    
    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers
    })

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

def reward_function(samples: List[str], prompts: List[str], answers: List[str], **kwargs) -> List[float]:
    """Binary reward function"""
    rewards = []
    for sample, prompt, expected in zip(samples, prompts, answers):
        extracted = extract_answer(sample, prompt)
        if extracted == expected:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards

def main():
    # Model configuration
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="supervised_then_grpo",
        tags=["grpo-setup", "standardized-eval", "hybrid-approach", "supervised+grpo"],
        config={
            "model": model_name,
            "supervised_samples": 500,
            "grpo_samples": 130,
            "seed": SEED,
            "eval_dataset": "morgan/arithmetic_eval",
            "number_range": "0-10",
            "approach": "supervised_then_grpo"
        }
    )
    
    print("\n" + "="*60)
    print("HYBRID APPROACH: SUPERVISED + GRPO")
    print("="*60)
    print("Phase 1: Supervised fine-tuning to teach format")
    print("Phase 2: GRPO to optimize performance")
    print("="*60 + "\n")
    
    # Phase 1: Supervised Fine-tuning
    print("Phase 1: Creating supervised training dataset...")
    supervised_dataset = create_supervised_dataset(n_samples=500)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)
    
    tokenized_dataset = supervised_dataset.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments for supervised learning
    training_args = TrainingArguments(
        output_dir="./supervised-arithmetic",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="no",
        evaluation_strategy="no",
        report_to="wandb",
        run_name="supervised_phase",
        bf16=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting supervised fine-tuning...")
    trainer.train()
    
    # Evaluate after supervised training
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    print("\nEvaluating after supervised training...")
    supervised_eval = evaluate_on_standard_dataset(model, tokenizer, device)
    print(f"Accuracy after supervised training: {supervised_eval['eval_accuracy']:.2%}")
    
    wandb.log({
        "supervised/eval_accuracy": supervised_eval['eval_accuracy'],
        **{f"supervised/{k}": v for k, v in supervised_eval.items()}
    })
    
    # Phase 2: GRPO Fine-tuning
    print("\nPhase 2: Creating GRPO dataset...")
    grpo_dataset = create_grpo_dataset(n_samples=130)
    
    # Prepare dataset
    def prepare_dataset(sample):
        return {
            "prompt": sample["prompt"],
            "answer": sample["answer"]
        }
    
    grpo_dataset = grpo_dataset.map(prepare_dataset)
    
    # Create reward function wrapper
    def reward_wrapper(completions, prompts=None, **kwargs):
        if prompts is None:
            prompts = kwargs.get('prompt', [])
        
        batch_indices = kwargs.get('batch_indices', None)
        
        if batch_indices is not None:
            answers = [grpo_dataset[idx]["answer"] for idx in batch_indices]
        else:
            prompt_to_answer = {d["prompt"]: d["answer"] for d in grpo_dataset}
            answers = [prompt_to_answer.get(p, "") for p in prompts]
        
        return reward_function(completions, prompts, answers, **kwargs)
    
    # GRPO configuration
    config = GRPOConfig(
        output_dir="./grpo-after-supervised",
        run_name="grpo_phase",
        per_device_train_batch_size=256,
        num_train_epochs=30,  # Fewer epochs needed after supervised
        learning_rate=5e-6,
        logging_steps=1,
        gradient_accumulation_steps=1,
        save_strategy="no",
        report_to="wandb",
        remove_unused_columns=False,
        log_completions=True,
        num_generations=16,
        temperature=0.7,
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
    
    # Initialize GRPO trainer with the supervised model
    grpo_trainer = GRPOTrainerFixed(
        model=model,  # Use the supervised model
        args=config,
        train_dataset=grpo_dataset,
        reward_funcs=[reward_wrapper],
        tokenizer=tokenizer,
    )
    
    print("\nStarting GRPO fine-tuning on supervised model...")
    grpo_trainer.train()
    
    print("\nGRPO training completed!")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_eval_metrics = evaluate_on_standard_dataset(model, tokenizer, device)
    
    wandb.log({
        "final/eval_accuracy": final_eval_metrics['eval_accuracy'],
        **{f"final/{k}": v for k, v in final_eval_metrics.items()}
    })
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS - Hybrid Approach")
    print("="*60)
    print(f"Base model baseline: ~38%")
    print(f"After supervised: {supervised_eval['eval_accuracy']:.2%}")
    print(f"After GRPO: {final_eval_metrics['eval_accuracy']:.2%}")
    print(f"GRPO-only best: 45.5%")
    print(f"Total improvement: {final_eval_metrics['eval_accuracy'] - 0.38:.2%}")
    print("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    main()