#!/usr/bin/env python3
"""GRPO arithmetic training with curriculum learning."""

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import wandb
import os
from dotenv import load_dotenv
import random
import re
from typing import Dict, Any, List

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

def create_curriculum_dataset(stage: int, n_samples: int = 100) -> List[Dict]:
    """Create dataset for specific curriculum stage."""
    prompts = []
    
    if stage == 1:
        # Stage 1: Addition only, numbers 0-10
        for _ in range(n_samples):
            a = random.randint(0, 10)
            b = random.randint(0, 10)
            answer = a + b
            prompts.append({
                "prompt": f"Calculate: {a} + {b} = ",
                "expected": str(answer),
                "stage": 1
            })
    
    elif stage == 2:
        # Stage 2: All operations, numbers 0-10
        for _ in range(n_samples):
            a = random.randint(0, 10)
            b = random.randint(0, 10)
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
                "stage": 2
            })
    
    else:  # stage == 3
        # Stage 3: All operations, numbers 0-20
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
                "stage": 3
            })
    
    return prompts

def main():
    # Configuration with curriculum learning
    n_samples_per_stage = 100
    batch_size = 64
    num_generations = 16
    lr = 5e-6
    temperature = 0.7
    epochs_stage1 = 20
    epochs_stage2 = 30  # 21-50
    epochs_stage3 = 50  # 51-100
    total_epochs = epochs_stage1 + epochs_stage2 + epochs_stage3
    seed = 777
    beta = 0.1  # Standard KL penalty
    
    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Initialize wandb
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name=f"grpo_arithmetic_curriculum",
        config={
            "task": "arithmetic_curriculum",
            "n_samples_per_stage": n_samples_per_stage,
            "batch_size": batch_size,
            "num_generations": num_generations,
            "learning_rate": lr,
            "temperature": temperature,
            "total_epochs": total_epochs,
            "seed": seed,
            "beta": beta,
            "model": "Qwen/Qwen2-0.5B-Instruct",
            "log_completions": True,
            "wandb_log_unique_prompts": True,
            "curriculum": {
                "stage1": {"epochs": epochs_stage1, "desc": "Addition only, 0-10"},
                "stage2": {"epochs": epochs_stage2, "desc": "All ops, 0-10"},
                "stage3": {"epochs": epochs_stage3, "desc": "All ops, 0-20"}
            }
        },
        tags=["grpo-setup", "overfit", "arithmetic", "curriculum", "break-barrier"]
    )
    
    print(f"\n{'='*60}")
    print(f"ARITHMETIC GRPO WITH CURRICULUM LEARNING")
    print(f"{'='*60}")
    print(f"Curriculum Structure:")
    print(f"  Stage 1 (epochs 1-{epochs_stage1}): Addition only, numbers 0-10")
    print(f"  Stage 2 (epochs {epochs_stage1+1}-{epochs_stage1+epochs_stage2}): All operations, numbers 0-10")
    print(f"  Stage 3 (epochs {epochs_stage1+epochs_stage2+1}-{total_epochs}): All operations, numbers 0-20")
    print(f"\nTraining Configuration:")
    print(f"  Samples per stage: {n_samples_per_stage}")
    print(f"  Batch size: {batch_size}")
    print(f"  Generations per prompt: {num_generations}")
    print(f"  Learning rate: {lr}")
    print(f"  Temperature: {temperature}")
    print(f"  Beta (KL penalty): {beta}")
    print(f"{'='*60}\n")
    
    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Train through curriculum stages
    current_epoch = 0
    
    for stage in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"STARTING CURRICULUM STAGE {stage}")
        print(f"{'='*60}")
        
        # Create dataset for this stage
        data = create_curriculum_dataset(stage, n_samples_per_stage)
        dataset = Dataset.from_list([{"prompt": d["prompt"]} for d in data])
        expected_answers = {d["prompt"]: d["expected"] for d in data}
        
        # Determine epochs for this stage
        if stage == 1:
            stage_epochs = epochs_stage1
        elif stage == 2:
            stage_epochs = epochs_stage2
        else:
            stage_epochs = epochs_stage3
        
        # Print stage info
        print(f"Stage {stage} Dataset:")
        for i in range(min(10, len(data))):
            print(f"  [{i}] {data[i]['prompt']} → {data[i]['expected']}")
        print(f"Training for {stage_epochs} epochs...\n")
        
        # Log stage transition
        wandb.log({"curriculum_stage": stage, "epoch": current_epoch})
        
        # Create reward function for this stage
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
        
        # GRPO config for this stage
        config = GRPOConfig(
            output_dir=f"./grpo_arithmetic_curriculum_stage{stage}",
            num_train_epochs=stage_epochs,
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
            seed=seed + stage,  # Different seed per stage
            bf16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            log_completions=True,
            wandb_log_unique_prompts=True,
            # Optimization parameters
            warmup_ratio=0.1 if stage == 1 else 0.0,  # Warmup only in first stage
            weight_decay=0.01,
            max_grad_norm=1.0,
            optim="adamw_torch",
            scale_rewards=True,
        )
        
        # Create trainer with fixed logging
        trainer = GRPOTrainerFixed(
            model=model,  # Use same model instance across stages
            tokenizer=tokenizer,
            args=config,
            train_dataset=dataset,
            reward_funcs=[reward_function],
        )
        
        # Train this stage
        print(f"Starting Stage {stage} training...")
        trainer.train()
        
        # Update epoch counter
        current_epoch += stage_epochs
        
        # Evaluate after each stage
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        print(f"\nStage {stage} Evaluation:")
        correct = 0
        test_samples = 20
        
        with torch.no_grad():
            for i in range(min(test_samples, len(data))):
                prompt = data[i]['prompt']
                expected = data[i]['expected']
                
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
                
                if i < 5:  # Print first 5
                    print(f"  {prompt} → {expected} (got: {extracted}) {'✓' if is_correct else '✗'}")
        
        stage_accuracy = 100 * correct / test_samples
        print(f"Stage {stage} Accuracy: {correct}/{test_samples} ({stage_accuracy:.1f}%)")
        wandb.log({f"stage_{stage}_accuracy": stage_accuracy, "epoch": current_epoch})
    
    # Final evaluation on full test set
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON FULL TEST SET")
    print(f"{'='*60}")
    
    # Create test set with all operations and full range
    test_data = create_curriculum_dataset(3, 100)  # Stage 3 = full difficulty
    correct = 0
    eval_table = wandb.Table(columns=["prompt", "expected", "generated", "extracted", "is_correct"])
    
    model.eval()
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
    
    final_accuracy = 100 * correct / len(test_data)
    print(f"\nFinal Test Accuracy: {correct}/{len(test_data)} ({final_accuracy:.1f}%)")
    print(f"{'='*60}")
    
    wandb.log({
        "final_test_accuracy": final_accuracy,
        "final_evaluation": eval_table
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()