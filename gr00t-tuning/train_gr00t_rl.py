#!/usr/bin/env python3
"""
GR00T N1.5 Reinforcement Learning Training Script
Implements GRPO-style training with proper WandB logging
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.distributions import Categorical
from dotenv import load_dotenv
import wandb
import numpy as np

# Load environment variables
load_dotenv()

class RobotCommandReward:
    """Reward function for robot command generation"""
    
    def __init__(self):
        self.valid_commands = {
            "MOVE_ARM", "ROTATE_BASE", "OPEN_GRIPPER", 
            "CLOSE_GRIPPER", "SCAN_ENVIRONMENT"
        }
    
    def compute_reward(self, prompt: str, response: str) -> float:
        """Compute reward for robot command generation"""
        reward = 0.0
        
        # Check if response contains a valid command
        command_found = False
        for cmd in self.valid_commands:
            if cmd in response:
                command_found = True
                reward += 1.0
                break
        
        # Check for proper parameter format
        if command_found and "(" in response and ")" in response:
            reward += 0.5
        
        # Check for parameter values
        if "=" in response:
            reward += 0.5
        
        # Penalty for repetition
        words = response.split()
        if len(words) > len(set(words)):
            reward -= 0.5
        
        return reward

def train_gr00t_rl(
    model_name: str = "Qwen/Qwen2-0.5B-Instruct",
    num_epochs: int = 50,
    batch_size: int = 4,
    num_generations: int = 8,
    learning_rate: float = 5e-6,
    temperature: float = 0.7
):
    """Main training function"""
    
    # Initialize WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "pippa"),
        entity=os.getenv("WANDB_ENTITY", "wild-ai"),
        name="gr00t-rl-training",
        tags=["gr00t-overfit", "rl-training"],
        config={
            "model": model_name,
            "batch_size": batch_size,
            "num_generations": num_generations,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "num_epochs": num_epochs,
            "task": "robot_command_generation"
        }
    )
    
    # Initialize model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Initialize reward function
    reward_fn = RobotCommandReward()
    
    # Training data
    prompts = [
        "Move the robot arm to position (10, 20, 30)",
        "Rotate the base 90 degrees clockwise",
        "Open the gripper to pick up the object",
        "Scan the environment for obstacles"
    ]
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_rewards = []
        
        for prompt_idx, prompt in enumerate(prompts):
            # Generate multiple responses
            prompt_text = f"Human: {prompt}\nAssistant:"
            inputs = tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                # Generate num_generations responses
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=num_generations,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Compute rewards
            rewards = []
            responses = []
            for gen_id in generated_ids:
                response = tokenizer.decode(gen_id[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                responses.append(response)
                reward = reward_fn.compute_reward(prompt, response)
                rewards.append(reward)
            
            rewards = torch.tensor(rewards, device=device)
            
            # Normalize rewards (GRPO style)
            if rewards.std() > 0:
                normalized_rewards = (rewards - rewards.mean()) / rewards.std()
            else:
                normalized_rewards = rewards - rewards.mean()
            
            # Compute loss for each generation
            batch_loss = 0
            for i, (gen_id, norm_reward) in enumerate(zip(generated_ids, normalized_rewards)):
                # Get log probabilities
                outputs = model(input_ids=gen_id.unsqueeze(0), labels=gen_id.unsqueeze(0))
                
                # Weight loss by normalized reward
                weighted_loss = -outputs.loss * norm_reward
                batch_loss += weighted_loss
            
            # Average loss across generations
            batch_loss = batch_loss / num_generations
            
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            epoch_losses.append(batch_loss.item())
            epoch_rewards.extend(rewards.tolist())
            
            # Log per-step metrics
            wandb.log({
                "train/step_loss": batch_loss.item(),
                "rewards/step_mean": rewards.mean().item(),
                "rewards/step_max": rewards.max().item(),
                "rewards/step_min": rewards.min().item(),
                "train/prompt_idx": prompt_idx,
                "train/epoch": epoch
            }, step=global_step)
            
            global_step += 1
        
        # Log epoch metrics
        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean(epoch_rewards)
        max_reward = np.max(epoch_rewards)
        
        wandb.log({
            "train/epoch_loss": avg_loss,
            "train/epoch": epoch,
            "rewards/epoch_mean": avg_reward,
            "rewards/epoch_max": max_reward,
            "rewards/epoch_std": np.std(epoch_rewards),
            "train/learning_rate": optimizer.param_groups[0]['lr']
        }, step=global_step)
        
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.2f}, Max Reward = {max_reward:.2f}")
        
        # Early stopping if we achieve good rewards
        if avg_reward > 1.8:
            print(f"Early stopping: achieved average reward of {avg_reward:.2f}")
            break
    
    # Test the model
    print("\nTesting trained model:")
    model.eval()
    
    test_prompts = [
        "Move the robot arm to position (50, 60, 70)",
        "Close the gripper gently"
    ]
    
    for test_prompt in test_prompts:
        test_input = f"Human: {test_prompt}\nAssistant:"
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {test_prompt}")
        print(f"Response: {response}")
        
        # Compute test reward
        test_reward = reward_fn.compute_reward(test_prompt, response)
        print(f"Reward: {test_reward}")
        
        wandb.log({
            "test/reward": test_reward,
            "test/example": wandb.Table(
                columns=["prompt", "response", "reward"],
                data=[[test_prompt, response, test_reward]]
            )
        }, step=global_step)
        global_step += 1
    
    wandb.finish()
    print("\nTraining complete!")

if __name__ == "__main__":
    train_gr00t_rl()