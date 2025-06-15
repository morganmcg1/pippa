#!/bin/bash
# Test what the current proposer model is generating

ssh ubuntu@192.222.52.59 << 'EOF'
cd ~/pippa/absolute_zero

# Create a quick test script
cat > test_current_proposer.py << 'PYTHON'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the latest proposer model
model_path = "./absolute_zero_proposer_iter_6"  # Use iter 6 since 7 might still be training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading proposer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()

# Test prompts
prompts = [
    "Generate a simple arithmetic problem in the format 'Calculate: X + Y = ': ",
    "Calculate: ",
    "Create a math problem with addition or subtraction in the format 'Calculate: X - Y = ': ",
]

print("\nTesting current proposer outputs:\n")
for prompt in prompts:
    print(f"Prompt: '{prompt}'")
    
    for i in range(3):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated[len(prompt):].strip()
        print(f"  Sample {i+1}: '{completion}'")
    print()
PYTHON

# Run the test
source az_venv/bin/activate
python test_current_proposer.py
EOF