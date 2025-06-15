#!/usr/bin/env python3
"""
Test script for unified Absolute Zero implementation.
Tests key components before full training.
"""

import sys
sys.path.append('.')

from train_absolute_zero_unified import UnifiedAbsoluteZeroTrainer, TaskBuffer, TRRPlusBaselines
import torch

def test_task_creation():
    """Test task creation for all three types."""
    print("\n=== Testing Task Creation ===")
    
    device = torch.device("cpu")
    trainer = UnifiedAbsoluteZeroTrainer("Qwen/Qwen2-0.5B-Instruct", device)
    
    # Test deduction task
    task = trainer.create_deduction_task(5, '+', 3)
    print(f"Deduction task: {task}")
    assert task['expression'] == '5 + 3'
    assert task['answer'] == '8'
    
    # Test abduction task
    task = trainer.create_abduction_task('+', 10)
    print(f"Abduction task: {task}")
    assert task['result'] == 10
    assert task['operation'] == '+'
    assert len(task['valid_pairs']) > 0
    
    # Test induction task
    task = trainer.create_induction_task('addition', 6)
    print(f"Induction task: {task}")
    assert len(task['all_examples']) == 6
    assert len(task['visible_examples']) == 3
    assert len(task['hidden_examples']) == 3
    
    print("✓ All task creation tests passed!")


def test_prompts():
    """Test prompt generation."""
    print("\n=== Testing Prompt Generation ===")
    
    device = torch.device("cpu")
    trainer = UnifiedAbsoluteZeroTrainer("Qwen/Qwen2-0.5B-Instruct", device)
    
    # Create sample tasks
    deduction_task = trainer.create_deduction_task(7, '-', 2)
    abduction_task = trainer.create_abduction_task('*', 12)
    induction_task = trainer.create_induction_task('multiplication', 4)
    
    # Test proposer prompts
    proposer_prompt = trainer.create_proposer_prompt('deduction', [deduction_task])
    print(f"Proposer prompt (deduction):\n{proposer_prompt}")
    
    # Test solver prompts
    solver_prompt = trainer.create_solver_prompt(deduction_task, 'deduction')
    print(f"\nSolver prompt (deduction):\n{solver_prompt}")
    
    solver_prompt = trainer.create_solver_prompt(abduction_task, 'abduction')
    print(f"\nSolver prompt (abduction):\n{solver_prompt}")
    
    solver_prompt = trainer.create_solver_prompt(induction_task, 'induction')
    print(f"\nSolver prompt (induction):\n{solver_prompt[:200]}...")
    
    print("✓ All prompt tests passed!")


def test_parsing():
    """Test parsing of proposer generations."""
    print("\n=== Testing Parse Functions ===")
    
    device = torch.device("cpu")
    trainer = UnifiedAbsoluteZeroTrainer("Qwen/Qwen2-0.5B-Instruct", device)
    
    # Test deduction parsing
    generation = "Calculate: 8 + 4 = "
    parsed = trainer.parse_proposer_generation(generation, 'deduction')
    print(f"Parsed deduction: {parsed}")
    assert parsed is not None
    assert parsed['expression'] == '8 + 4'
    
    # Test abduction parsing
    generation = "Find: ? - ? = 5"
    parsed = trainer.parse_proposer_generation(generation, 'abduction')
    print(f"Parsed abduction: {parsed}")
    assert parsed is not None
    assert parsed['result'] == 5
    
    # Test induction parsing
    generation = "Pattern: (2,3)→6, (4,5)→20, (1,7)→7"
    parsed = trainer.parse_proposer_generation(generation, 'induction')
    print(f"Parsed induction: {parsed}")
    
    print("✓ All parsing tests passed!")


def test_evaluation():
    """Test solver evaluation."""
    print("\n=== Testing Solver Evaluation ===")
    
    device = torch.device("cpu")
    trainer = UnifiedAbsoluteZeroTrainer("Qwen/Qwen2-0.5B-Instruct", device)
    
    # Test deduction evaluation
    task = trainer.create_deduction_task(6, '*', 7)
    reward = trainer.evaluate_solver_response("42", task, 'deduction')
    print(f"Deduction eval (correct): {reward}")
    assert reward == 1.0
    
    reward = trainer.evaluate_solver_response("41", task, 'deduction')
    print(f"Deduction eval (wrong): {reward}")
    assert reward == -1.0
    
    # Test abduction evaluation
    task = trainer.create_abduction_task('+', 15)
    reward = trainer.evaluate_solver_response("7 and 8", task, 'abduction')
    print(f"Abduction eval (correct): {reward}")
    assert reward == 1.0
    
    print("✓ All evaluation tests passed!")


def test_buffers():
    """Test task buffers."""
    print("\n=== Testing Task Buffers ===")
    
    buffer = TaskBuffer(maxlen=10)
    
    # Add tasks
    for i in range(15):
        buffer.add({'id': i})
    
    print(f"Buffer length (should be 10): {len(buffer)}")
    assert len(buffer) == 10
    
    # Test sampling
    samples = buffer.sample(5)
    print(f"Sampled {len(samples)} tasks")
    assert len(samples) == 5
    
    # Test oversampling
    samples = buffer.sample(20)
    print(f"Oversampled {len(samples)} tasks")
    assert len(samples) == 20
    
    print("✓ All buffer tests passed!")


def test_baselines():
    """Test TRR++ baselines."""
    print("\n=== Testing TRR++ Baselines ===")
    
    baselines = TRRPlusBaselines()
    
    # Add some rewards
    baselines.update('proposer', 'deduction', 0.5)
    baselines.update('proposer', 'deduction', 0.7)
    baselines.update('solver', 'abduction', -0.3)
    
    # Test baseline computation
    baseline = baselines.get_baseline('proposer', 'deduction')
    print(f"Proposer deduction baseline: {baseline}")
    assert abs(baseline - 0.6) < 0.01
    
    # Test advantage
    advantage = baselines.compute_advantage('proposer', 'deduction', 0.9)
    print(f"Advantage for 0.9 reward: {advantage}")
    assert abs(advantage - 0.3) < 0.01
    
    print("✓ All baseline tests passed!")


def main():
    """Run all tests."""
    print("Testing Unified Absolute Zero Implementation")
    print("=" * 50)
    
    try:
        test_task_creation()
        test_prompts()
        test_parsing()
        test_evaluation()
        test_buffers()
        test_baselines()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED! ✓")
        print("Implementation is ready for training.")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()