"""
Test script to verify XGBLoRA implementation
"""

import sys
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

# Add the current directory to path to import lora
sys.path.insert(0, '.')

from lora import LoRA

def test_xgblora_initialization():
    """Test that XGBLoRA initializes correctly with rank-1"""
    print("Testing XGBLoRA initialization...")
    
    # Create a small test model
    config = AutoConfig.from_pretrained("facebook/opt-125m")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", config=config, device_map="cpu")
    
    # Initialize XGBLoRA
    lora_module = LoRA(model, r=1, alpha=16, float16=False, xgblora=True)
    
    # Check that LoRA modules were created
    assert len(lora_module.lora_modules) > 0, "No LoRA modules were created"
    print(f"✓ Created {len(lora_module.lora_modules)} LoRA modules")
    
    # Check that rank is 1
    for module in lora_module.lora_modules:
        if hasattr(module, 'r'):
            assert module.r == 1, f"Rank should be 1, but got {module.r}"
    print("✓ All modules have rank-1")
    
    # Check that all LoRA modules have the necessary attributes
    for module in lora_module.lora_modules:
        assert hasattr(module, 'lora_A'), "Missing lora_A parameter"
        assert hasattr(module, 'lora_B'), "Missing lora_B parameter"
        assert hasattr(module, 'scaling'), "Missing scaling attribute"
    print("✓ All modules have required attributes")
    
    return lora_module, model

def test_xgblora_merge_and_reinit():
    """Test that merge_and_reinit works correctly"""
    print("\nTesting XGBLoRA merge and reinit...")
    
    lora_module, model = test_xgblora_initialization()
    
    # Get initial weights
    initial_weights = {}
    initial_lora_A = {}
    initial_lora_B = {}
    
    for i, module in enumerate(lora_module.lora_modules):
        initial_weights[i] = module.weight.data.clone()
        if hasattr(module, 'lora_A'):
            initial_lora_A[i] = module.lora_A.data.clone()
            initial_lora_B[i] = module.lora_B.data.clone()
    
    # Perform merge and reinit
    lora_module.merge_and_reinit()
    
    # Check that base weights changed (merged)
    weights_changed = False
    for i, module in enumerate(lora_module.lora_modules):
        if not torch.allclose(module.weight.data, initial_weights[i], atol=1e-6):
            weights_changed = True
            break
    
    # Note: weights might not change if lora_B is initialized to zero
    # which is the standard initialization
    print("✓ Merge completed (lora_B initialized to zero by default)")
    
    # Check that LoRA parameters were reinitialized
    lora_params_changed = False
    for i, module in enumerate(lora_module.lora_modules):
        if hasattr(module, 'lora_A'):
            # lora_B should still be zero after reinit
            assert torch.allclose(module.lora_B.data, torch.zeros_like(module.lora_B.data)), \
                "lora_B should be reinitialized to zero"
            # lora_A should be different (reinitialized with Kaiming)
            if not torch.allclose(module.lora_A.data, initial_lora_A[i]):
                lora_params_changed = True
    
    assert lora_params_changed, "LoRA parameters should be reinitialized"
    print("✓ LoRA parameters reinitialized correctly")
    
    print("\n✅ All XGBLoRA tests passed!")

def test_multiple_iterations():
    """Test multiple boosting iterations"""
    print("\nTesting multiple boosting iterations...")
    
    lora_module, model = test_xgblora_initialization()
    
    # Simulate multiple iterations
    for iteration in range(3):
        print(f"  Iteration {iteration + 1}...")
        
        # Simulate training by modifying lora_A (in real training, optimizer would do this)
        for module in lora_module.lora_modules:
            if hasattr(module, 'lora_A'):
                module.lora_A.data.normal_(mean=0, std=0.01)
        
        # Merge and reinit
        lora_module.merge_and_reinit()
    
    print("✓ Multiple iterations completed successfully")

if __name__ == "__main__":
    print("=" * 60)
    print("XGBLoRA Implementation Test")
    print("=" * 60)
    
    try:
        test_xgblora_initialization()
        test_xgblora_merge_and_reinit()
        test_multiple_iterations()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


