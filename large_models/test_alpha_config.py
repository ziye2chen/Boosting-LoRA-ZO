#!/usr/bin/env python3
"""
Diagnostic script to verify LoRA alpha configuration for both standard LoRA and XGBLoRA.
This helps debug why XGBLoRA and LoRA might have different training behavior.
"""

import sys
import torch
from pathlib import Path

# Add the directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run import OurArguments, Framework
import argparse


def test_configuration(mode, rank, alpha):
    """Test a specific configuration and print the resulting LoRA settings."""
    print(f"\n{'='*80}")
    print(f"Testing: {mode} with rank={rank}, alpha={alpha}")
    print(f"{'='*80}")
    
    # Build arguments
    args = OurArguments(
        model_name="facebook/opt-125m",  # Small model for quick testing
        task_name="SST2",
        trainer="zo",
        lora=(mode == "lora"),
        xgblora=(mode == "xgblora"),
        lora_r=rank,
        lora_alpha=alpha,
        max_steps=10,  # Very short
        num_train=10,
        num_dev=10,
        num_eval=10,
        per_device_train_batch_size=2,
        output_dir=f"test_output_{mode}_r{rank}_a{alpha}",
        result_file=f"test_result_{mode}_r{rank}_a{alpha}.json",
    )
    
    print(f"\nArguments:")
    print(f"  args.lora: {args.lora}")
    print(f"  args.xgblora: {args.xgblora}")
    print(f"  args.lora_r: {args.lora_r}")
    print(f"  args.lora_alpha: {args.lora_alpha}")
    
    # Initialize framework
    framework = Framework(args, None)
    
    # Load model (this will create LoRA modules)
    print(f"\nLoading model and initializing LoRA...")
    model, tokenizer = framework.load_model()
    
    # Check LoRA modules
    if hasattr(framework, 'lora_module') and framework.lora_module is not None:
        print(f"\nLoRA module created successfully!")
        print(f"  XGBLoRA mode: {framework.lora_module.xgblora}")
        print(f"  Number of LoRA modules: {len(framework.lora_module.lora_modules) if framework.lora_module.xgblora else 'N/A (standard LoRA)'}")
        
        # Check the actual LoRA linear layers
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                print(f"\n  Found LoRA layer: {name}")
                print(f"    rank (r): {module.r}")
                print(f"    alpha: {module.lora_alpha}")
                print(f"    scaling (alpha/r): {module.scaling}")
                print(f"    lora_A shape: {module.lora_A.shape}")
                print(f"    lora_B shape: {module.lora_B.shape}")
                print(f"    merged: {module.merged}")
                # Only print first few to avoid clutter
                break
    else:
        print(f"\n  ERROR: No LoRA module was created!")
    
    print(f"\n{'='*80}\n")
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LoRA alpha configuration")
    parser.add_argument("--mode", choices=["lora", "xgblora"], default="both",
                        help="Which mode to test (lora, xgblora, or both)")
    parser.add_argument("--rank", type=int, default=1, help="LoRA rank to test")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha to test")
    args = parser.parse_args()
    
    print("="*80)
    print("LoRA Alpha Configuration Diagnostic Tool")
    print("="*80)
    
    if args.mode == "both":
        # Test both configurations
        test_configuration("lora", args.rank, args.alpha)
        test_configuration("xgblora", args.rank, args.alpha)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("If both configurations show the same alpha and scaling values,")
        print("then the configuration is correct. If they differ, there's a bug")
        print("in how arguments are being passed or processed.")
        print("="*80)
    else:
        test_configuration(args.mode, args.rank, args.alpha)


