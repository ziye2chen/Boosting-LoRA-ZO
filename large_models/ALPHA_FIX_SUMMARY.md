# LoRA Alpha Configuration Fix

## Problem

When running XGBLoRA with `xgblora_steps_per_iteration=2000` and `max_steps=2000` (no merge during training), the training loss curve was significantly different from standard LoRA with rank-1 and alpha=16, even though they should behave identically.

Expected behavior: Since there's no merge during training, XGBLoRA with rank-1 should be functionally equivalent to standard LoRA with rank-1.

Observed behavior: 
1. Training loss curves were drastically different (XGBLoRA learning much slower)
2. XGBLoRA was merging at every epoch (iteration 18, 19, etc.) instead of at step 2000
3. The merge happened even with `xgblora_steps_per_iteration=2000`

## Root Causes

### Issue 1: Missing Alpha Configuration
The XGBLoRA experiment configuration in `compare_xgblora_lora.py` was not explicitly passing the `--lora_alpha` argument, relying instead on the `--extra_args` mechanism. This could lead to argument ordering issues where the alpha value might not be correctly propagated.

### Issue 2: Epoch-Based Merge Overriding Step-Based Merge (CRITICAL BUG)
The epoch-end merge logic in `trainer.py` was running **even when step-based merging was configured**. The code checked `xgblora_merge_frequency > 0` without first checking if step-based merging (`xgblora_steps_per_iteration > 0`) was enabled. This caused merges to happen at every epoch regardless of the `xgblora_steps_per_iteration` setting.

## Fixes Applied

### 1. Fixed Epoch-Based Merge Override in `trainer.py` (CRITICAL)

**The Bug:**
```python
# OLD CODE (BUGGY)
if hasattr(args, 'xgblora') and args.xgblora and hasattr(self, 'lora_module'):
    current_iteration = epoch + 1
    if hasattr(args, 'xgblora_merge_frequency') and args.xgblora_merge_frequency > 0:
        if current_iteration % args.xgblora_merge_frequency == 0:
            self.lora_module.merge_and_reinit()  # This runs even with step-based merge!
```

This caused merges at **every epoch** (when `xgblora_merge_frequency=1`) even when `xgblora_steps_per_iteration=2000` was set.

**The Fix:**
```python
# NEW CODE (FIXED)
if hasattr(args, 'xgblora') and args.xgblora and hasattr(self, 'lora_module'):
    # Only merge at epoch boundaries if step-based merging is disabled
    use_step_based = hasattr(args, 'xgblora_steps_per_iteration') and args.xgblora_steps_per_iteration > 0
    if not use_step_based:
        # Epoch-based merging ONLY runs if step-based is disabled
        current_iteration = epoch + 1
        if hasattr(args, 'xgblora_merge_frequency') and args.xgblora_merge_frequency > 0:
            if current_iteration % args.xgblora_merge_frequency == 0:
                logger.info(f"XGBLoRA: Merging and reinitializing at epoch {current_iteration}")
                self.lora_module.merge_and_reinit()
```

Now epoch-based merge **only runs when step-based merge is disabled**.

### 2. Fixed `compare_xgblora_lora.py`

**Before:**
```python
ExperimentConfig(
    name="xgblora_zo",
    display_name="XGBLoRA+ZO",
    extra_args=[
        "--xgblora",
        "--xgblora_steps_per_iteration",
        str(args.xgblora_steps_per_iteration),
        "--xgblora_merge_frequency",
        "1",
        # Missing --lora_alpha!
    ],
),
```

**After:**
```python
ExperimentConfig(
    name="xgblora_zo",
    display_name="XGBLoRA+ZO",
    extra_args=[
        "--xgblora",
        "--xgblora_steps_per_iteration",
        str(args.xgblora_steps_per_iteration),
        "--xgblora_merge_frequency",
        "1",
        "--lora_alpha",
        str(args.lora_alpha),  # Now explicitly passed!
    ],
),
```

### 2. Fixed `run_xgblora.py`

Added explicit alpha parameter passing and improved code organization:

```python
# XGBLoRA-specific args (placed after base_args to ensure they take precedence)
xgblora_args = [
    "--xgblora",
    "--xgblora_steps_per_iteration",
    str(xgblora_steps),
    "--xgblora_merge_frequency",
    "1",
    "--lora_alpha",
    str(lora_alpha),
]
```

### 3. Enhanced Diagnostic Logging in `run.py`

Added comprehensive logging to show the **actual** configuration being used (not just the argument defaults):

```python
if self.args.xgblora:
    logger.info(f"=" * 80)
    logger.info(f"XGBLoRA Configuration:")
    logger.info(f"  Actual rank used: {lora_r} (forced to 1 for XGBLoRA)")
    logger.info(f"  Alpha: {self.args.lora_alpha}")
    logger.info(f"  Scaling (alpha/r): {self.args.lora_alpha/lora_r}")
    logger.info(f"  Steps per iteration: {self.args.xgblora_steps_per_iteration}")
    logger.info(f"  Merge frequency (epochs): {self.args.xgblora_merge_frequency}")
    logger.info(f"=" * 80)
```

This shows the **actual rank used** (1 for XGBLoRA) vs the argument value shown in the args printout (which might be 8).

### 4. Created Diagnostic Script

Added `test_alpha_config.py` to verify alpha configuration:

```bash
# Test both LoRA and XGBLoRA with rank=1, alpha=16
python large_models/test_alpha_config.py --rank 1 --alpha 16

# Test only XGBLoRA
python large_models/test_alpha_config.py --mode xgblora --rank 1 --alpha 16
```

### 4. Added Verification Logging in `lora.py`

Added logging in the LoRA initialization to confirm the actual rank being used:

```python
logger.info(f"Initializing LoRA with: r={r}, alpha={alpha}, xgblora={xgblora}")
if xgblora and r != 1:
    logger.warning(f"XGBLoRA should use rank=1, but got r={r}. This may cause unexpected behavior!")
```

## How to Verify the Fix

### Method 1: Check Logs (RECOMMENDED)

Run your comparison command and look for the new detailed configuration logging:

```bash
python large_models/compare_xgblora_lora.py \
  --output_root E:\aOutput\test_fix \
  --loss_figure_path E:\aOutput\test_fix\loss.png \
  --model_name facebook/opt-350m \
  --task_name SST2 \
  --max_steps 2000 \
  --xgblora_steps_per_iteration 2000 \
  --lora_rank 1 \
  --lora_alpha 16 \
  --seeds 0
```

Look for the detailed configuration blocks like:

**For XGBLoRA:**
```
================================================================================
XGBLoRA Configuration:
  Actual rank used: 1 (forced to 1 for XGBLoRA)
  Alpha: 16
  Scaling (alpha/r): 16.0
  Steps per iteration: 2000
  Merge frequency (epochs): 1
================================================================================
Initializing LoRA with: r=1, alpha=16, xgblora=True
```

**For Standard LoRA:**
```
================================================================================
Standard LoRA Configuration:
  Rank: 1
  Alpha: 16
  Scaling (alpha/r): 16.0
================================================================================
Initializing LoRA with: r=1, alpha=16, xgblora=False
```

Both should show:
- **Actual rank: 1** ✓
- **Alpha: 16** ✓
- **Scaling: 16.0** ✓

### Method 1b: Verify No Unwanted Merges

With `xgblora_steps_per_iteration=2000` and `max_steps=2000`, you should see:
- **ZERO merge messages** during training (no "Merging and reinitializing at epoch X")
- Training completes without any mid-training merges
- If you see merge messages at epochs 1, 2, 3, etc., the bug is NOT fixed

### Method 2: Run Diagnostic Script

```bash
python large_models/test_alpha_config.py --rank 1 --alpha 16
```

This will print detailed information about the LoRA configuration for both standard LoRA and XGBLoRA. Verify that:
- Both show `alpha: 16`
- Both show `scaling (alpha/r): 16.0`
- Both show `rank (r): 1`

### Method 3: Compare Loss Curves

After the fix, re-run your comparison:

```bash
python large_models/compare_xgblora_lora.py \
  --output_root E:\aOutput\test_2000_fixed \
  --loss_figure_path E:\aOutput\test_2000_fixed\loss.png \
  --model_name facebook/opt-350m \
  --task_name SST2 \
  --learning_rate 1e-5 \
  --zo_eps 1e-3 \
  --max_steps 2000 \
  --xgblora_steps_per_iteration 2000 \
  --eval_steps 100 \
  --lora_rank 1 \
  --lora_alpha 16 \
  --seeds 0
```

**Expected result:** The loss curves for LoRA+ZO and XGBLoRA+ZO should now be **nearly identical** (within random variation), since both are using rank-1 with alpha=16 and there's no merge during training.

## Additional Notes

### Proper Usage

When using `compare_xgblora_lora.py`, you no longer need to pass `--extra_args --lora_alpha X`. Simply use:

```bash
python large_models/compare_xgblora_lora.py \
  --lora_alpha 16 \  # This is enough now
  ... other args ...
```

The alpha value will be correctly propagated to both LoRA and XGBLoRA configurations.

### Understanding the Parameters

- `--lora_alpha`: Controls the scaling factor along with rank. Effective scaling = alpha / rank.
- `--lora_rank`: Controls the rank of LoRA adapters (ignored for XGBLoRA, which always uses rank-1).
- `--xgblora_steps_per_iteration`: How many steps to train each rank-1 adapter before merging.

For a fair comparison:
- **Same scaling**: Use alpha=16 for both LoRA (rank-1) and XGBLoRA (rank-1) → scaling=16/1=16
- **Same capacity**: If XGBLoRA does 20 merges, compare with LoRA rank-20

## Testing Checklist

- [ ] Run diagnostic script and verify alpha values match
- [ ] Check logs for "Using XGBLoRA...alpha=16, scaling=16.0"
- [ ] Compare loss curves with xgblora_steps_per_iteration=max_steps (should be identical)
- [ ] Compare loss curves with xgblora_steps_per_iteration < max_steps (XGBLoRA should be better with proper boosting)

## Files Modified

1. `large_models/compare_xgblora_lora.py` - Fixed XGBLoRA experiment configuration
2. `large_models/run_xgblora.py` - Fixed argument passing
3. `large_models/run.py` - Added diagnostic logging
4. `large_models/test_alpha_config.py` - New diagnostic script
5. `large_models/ALPHA_FIX_SUMMARY.md` - This document

