# Critical Bug Fixes for XGBLoRA

## Summary

Fixed critical bugs that caused XGBLoRA to behave incorrectly when `xgblora_steps_per_iteration` was set. The main issue was that epoch-based merging was happening even when step-based merging was configured, causing unexpected behavior and incorrect comparisons with standard LoRA.

---

## Bug #1: Epoch-Based Merge Overriding Step-Based Merge (CRITICAL)

### Problem
When setting `--xgblora_steps_per_iteration 2000 --max_steps 2000`, XGBLoRA was still merging at every epoch (iterations 1, 2, 3, ...), instead of having NO merges until the end.

### Root Cause
In `trainer.py`, the epoch-end merge logic ran **unconditionally** if `xgblora_merge_frequency > 0`, without checking if step-based merging was enabled.

```python
# BUGGY CODE (OLD)
if hasattr(args, 'xgblora') and args.xgblora and hasattr(self, 'lora_module'):
    current_iteration = epoch + 1
    if hasattr(args, 'xgblora_merge_frequency') and args.xgblora_merge_frequency > 0:
        if current_iteration % args.xgblora_merge_frequency == 0:
            self.lora_module.merge_and_reinit()  # ❌ This runs even with step-based merge!
```

### Fix
Added a check to only use epoch-based merging when step-based merging is disabled:

```python
# FIXED CODE (NEW)
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

**Files Changed:** `large_models/trainer.py` (lines 653-668)

---

## Bug #2: Missing Alpha Configuration in Experiment Scripts

### Problem
`compare_xgblora_lora.py` was not explicitly passing `--lora_alpha` to XGBLoRA experiments, relying only on `--extra_args`, which could lead to incorrect alpha values being used.

### Fix
Added explicit `--lora_alpha` to the XGBLoRA experiment configuration:

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
        str(args.lora_alpha),  # ✓ Now explicitly passed
    ],
),
```

**Files Changed:** 
- `large_models/compare_xgblora_lora.py` (lines 521-532)
- `large_models/run_xgblora.py` (lines 182-198)

---

## Bug #3: Unclear Configuration in run_xgblora.py

### Problem
`run_xgblora.py` always set `--xgblora_merge_frequency "1"`, even when using step-based merging, which was confusing.

### Fix
Made merge_frequency conditional based on whether step-based merging is used:

```python
# XGBLoRA-specific args
xgblora_args = [
    "--xgblora",
    "--xgblora_steps_per_iteration",
    str(xgblora_steps),
    "--lora_alpha",
    str(lora_alpha),
]

# Only set merge_frequency for epoch-based merging (when step-based is disabled)
if xgblora_steps == 0:
    xgblora_args.extend(["--xgblora_merge_frequency", "1"])
else:
    # Explicitly set to 0 to disable epoch-based merging when using step-based
    xgblora_args.extend(["--xgblora_merge_frequency", "0"])
```

**Files Changed:** `large_models/run_xgblora.py` (lines 182-198)

---

## Enhancement: Better Diagnostic Logging

### Added in `run.py`
Enhanced logging to show the **actual** rank being used (not just the argument default):

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

**Files Changed:** `large_models/run.py` (lines 192-210)

### Added in `lora.py`
Added initialization logging to verify rank:

```python
logger.info(f"Initializing LoRA with: r={r}, alpha={alpha}, xgblora={xgblora}")
if xgblora and r != 1:
    logger.warning(f"XGBLoRA should use rank=1, but got r={r}. This may cause unexpected behavior!")
```

**Files Changed:** `large_models/lora.py` (lines 116-118)

---

## How to Verify the Fixes

### Test Case: No Merges During Training

Run XGBLoRA with `xgblora_steps_per_iteration = max_steps`:

```bash
python large_models/compare_xgblora_lora.py \
  --output_root E:\aOutput\test_fix \
  --loss_figure_path E:\aOutput\test_fix\loss.png \
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

**Expected Behavior:**
1. ✅ No "Merging and reinitializing at epoch X" messages during training
2. ✅ Configuration shows: `Actual rank used: 1`, `Alpha: 16`, `Scaling: 16.0`
3. ✅ Loss curves for LoRA+ZO and XGBLoRA+ZO are nearly identical
4. ✅ Training completes without any mid-training merges

**Before the Fix:**
- ❌ Merge messages at epoch 1, 2, 3, ..., 18, ...
- ❌ XGBLoRA loss curve much slower than LoRA
- ❌ Different behavior despite no merges expected

---

## Test Case: Multiple Merges During Training

Run XGBLoRA with `xgblora_steps_per_iteration = 100`:

```bash
python large_models/compare_xgblora_lora.py \
  --output_root E:\aOutput\test_merges \
  --loss_figure_path E:\aOutput\test_merges\loss.png \
  --model_name facebook/opt-350m \
  --task_name SST2 \
  --max_steps 2000 \
  --xgblora_steps_per_iteration 100 \
  --eval_steps 100 \
  --lora_rank 1 \
  --lora_alpha 16 \
  --seeds 0
```

**Expected Behavior:**
1. ✅ Merge messages at step 100, 200, 300, ..., 2000 (20 merges total)
2. ✅ NO merge messages at epochs (e.g., "epoch 1", "epoch 2")
3. ✅ Configuration shows step-based merging is enabled
4. ✅ XGBLoRA loss may differ from LoRA (expected due to boosting)

---

## Summary of Files Changed

1. **`large_models/trainer.py`** - Fixed epoch-based merge override (CRITICAL)
2. **`large_models/compare_xgblora_lora.py`** - Added explicit alpha configuration
3. **`large_models/run_xgblora.py`** - Fixed merge frequency logic
4. **`large_models/run.py`** - Enhanced diagnostic logging
5. **`large_models/lora.py`** - Added initialization logging
6. **`large_models/ALPHA_FIX_SUMMARY.md`** - Documentation
7. **`large_models/test_alpha_config.py`** - Diagnostic tool

---

## Impact

These fixes ensure that:
- XGBLoRA behaves correctly when `xgblora_steps_per_iteration > 0`
- Fair comparisons between XGBLoRA and standard LoRA are possible
- Step-based and epoch-based merging don't interfere with each other
- Alpha/rank configuration is explicit and verifiable through logs

**All users should update to these fixed versions before running new experiments.**


