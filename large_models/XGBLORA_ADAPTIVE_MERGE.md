# XGBLoRA Adaptive Merge Implementation

This document describes the **adaptive merging strategy** for XGBLoRA, which replaces fixed-step merging with a loss-based, patience-driven approach as described in the XGBLoRA paper.

## Overview

Instead of merging LoRA adapters at fixed step intervals, adaptive merge uses:
1. **Exponential Moving Average (EMA)** to smooth training loss
2. **Best loss tracking** to save the best adapter checkpoint
3. **Patience counter** to detect convergence
4. **Rollback mechanism** to restore best weights before merging

## Algorithm Flow

### Stage 1: Initialization
- Initialize base model W_base
- Set smoothed loss L̃₀ = Initial Loss
- Set best loss record L_best = ∞
- Set patience counter p = 0

### Stage 2: Iterative Training (Outer Loop: k = 1...N)

Initialize rank-1 adapter ΔWₖ (B=0, A~N(0,σ²))

#### Inner Loop: Per Training Step

**1. ZO Update Step**
- Perform parameter update using ZO optimizer
- Obtain current raw loss Lₜ

**2. Loss Smoothing**
- Apply EMA: L̃ₜ = β·L̃ₜ₋₁ + (1-β)·Lₜ
- Default β = 0.9

**3. Best Checkpoint & Patience Update**

```
If L̃ₜ < L_best - ε:  # Significant improvement
    • Update best record: L_best ← L̃ₜ
    • Reset patience: p ← 0
    • Save checkpoint: ΔW*ₖ ← ΔWₖ
Else:  # No improvement
    • Increment patience: p ← p + 1
```

**4. Convergence Trigger**

```
If p ≥ P (Patience Exhausted) OR t ≥ MaxSteps:
    • Rollback: ΔWₖ ← ΔW*ₖ (restore best weights)
    • Merge: W_base ← W_base + (α/r)·ΔW*ₖ
    • Reset for next iteration: p ← 0, L̃ ← None
    • Initialize ΔWₖ₊₁
```

## New Parameters

### In `run.py` / `compare_xgblora_lora.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--xgblora_use_adaptive_merge` | bool | `True` | Enable adaptive merge (disables step-based) |
| `--xgblora_patience` | int | `100` | Patience steps before merging (0 = disabled) |
| `--xgblora_ema_beta` | float | `0.9` | EMA smoothing parameter (0 = no smoothing) |
| `--xgblora_improvement_threshold` | float | `0.01` | Epsilon threshold for improvement |
| `--xgblora_steps_per_iteration` | int | `0` | If >0, override adaptive with step-based |

## Usage Examples

### 1. Adaptive Merge (Recommended)

```bash
python run.py \
  --model_name facebook/opt-350m \
  --task_name SST2 \
  --trainer zo \
  --xgblora \
  --xgblora_use_adaptive_merge \
  --xgblora_patience 100 \
  --xgblora_ema_beta 0.9 \
  --xgblora_improvement_threshold 0.01 \
  --learning_rate 1e-5 \
  --zo_eps 1e-3 \
  --max_steps 10000 \
  --output_dir E:\aOutput\xgblora_adaptive
```

### 2. Comparison: Adaptive vs Fixed-Step

```bash
# Adaptive merge (patience=100)
python compare_xgblora_lora.py \
  --output_root E:\aOutput\adaptive_test \
  --model_name facebook/opt-350m \
  --task_name DROP \
  --learning_rate 1e-5 \
  --zo_eps 1e-3 \
  --max_steps 2000 \
  --xgblora_use_adaptive_merge \
  --xgblora_patience 100 \
  --xgblora_ema_beta 0.9 \
  --seeds 0 1 2
```

### 3. Disable Adaptive (Use Step-Based)

```bash
python run.py \
  --xgblora \
  --xgblora_steps_per_iteration 1000 \
  --xgblora_use_adaptive_merge false \
  ...
```

## Implementation Details

### Key Functions in `trainer.py`

#### 1. `xgblora_save_best_adapter()`
Saves current adapter weights (lora_A, lora_B) to disk checkpoint.

#### 2. `xgblora_load_best_adapter()`
Restores adapter weights from best checkpoint (rollback).

#### 3. `xgblora_update_smoothed_loss(raw_loss)`
Updates smoothed loss using EMA and checks for improvement.
Returns: `(smoothed_loss, should_merge, merge_reason)`

#### 4. `xgblora_trigger_merge(reason)`
Performs complete merge cycle:
1. Rollback to best checkpoint (if reason="patience")
2. Merge adapter into base model
3. Reinitialize adapter
4. Reset state for next iteration

### Training Loop Integration

```python
# In _inner_training_loop(), after each step:
if xgblora and xgblora_use_adaptive_merge:
    current_loss = tr_loss.item() / global_step
    smoothed_loss, should_merge, reason = self.xgblora_update_smoothed_loss(current_loss)
    
    if should_merge:
        self.xgblora_trigger_merge(reason=reason)
```

## Output Files

### Checkpoint Directory Structure
```
output_dir/
├── xgblora_best_adapter/
│   └── adapter_weights.pt       # Best adapter checkpoint
├── training_loss.jsonl          # Per-step training loss
├── eval_loss.jsonl              # Per-eval evaluation loss
└── trainer_state.json           # Full training state
```

### Log Messages

**Initialization:**
```
XGBLoRA Adaptive Merge initialized:
  Patience: 100
  EMA Beta: 0.9
  Improvement Threshold: 0.01
```

**During Training:**
```
XGBLoRA: Initial smoothed loss = 2.345678
XGBLoRA: Loss improved from 2.345 to 2.123 (Δ=0.222)
XGBLoRA: Saved best adapter checkpoint
XGBLoRA: Patience exhausted (100/100)
```

**Merge Trigger:**
```
================================================================================
XGBLoRA: Triggering merge for iteration 1
  Reason: patience
  Step: 1234
  Best smoothed loss: 1.234567
  Current smoothed loss: 1.245678
XGBLoRA: ✓ Rolled back to best adapter weights
XGBLoRA: Merging adapter into base model...
XGBLoRA: ✓ Merged and reinitialized adapter
XGBLoRA: ✓ Reset state for next iteration
================================================================================
```

## Advantages Over Step-Based Merge

| Aspect | Step-Based | Adaptive (Loss + Patience) |
|--------|------------|---------------------------|
| **Merge Decision** | Fixed intervals | Loss-driven |
| **Convergence Detection** | Manual tuning | Automatic |
| **Quality Guarantee** | No | Yes (best checkpoint) |
| **Flexibility** | Low (same steps for all tasks) | High (adapts per task) |
| **Rollback** | No | Yes (restore best) |
| **Smoothing** | No | Yes (EMA filter) |

## Hyperparameter Tuning Guide

### Patience (`xgblora_patience`)
- **Small (50-100)**: Faster iterations, less training per adapter
- **Large (200-500)**: More training per adapter, potentially better quality
- **0**: Disabled (only merge at max_steps)

### EMA Beta (`xgblora_ema_beta`)
- **High (0.95-0.99)**: Heavy smoothing, slow to react
- **Medium (0.85-0.95)**: Balanced smoothing
- **Low (0.5-0.85)**: Light smoothing, reactive to recent loss
- **0**: No smoothing (use raw loss)

### Improvement Threshold (`xgblora_improvement_threshold`)
- **Small (0.001-0.01)**: Sensitive to small improvements
- **Large (0.05-0.1)**: Only react to significant improvements

## Debugging Tips

### Check Merge Frequency
Look for log messages containing "XGBLoRA: Triggering merge".
- Too frequent → Increase patience or threshold
- Too rare → Decrease patience or threshold

### Check Loss Smoothing
- If loss is very noisy: Increase `ema_beta`
- If slow to detect improvement: Decrease `ema_beta`

### Check Checkpoint Behavior
```bash
# List saved checkpoints
ls output_dir/xgblora_best_adapter/

# Check if rollback is happening
grep "Rolled back" output_dir/training.log
```

## Performance Considerations

- **Checkpoint I/O**: Each improvement triggers a save (fast for LoRA, ~MB)
- **Memory**: Minimal overhead (stores one extra copy of adapter weights)
- **Compute**: Negligible overhead (<0.1% per step)

## References

- Original XGBLoRA Paper: Algorithm 1
- EMA Smoothing: Standard exponential moving average
- Early Stopping: Classical machine learning patience mechanism

