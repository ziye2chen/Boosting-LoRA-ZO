# XGBLoRA Adaptive Merge - Quick Start Guide

## What Changed?

XGBLoRA now supports **adaptive merging** based on loss smoothing and patience, replacing the fixed-step merging strategy. This aligns with the XGBLoRA paper's Algorithm 1.

## Key Features

✅ **Loss Smoothing (EMA)**: Filters noisy training loss using exponential moving average  
✅ **Best Checkpoint Tracking**: Automatically saves the best adapter weights  
✅ **Patience-Based Convergence**: Merges when no improvement for N steps  
✅ **Automatic Rollback**: Restores best checkpoint before merging  
✅ **Iteration Tracking**: Counts and logs each boosting iteration  

## Quick Usage

### Basic Command (Adaptive Merge - Recommended)

```bash
python run.py \
  --model_name facebook/opt-350m \
  --task_name SST2 \
  --trainer zo \
  --xgblora \
  --xgblora_use_adaptive_merge \
  --xgblora_patience 100 \
  --learning_rate 1e-5 \
  --zo_eps 1e-3 \
  --max_steps 2000 \
  --output_dir E:\aOutput\xgblora_test
```

### With Custom Parameters

```bash
python run.py \
  --xgblora \
  --xgblora_use_adaptive_merge \
  --xgblora_patience 150 \
  --xgblora_ema_beta 0.95 \
  --xgblora_improvement_threshold 0.005 \
  ...
```

### Comparison Script

```bash
python compare_xgblora_lora.py \
  --output_root E:\aOutput\adaptive_test \
  --model_name facebook/opt-350m \
  --task_name DROP \
  --learning_rate 1e-5 \
  --zo_eps 1e-3 \
  --max_steps 2000 \
  --xgblora_use_adaptive_merge \
  --xgblora_patience 100 \
  --lora_rank 1 \
  --lora_alpha 4 \
  --seeds 0
```

### Use Old Step-Based Merge (Optional)

```bash
python run.py \
  --xgblora \
  --xgblora_steps_per_iteration 1000 \
  --xgblora_use_adaptive_merge false \
  ...
```

## Expected Logs

When running with adaptive merge, you'll see:

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
XGBLoRA: Saved best adapter checkpoint to ...
```

**When Merging:**
```
XGBLoRA: Patience exhausted (100/100)
================================================================================
XGBLoRA: Triggering merge for iteration 1
  Reason: patience
  Step: 1234
  Best smoothed loss: 1.234567
XGBLoRA: ✓ Rolled back to best adapter weights
XGBLoRA: ✓ Merged and reinitialized adapter
XGBLoRA: ✓ Reset state for next iteration
================================================================================
```

## Parameters Quick Reference

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--xgblora_use_adaptive_merge` | `True` | Enable adaptive merge |
| `--xgblora_patience` | `100` | Steps without improvement before merge |
| `--xgblora_ema_beta` | `0.9` | Smoothing factor (0=raw, 1=frozen) |
| `--xgblora_improvement_threshold` | `0.01` | Minimum loss reduction to count as improvement |

## Common Scenarios

### Scenario 1: Fast Iteration (Quick Experiments)
```bash
--xgblora_patience 50 --xgblora_ema_beta 0.85
```
→ Merges quickly, good for testing

### Scenario 2: Stable Convergence (Production)
```bash
--xgblora_patience 200 --xgblora_ema_beta 0.95
```
→ Trains each adapter thoroughly

### Scenario 3: Noisy Task
```bash
--xgblora_patience 150 --xgblora_ema_beta 0.98 --xgblora_improvement_threshold 0.02
```
→ Heavy smoothing, requires clear improvement

### Scenario 4: Clean Task with Multiple Perturbations
```bash
--xgblora_patience 100 \
--xgblora_ema_beta 0.9 \
--zo_num_perturbations 3
```
→ Better gradients + adaptive merge

## Output Files

After training, check:

```
output_dir/
├── xgblora_best_adapter/
│   └── adapter_weights.pt       ← Best checkpoint
├── training_loss.jsonl          ← Per-step loss
└── eval_loss.jsonl              ← Evaluation loss
```

## Troubleshooting

### "No merges happening"
- Check logs for "XGBLoRA: Loss improved" messages
- If patience counter keeps resetting → Loss is still improving
- Solution: Increase `max_steps` or decrease `patience`

### "Merging too frequently"
- Patience is too small for the task
- Solution: Increase `patience` (e.g., 200)

### "Loss very noisy"
- Increase `ema_beta` (e.g., 0.95 or 0.98)
- Consider using `--zo_num_perturbations 2` or higher

### "Want to see merge at specific steps"
- Disable adaptive merge:
```bash
--xgblora_steps_per_iteration 1000 --xgblora_use_adaptive_merge false
```

## Comparison with Step-Based

| Feature | Step-Based (Old) | Adaptive (New) |
|---------|------------------|----------------|
| Merge trigger | Fixed steps | Loss + patience |
| Task-adaptive | ❌ No | ✅ Yes |
| Saves best weights | ❌ No | ✅ Yes |
| Rollback on merge | ❌ No | ✅ Yes |
| Loss smoothing | ❌ No | ✅ Yes (EMA) |
| Auto-convergence | ❌ No | ✅ Yes |

## Need More Details?

See `XGBLORA_ADAPTIVE_MERGE.md` for complete algorithm description and technical details.

## Test It Out

Run a quick test (100 steps):
```bash
bash test_adaptive_merge.sh
```

Or test your parameters:
```bash
python run.py \
  --xgblora \
  --xgblora_use_adaptive_merge \
  --xgblora_patience 20 \
  --max_steps 100 \
  --num_train 100 \
  --output_dir test_output \
  --overwrite_output_dir \
  ...
```

Watch for "XGBLoRA: Triggering merge" messages in the logs!

