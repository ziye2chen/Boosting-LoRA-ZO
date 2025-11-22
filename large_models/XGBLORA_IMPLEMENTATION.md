# XGBLoRA Implementation

## Overview

This implementation adds **XGBLoRA (eXtreme Gradient Boosting LoRA)** support to the MeZO framework. XGBLoRA is a parameter-efficient fine-tuning technique that applies gradient boosting principles to Large Language Models (LLMs) using Low-Rank Adapters (LoRA) as "weak learners."

## Key Features

1. **Gradient Boosting with LoRA**: Instead of training a single high-rank adapter, XGBLoRA trains multiple rank-1 adapters iteratively
2. **Progressive Refinement**: After each iteration:
   - Train a lightweight rank-1 booster to minimize residual errors
   - Merge the booster parameters into the base model
   - Instantiate a new booster for the next iteration
3. **All Layers**: Uses LoRA on all attention layers (q_proj and v_proj) rather than randomly selecting layers
4. **Memory Efficient**: Requires significantly fewer trainable parameters than standard LoRA

## Implementation Details

### Modified Files

#### 1. `lora.py`
- Added `xgblora` parameter to `LoRA` class constructor
- Added `lora_modules` list to track all LoRA layers
- Implemented `merge_and_reinit()` method:
  - Merges current LoRA weights (lora_B @ lora_A) into base weights
  - Reinitializes LoRA parameters for next boosting iteration

#### 2. `trainer.py`
- Added XGBLoRA boosting logic in the training loop
- Two merge strategies:
  - **Epoch-based**: Merge at the end of each epoch (controlled by `xgblora_merge_frequency`)
  - **Step-based**: Merge every N steps (controlled by `xgblora_steps_per_iteration`)
- Checks for `xgblora` flag and `lora_module` attribute before merging

#### 3. `run.py`
- Added new command-line arguments:
  - `--xgblora`: Enable XGBLoRA mode
  - `--xgblora_steps_per_iteration`: Number of steps per boosting iteration (0 = epoch-based)
  - `--xgblora_merge_frequency`: Merge frequency in epochs (only when step-based is disabled)
- Modified LoRA initialization:
  - Uses rank-1 by default when XGBLoRA is enabled
  - Passes `xgblora=True` to LoRA constructor
  - Stores LoRA module reference for trainer access

#### 4. `mezo.sh`
- Added `xgblora` mode option
- Added `XGBLORA_STEPS` environment variable (default: 1000)
- Updated extra arguments handling for XGBLoRA

## Usage

### Basic Usage with MeZO

```bash
# Run XGBLoRA with MeZO on SST-2 task
MODE=xgblora TASK=SST2 bash mezo.sh

# Customize boosting iteration steps
MODE=xgblora XGBLORA_STEPS=500 TASK=SST2 bash mezo.sh

# Run on different models and tasks
MODE=xgblora MODEL=facebook/opt-1.3b TASK=RTE bash mezo.sh
```

### Advanced Usage with Custom Settings

```bash
# Use step-based merging with custom hyperparameters
python run.py \
    --model_name facebook/opt-350m \
    --task_name SST2 \
    --xgblora \
    --xgblora_steps_per_iteration 1000 \
    --trainer zo \
    --learning_rate 1e-5 \
    --zo_eps 1e-3 \
    --per_device_train_batch_size 16 \
    --max_steps 20000 \
    --num_train 1000 \
    --num_dev 500 \
    --load_float16
```

### Epoch-Based Merging

```bash
# Merge at the end of every epoch
python run.py \
    --model_name facebook/opt-350m \
    --task_name SST2 \
    --xgblora \
    --xgblora_steps_per_iteration 0 \
    --xgblora_merge_frequency 1 \
    --trainer zo \
    --num_train_epochs 10 \
    --learning_rate 1e-5
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--xgblora` | False | Enable XGBLoRA mode |
| `--xgblora_steps_per_iteration` | 0 | Steps per boosting iteration (0 = use epoch-based) |
| `--xgblora_merge_frequency` | 1 | Merge frequency in epochs (when step-based is 0) |
| `--lora_r` | 8 (ignored for XGBLoRA) | Rank for standard LoRA |
| `--lora_alpha` | 16 | Scaling factor for LoRA updates |

**Note**: When `--xgblora` is enabled, the rank is automatically set to 1 regardless of `--lora_r` setting.

## Algorithm Flow

```
For each boosting iteration:
    1. Train rank-1 LoRA adapter on current residuals
    2. At iteration end (epoch or N steps):
       a. Merge: W_base = W_base + α * (B @ A)
       b. Reset: Reinitialize A and B for next iteration
    3. Repeat until max_steps or convergence
```

## Comparison with Standard LoRA

| Aspect | Standard LoRA | XGBLoRA |
|--------|--------------|---------|
| Rank | r (typically 8-64) | 1 (fixed) |
| Adapters | Single high-rank | Multiple rank-1 (boosting) |
| Updates | One-shot training | Iterative with merging |
| Memory | O(r × d) | O(1 × d) per iteration |
| Performance | Good | Matches/exceeds with fewer params |

## Benefits

1. **Higher Efficiency**: Fewer trainable parameters per iteration
2. **Better Performance**: Boosting can capture complex patterns progressively
3. **Flexibility**: Can control number of boosting iterations
4. **Hardware Friendly**: Enables training on consumer-grade GPUs

## Notes

- XGBLoRA automatically applies to all attention layers (q_proj and v_proj)
- Compatible with MeZO (zeroth-order optimization)
- Can be combined with float16/bfloat16 for additional memory savings
- Checkpoint saving/loading is supported through standard mechanisms

## Example Output

```
Inject lora to: model.decoder.layers.0.self_attn
Inject lora to: model.decoder.layers.1.self_attn
...
Using XGBLoRA with rank-1 adapters
***** Running training *****
  Num examples = 1000
  Total optimization steps = 20000
  
... training ...

XGBLoRA: Merging and reinitializing at step 1000
LoRA merge and reinitialization complete

... continues training ...

XGBLoRA: Merging and reinitializing at step 2000
LoRA merge and reinitialization complete
```

## References

- XGBLoRA paper: `XGBLoRA.pdf`
- Original LoRA: https://arxiv.org/abs/2106.09685
- MeZO: https://arxiv.org/abs/2305.17333


