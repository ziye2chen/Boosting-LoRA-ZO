# XGBLoRA Quick Start Guide

## What is XGBLoRA?

XGBLoRA (eXtreme Gradient Boosting LoRA) is a novel parameter-efficient fine-tuning method that:
- Uses **rank-1 LoRA adapters** as "weak learners" in a gradient boosting framework
- **Iteratively trains** multiple boosters instead of a single high-rank adapter
- **Merges and reinitializes** adapters after each boosting iteration
- Achieves **better performance with fewer parameters** than standard LoRA

## Installation

No additional dependencies required! XGBLoRA uses the existing MeZO framework.

## Quick Start Examples

### Example 1: Basic XGBLoRA with MeZO on SST-2

```bash
cd large_models
MODE=xgblora TASK=SST2 bash mezo.sh
```

This will:
- Train on SST-2 sentiment classification task
- Use rank-1 LoRA adapters
- Merge every 1000 steps (default)
- Use MeZO (zeroth-order optimization)

### Example 2: Adjust Boosting Frequency

```bash
# Merge every 500 steps
MODE=xgblora XGBLORA_STEPS=500 TASK=SST2 bash mezo.sh

# Merge every 2000 steps
MODE=xgblora XGBLORA_STEPS=2000 TASK=SST2 bash mezo.sh
```

### Example 3: Different Tasks

```bash
# Natural Language Inference (RTE)
MODE=xgblora TASK=RTE bash mezo.sh

# Question Answering (BoolQ)
MODE=xgblora TASK=BoolQ bash mezo.sh

# Textual Entailment (CB)
MODE=xgblora TASK=CB bash mezo.sh
```

### Example 4: Different Model Sizes

```bash
# OPT-125M (smallest, fastest)
MODE=xgblora MODEL=facebook/opt-125m TASK=SST2 bash mezo.sh

# OPT-350M (default)
MODE=xgblora MODEL=facebook/opt-350m TASK=SST2 bash mezo.sh

# OPT-1.3B (larger model)
MODE=xgblora MODEL=facebook/opt-1.3b TASK=SST2 bash mezo.sh

# OPT-6.7B (requires more GPU memory)
MODE=xgblora MODEL=facebook/opt-6.7b TASK=SST2 bash mezo.sh
```

### Example 5: Custom Hyperparameters

```bash
# Custom learning rate and batch size
MODE=xgblora LR=5e-6 BS=32 TASK=SST2 bash mezo.sh

# Custom training steps
MODE=xgblora STEPS=10000 EVAL_STEPS=2000 TASK=SST2 bash mezo.sh

# More training data
MODE=xgblora TRAIN=2000 DEV=1000 TASK=SST2 bash mezo.sh
```

## Understanding the Parameters

### Environment Variables (for mezo.sh)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | ft | Set to `xgblora` to enable XGBLoRA |
| `XGBLORA_STEPS` | 1000 | Steps per boosting iteration |
| `MODEL` | facebook/opt-350m | HuggingFace model name |
| `TASK` | - | Task name (SST2, RTE, BoolQ, etc.) |
| `BS` | 16 | Batch size |
| `LR` | 1e-5 | Learning rate |
| `EPS` | 1e-3 | MeZO epsilon |
| `SEED` | 0 | Random seed |
| `TRAIN` | 1000 | Number of training samples |
| `DEV` | 500 | Number of dev samples |
| `STEPS` | 20000 | Total training steps |
| `EVAL_STEPS` | 4000 | Evaluation frequency |

### Direct Python Usage

```bash
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
    --num_eval 1000 \
    --load_float16 \
    --output_dir ./output \
    --evaluation_strategy steps \
    --eval_steps 4000 \
    --save_steps 4000 \
    --logging_steps 10
```

## Comparing XGBLoRA vs Standard LoRA

### Run Standard LoRA
```bash
MODE=lora TASK=SST2 bash mezo.sh
```

### Run XGBLoRA
```bash
MODE=xgblora TASK=SST2 bash mezo.sh
```

### Compare Results
XGBLoRA typically shows:
- ✅ Similar or better accuracy
- ✅ Fewer trainable parameters per iteration
- ✅ Better convergence with proper boosting frequency

## Monitoring Training

### Log Output
Training logs show when XGBLoRA merges:

```
Step 1000: XGBLoRA: Merging and reinitializing at step 1000
Step 1000: LoRA merge and reinitialization complete
...
Step 2000: XGBLoRA: Merging and reinitializing at step 2000
Step 2000: LoRA merge and reinitialization complete
```

### Evaluation Metrics
Results are saved in `result/` directory with format:
```
{TASK}-{MODEL}-mezo-xgblora-{STEPS}-{BS}-{LR}-{EPS}-{SEED}-trainset0.json
```

## Tips for Best Results

1. **Choose Appropriate Merge Frequency**:
   - Smaller models: 500-1000 steps
   - Larger models: 1000-2000 steps
   - More complex tasks: longer intervals

2. **Learning Rate**:
   - Start with 1e-5 (default)
   - Increase to 5e-5 for faster convergence
   - Decrease to 1e-6 for stability

3. **Batch Size**:
   - Larger batch size (32-64) if GPU memory allows
   - Helps with stable gradient estimation in MeZO

4. **Total Steps**:
   - Typical range: 10000-20000 steps
   - More steps with longer merge frequency
   - Monitor dev set performance

## Troubleshooting

### Out of Memory Error
```bash
# Use smaller batch size
MODE=xgblora BS=8 TASK=SST2 bash mezo.sh

# Or use a smaller model
MODE=xgblora MODEL=facebook/opt-125m TASK=SST2 bash mezo.sh
```

### Slow Training
```bash
# Reduce number of steps
MODE=xgblora STEPS=10000 TASK=SST2 bash mezo.sh

# Increase batch size (if GPU allows)
MODE=xgblora BS=32 TASK=SST2 bash mezo.sh
```

### Poor Performance
```bash
# Try different merge frequency
MODE=xgblora XGBLORA_STEPS=500 TASK=SST2 bash mezo.sh

# Adjust learning rate
MODE=xgblora LR=5e-6 TASK=SST2 bash mezo.sh

# Increase training data
MODE=xgblora TRAIN=2000 DEV=1000 TASK=SST2 bash mezo.sh
```

## Testing the Implementation

Run the test script to verify everything works:

```bash
cd large_models
python test_xgblora.py
```

Expected output:
```
============================================================
XGBLoRA Implementation Test
============================================================
Testing XGBLoRA initialization...
✓ Created 48 LoRA modules
✓ All modules have rank-1
✓ All modules have required attributes

Testing XGBLoRA merge and reinit...
...
✅ All XGBLoRA tests passed!
```

## Next Steps

1. Try XGBLoRA on your task: `MODE=xgblora TASK=YourTask bash mezo.sh`
2. Experiment with different merge frequencies
3. Compare with standard LoRA and full fine-tuning
4. Share your results!

## Support

For detailed documentation, see `XGBLORA_IMPLEMENTATION.md`

For issues or questions, check the implementation in:
- `lora.py` - LoRA implementation with XGBLoRA support
- `trainer.py` - Training loop with boosting iterations
- `run.py` - Command-line interface


