# MeZO Multiple Perturbations Guide

## Overview

MeZO (Memory-efficient Zeroth-order) optimization estimates gradients using finite differences with random perturbations. By default, it uses **1 random direction per step**, requiring 2 forward passes (at θ+εz and θ-εz).

You can now **average over multiple perturbation directions** to get more accurate gradient estimates, at the cost of additional forward passes.

## New Parameter: `--zo_num_perturbations`

Controls how many random directions to sample and average per optimization step.

- **Default**: `1` (standard MeZO)
- **Higher values**: More accurate gradients, but slower (linear cost increase)

## Mathematical Details

### Standard MeZO (zo_num_perturbations=1)
```
∇f(θ) ≈ [f(θ + ε·z) - f(θ - ε·z)] / (2ε) · z
```
- 2 forward passes per step
- Single random direction z ~ N(0, I)

### Multi-Perturbation MeZO (zo_num_perturbations=K)
```
∇f(θ) ≈ (1/K) · Σᵢ [f(θ + ε·zᵢ) - f(θ - ε·zᵢ)] / (2ε) · zᵢ
```
- 2K forward passes per step
- Averages over K independent random directions {z₁, z₂, ..., zₖ} ~ N(0, I)

## Usage Examples

### 1. Standard MeZO (baseline)
```bash
python run.py \
  --model_name facebook/opt-350m \
  --task_name SST2 \
  --trainer zo \
  --lora \
  --learning_rate 1e-5 \
  --zo_eps 1e-3 \
  --zo_num_perturbations 1 \
  --max_steps 2000
```

### 2. Multi-Perturbation MeZO (more accurate)
```bash
python run.py \
  --model_name facebook/opt-350m \
  --task_name SST2 \
  --trainer zo \
  --lora \
  --learning_rate 1e-5 \
  --zo_eps 1e-3 \
  --zo_num_perturbations 5 \
  --max_steps 2000
```
⚠️ **Note**: This will be ~5x slower per step, so you may want to reduce `--max_steps` proportionally.

### 3. XGBLoRA with Multiple Perturbations
```bash
python run.py \
  --model_name facebook/opt-350m \
  --task_name SST2 \
  --trainer zo \
  --xgblora \
  --xgblora_steps_per_iteration 1000 \
  --learning_rate 1e-5 \
  --zo_eps 1e-3 \
  --zo_num_perturbations 3 \
  --max_steps 10000
```

### 4. Comparison Script
```bash
python compare_xgblora_lora.py \
  --output_root E:\aOutput\multi_pert_test \
  --model_name facebook/opt-350m \
  --task_name SST2 \
  --learning_rate 1e-5 \
  --zo_eps 1e-3 \
  --zo_num_perturbations 3 \
  --max_steps 2000 \
  --xgblora_steps_per_iteration 200 \
  --seeds 0 1 2
```

## Trade-offs

| zo_num_perturbations | Forward Passes/Step | Training Speed | Gradient Quality |
|---------------------|---------------------|----------------|------------------|
| 1 (default)         | 2                   | 1x (baseline)  | Standard         |
| 2                   | 4                   | ~2x slower     | Better           |
| 5                   | 10                  | ~5x slower     | Much Better      |
| 10                  | 20                  | ~10x slower    | Highest          |

## When to Use Multiple Perturbations

### ✅ **Good Use Cases**
1. **Small models** where forward passes are fast
2. **High-variance gradients** (noisy tasks)
3. **Research experiments** comparing gradient estimation quality
4. **Later training stages** when fine-grained updates matter

### ❌ **Not Recommended**
1. **Large models** (>1B params) - too slow
2. **Limited compute budget** - better to run more steps with K=1
3. **Early training** - single perturbation usually sufficient

## Expected Benefits

Research shows that averaging multiple perturbations can:
- **Reduce gradient variance** by factor of √K
- **Improve convergence stability** in later training
- **Help escape local minima** with better direction estimates

However, the **wall-clock time vs. accuracy trade-off** depends heavily on your hardware and model size.

## Implementation Details

The implementation:
1. Samples K independent random directions {z₁, z₂, ..., zₖ}
2. Computes gradient estimate for each direction
3. Averages all estimates: `mean([grad₁, grad₂, ..., gradₖ])`
4. Applies averaged gradient in the average direction: `mean([z₁, z₂, ..., zₖ])`

See `trainer.py::zo_step()` and `trainer.py::zo_update()` for full implementation.

## References

- Original MeZO paper: [Malladi et al., 2023](https://arxiv.org/abs/2305.17333)
- Zeroth-order optimization: Classical gradient-free optimization literature
- Multiple perturbations: Standard variance reduction technique in stochastic optimization

