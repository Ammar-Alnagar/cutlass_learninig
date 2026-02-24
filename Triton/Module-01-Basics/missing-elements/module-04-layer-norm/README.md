# Module 04: Layer Normalization - Missing Elements Challenge

## Objective
Fix the incomplete layer normalization kernel.

## What's Missing
The `kernel.py` file has **17+ missing elements** including:

### Algorithm Steps to Implement
1. **Mean Calculation**: `mean = sum(x) / N`
2. **Variance Calculation**: `var = sum((x - mean)^2) / N`
3. **Normalization**: `x_norm = (x - mean) / sqrt(var + eps)`
4. **Affine Transform**: `output = weight * x_norm + bias`

### Key Triton Operations
- `tl.sum()` for reduction
- `tl.sqrt()` for square root
- Element-wise operations
- Loading scale/weight parameters

### Mathematical Background
Layer Normalization normalizes across features for each sample independently, 
unlike BatchNorm which normalizes across the batch.

## Hints
- Each program handles one row
- Use `eps` for numerical stability in sqrt
- Weight and bias are 1D tensors of size n_cols
- The default eps is 1e-5

## Success Criteria
- Kernel compiles without errors
- Results match PyTorch's layer_norm
- Handles various input sizes

## Running the Test
```bash
python kernel.py
```
