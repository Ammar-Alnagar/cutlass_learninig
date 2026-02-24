# Module 03: Softmax - Missing Elements Challenge

## Objective
Fix the incomplete softmax kernel implementing the numerically stable two-pass algorithm.

## What's Missing
The `kernel.py` file has **16+ missing elements** including:

### Algorithm Steps to Implement
1. **First Pass**: Find maximum value in each row (for numerical stability)
2. **Second Pass**: Compute `exp(x - max)` and sum
3. **Third Pass**: Normalize by dividing by the sum

### Key Triton Operations
- `tl.max()` for finding maximum
- `tl.exp()` for exponential
- `tl.sum()` for reduction
- Numerical stability pattern

### Mathematical Background
Softmax formula: `softmax(x)_i = exp(x_i) / sum(exp(x_j))`

Numerically stable version: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`

## Hints
- Each program handles one row of the input
- Use `tl.max()` to find the maximum for stability
- The subtraction of max prevents overflow in exp()
- Final output rows should sum to 1.0

## Success Criteria
- Kernel compiles without errors
- Results match PyTorch's softmax
- Each row sums to 1.0

## Running the Test
```bash
python kernel.py
```
