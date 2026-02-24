# Module 05: 2D Convolution - Missing Elements Challenge

## Objective
Fix the incomplete 2D convolution kernel - the most complex challenge in this series.

## What's Missing
The `kernel.py` file has **25+ missing elements** including:

### Algorithm Steps to Implement
1. Unpack flat program ID into 4D indices (b, oc, oh, ow)
2. Calculate input window position from output position
3. Loop over input channels
4. Loop over kernel height and width
5. Load input values with boundary checking
6. Load weight values
7. Accumulate products
8. Store final result

### Key Concepts
- **im2col pattern**: Converting convolution to matrix multiplication
- **Sliding window**: Each output element sees a different input window
- **Stride**: Step size when moving the kernel
- **Padding**: Extra border around input
- **Nested loops**: Over channels and kernel dimensions

### Mathematical Background
For each output position (b, oc, oh, ow):
```
output[b, oc, oh, ow] = sum over (ic, kh, kw) of:
    input[b, ic, oh*stride + kh, ow*stride + kw] * weight[oc, ic, kh, kw]
```

## Hints
- This is the most complex kernel - take it step by step
- Each program computes ONE output element
- The nested loops iterate over all contributions to that element
- Boundary checking is crucial for edge cases

## Success Criteria
- Kernel compiles without errors
- Results match PyTorch's conv2d
- Handles padding and stride correctly

## Running the Test
```bash
python kernel.py
```

## Tips
- Start by understanding the indexing math
- Draw a diagram of the convolution operation
- Test with small inputs first
