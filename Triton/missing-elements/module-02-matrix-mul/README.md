# Module 02: Matrix Multiplication - Missing Elements Challenge

## Objective
Fix the incomplete 2D matrix multiplication kernel by adding all missing elements.

## What's Missing
The `kernel.py` file has **21+ missing elements** including:

### Core Concepts to Implement
- 2D program ID calculation with swizzling
- Row and column index calculations
- K-dimension loop for accumulation
- `tl.dot()` operation for matrix multiply
- Proper stride handling for 2D tensors
- Boundary masks for non-divisible dimensions

### Key Triton Operations
- `tl.load()` with 2D offsets
- `tl.dot()` for matrix multiplication
- `tl.store()` with 2D indexing
- Accumulator pattern for reduction

## Hints
- Each `# TODO` marks a missing element
- `# HINT` comments provide guidance
- Matrix multiplication requires accumulation over K dimension
- Remember to handle strides for proper memory access

## Success Criteria
- The kernel compiles without errors
- Matrix multiplication produces correct results
- Handles various matrix sizes correctly

## Running the Test
```bash
python kernel.py
```

## Reference
Compare with a working implementation once you've attempted the challenge.
