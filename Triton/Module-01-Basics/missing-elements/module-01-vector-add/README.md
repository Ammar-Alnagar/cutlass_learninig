# Module 01: Vector Addition - Missing Elements Challenge

## Objective
Fix the incomplete Triton kernel by adding all missing elements to make it functional.

## What's Missing
The `kernel.py` file has **14 missing elements** that you need to identify and add:

### Kernel Function (Missing 1-10)
1. Import statement for triton
2. `@triton.jit` decorator
3. Program ID retrieval
4. Block start calculation
5. Offset range creation
6. Index calculation
7. Boundary mask
8. Load operation for x
9. Load operation for y
10. Store operation for output

### Launch Function (Missing 11-14)
11. BLOCK_SIZE selection
12. Output tensor allocation
13. Grid calculation
14. Kernel launch

## Hints
- Look at the comments marked with `# HINT` in the code
- Each `# TODO` marks a missing element
- Reference the basic_vector_add.py in the parent directory if you get stuck

## Success Criteria
- The kernel compiles without errors
- The test passes with matching results
- All 14 missing elements are correctly filled in

## Running the Test
```bash
python kernel.py
```

Expected output: `✓ Vector addition kernel works correctly!`
