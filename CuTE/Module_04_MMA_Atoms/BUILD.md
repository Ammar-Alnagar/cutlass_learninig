# Build Instructions for CuTe MMA Atoms

## Prerequisites
- CUDA Toolkit 12.x or later
- CuTe library (part of CUTLASS 3.x)
- Compatible GPU (RTX 4060 - sm_89 architecture)
- Modules 01-03 completed and understood

## Compilation Command
```bash
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I/path/to/cutlass/include \
     mma_atom_basics.cu -o mma_atom_basics
```

## Important Flags Explained
- `-arch=sm_89`: Target RTX 4060 architecture
- `--expt-relaxed-constexpr`: Enable extended constexpr support required by CuTe
- `-I/path/to/cutlass/include`: Include path to CuTe headers

## Example with Full Path
```bash
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I../../cutlass/include \
     mma_atom_basics.cu -o mma_atom_basics
```

## Running the Program
```bash
./mma_atom_basics
```

## Expected Output
The program will demonstrate:
1. Basic MMA atom operations and configurations
2. Thread-to-Tensor-Core mapping strategies
3. Accumulator register management
4. Mixed precision MMA configurations

## Troubleshooting
If you get compilation errors:
1. Ensure CuTe headers are accessible via the include path
2. Verify CUDA toolkit version supports sm_89
3. Check that `--expt-relaxed-constexpr` flag is present
4. Confirm that previous modules compiled successfully (foundational concepts required)