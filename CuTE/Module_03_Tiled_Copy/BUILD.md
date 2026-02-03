# Build Instructions for CuTe Tiled Copy

## Prerequisites
- CUDA Toolkit 12.x or later
- CuTe library (part of CUTLASS 3.x)
- Compatible GPU (RTX 4060 - sm_89 architecture)
- Modules 01-02 completed and understood

## Compilation Command
```bash
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I/path/to/cutlass/include \
     tiled_copy_basics.cu -o tiled_copy_basics
```

## Important Flags Explained
- `-arch=sm_89`: Target RTX 4060 architecture
- `--expt-relaxed-constexpr`: Enable extended constexpr support required by CuTe
- `-I/path/to/cutlass/include`: Include path to CuTe headers

## Example with Full Path
```bash
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I../../cutlass/include \
     tiled_copy_basics.cu -o tiled_copy_basics
```

## Running the Program
```bash
./tiled_copy_basics
```

## Expected Output
The program will demonstrate:
1. Basic TiledCopy operations and thread cooperation
2. Vectorized 128-bit load patterns
3. cp.async operations for asynchronous memory transfers
4. Memory coalescing strategies and their impact

## Troubleshooting
If you get compilation errors:
1. Ensure CuTe headers are accessible via the include path
2. Verify CUDA toolkit version supports sm_89
3. Check that `--expt-relaxed-constexpr` flag is present
4. Confirm that previous modules compiled successfully (foundational concepts required)