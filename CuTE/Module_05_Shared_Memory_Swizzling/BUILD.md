# Build Instructions for CuTe Shared Memory & Swizzling

## Prerequisites
- CUDA Toolkit 12.x or later
- CuTe library (part of CUTLASS 3.x)
- Compatible GPU (RTX 4060 - sm_89 architecture)
- Modules 01-04 completed and understood

## Compilation Command
```bash
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I/path/to/cutlass/include \
     shared_memory_layouts.cu -o shared_memory_layouts
```

## Important Flags Explained
- `-arch=sm_89`: Target RTX 4060 architecture
- `--expt-relaxed-constexpr`: Enable extended constexpr support required by CuTe
- `-I/path/to/cutlass/include`: Include path to CuTe headers

## Example with Full Path
```bash
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I../../cutlass/include \
     shared_memory_layouts.cu -o shared_memory_layouts
```

## Running the Program
```bash
./shared_memory_layouts
```

## Expected Output
The program will demonstrate:
1. Shared memory layout design and access patterns
2. Bank conflict identification and analysis
3. Swizzling techniques to resolve conflicts
4. Practical examples of conflict resolution

## Troubleshooting
If you get compilation errors:
1. Ensure CuTe headers are accessible via the include path
2. Verify CUDA toolkit version supports sm_89
3. Check that `--expt-relaxed-constexpr` flag is present
4. Confirm that previous modules compiled successfully (foundational concepts required)