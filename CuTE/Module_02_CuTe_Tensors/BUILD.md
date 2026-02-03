# Build Instructions for CuTe Tensors

## Prerequisites
- CUDA Toolkit 12.x or later
- CuTe library (part of CUTLASS 3.x)
- Compatible GPU (RTX 4060 - sm_89 architecture)
- Module 01 completed and tested

## Compilation Command
```bash
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I/path/to/cutlass/include \
     tensor_basics.cu -o tensor_basics
```

## Important Flags Explained
- `-arch=sm_89`: Target RTX 4060 architecture
- `--expt-relaxed-constexpr`: Enable extended constexpr support required by CuTe
- `-I/path/to/cutlass/include`: Include path to CuTe headers

## Example with Full Path
```bash
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I../../cutlass/include \
     tensor_basics.cu -o tensor_basics
```

## Running the Program
```bash
./tensor_basics
```

## Expected Output
The program will demonstrate:
1. Basic tensor creation from raw pointers and layouts
2. Tensor slicing operations to create sub-tensors
3. Layout transformations and tensor composition
4. Different memory access patterns and their implications

## Troubleshooting
If you get compilation errors:
1. Ensure CuTe headers are accessible via the include path
2. Verify CUDA toolkit version supports sm_89
3. Check that `--expt-relaxed-constexpr` flag is present
4. Confirm that Module 01 compiled successfully (layouts are foundational)