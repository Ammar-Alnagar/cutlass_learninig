# Build Instructions for CuTe Collective Mainloops

## Prerequisites
- CUDA Toolkit 12.x or later
- CuTe library (part of CUTLASS 3.x)
- Compatible GPU (RTX 4060 - sm_89 architecture)
- All previous modules (01-05) completed and understood

## Compilation Command
```bash
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I/path/to/cutlass/include \
     producer_consumer_pipeline.cu -o producer_consumer_pipeline
```

## Important Flags Explained
- `-arch=sm_89`: Target RTX 4060 architecture
- `--expt-relaxed-constexpr`: Enable extended constexpr support required by CuTe
- `-I/path/to/cutlass/include`: Include path to CuTe headers

## Example with Full Path
```bash
nvcc -std=c++17 -arch=sm_89 --expt-relaxed-constexpr \
     -I../../cutlass/include \
     producer_consumer_pipeline.cu -o producer_consumer_pipeline
```

## Running the Program
```bash
./producer_consumer_pipeline
```

## Expected Output
The program will demonstrate:
1. Complete producer-consumer pipeline with all stages
2. Collective operations involving thread cooperation
3. Full kernel example integrating all CuTe components
4. Performance profiling concepts and optimization strategies

## Troubleshooting
If you get compilation errors:
1. Ensure CuTe headers are accessible via the include path
2. Verify CUDA toolkit version supports sm_89
3. Check that `--expt-relaxed-constexpr` flag is present
4. Confirm that all previous modules compiled successfully (all concepts required)