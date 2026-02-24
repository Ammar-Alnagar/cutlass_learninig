# Module 01: Tiling Optimization

## Learning Objectives
1. Understand the fundamentals of tiling in GPU kernels
2. Learn to choose optimal block sizes
3. Implement multi-level tiling for matrix operations
4. Measure and analyze performance improvements

## What is Tiling?

**Tiling** (also called **blocking**) is an optimization technique where:
- Data is divided into smaller chunks called "tiles" or "blocks"
- Each thread block processes one tile
- Improves memory coalescing and cache utilization
- Reduces global memory traffic

## Key Concepts

### 1. Block Size Selection
```python
# Common block sizes (powers of 2)
BLOCK_SIZE = 128  # Good for simple kernels
BLOCK_SIZE = 256  # Often optimal
BLOCK_SIZE = 512  # For compute-heavy kernels
BLOCK_SIZE = 1024 # Maximum for many GPUs
```

### 2. 2D Tiling for Matrices
```python
BLOCK_SIZE_M = 64  # Rows per tile
BLOCK_SIZE_N = 64  # Columns per tile
BLOCK_SIZE_K = 32  # Reduction dimension tile
```

### 3. Memory Coalescing
- Threads in a warp should access consecutive memory addresses
- Tiling ensures coalesced access patterns
- Dramatically improves memory bandwidth utilization

## Exercises

### Exercise 1: Find Optimal Block Size
Run the benchmark and identify the best block size for your GPU.

```bash
python tiling_optimization.py
```

### Exercise 2: Experiment with Matrix Multiplication
Try different combinations of BLOCK_SIZE_M, BLOCK_SIZE_N, and BLOCK_SIZE_K.

### Exercise 3: Analyze Performance
- Why do some block sizes perform better?
- How does your GPU's architecture affect optimal tile sizes?

## Performance Tips

1. **Start with 256** for 1D kernels
2. **Use 64x64 tiles** for matrix operations
3. **Benchmark on your hardware** - optimal values vary
4. **Consider occupancy** - larger blocks may reduce parallelism
5. **Align with warp size** (32 for NVIDIA GPUs)

## Hardware Considerations

| GPU Architecture | Optimal Block Size | Warp Size |
|-----------------|-------------------|-----------|
| NVIDIA V100     | 256-512           | 32        |
| NVIDIA A100     | 256-1024          | 32        |
| NVIDIA H100     | 512-1024          | 32        |
| AMD MI250       | 256-512           | 64        |

## Next Steps
After mastering tiling, move to Module 02: Pipelining for advanced optimization.
