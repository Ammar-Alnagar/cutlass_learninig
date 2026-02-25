# Module 06: Collective Mainloops - Exercises

## Overview
This directory contains hands-on exercises to practice Collective Mainloops concepts. Learn to build complete GEMM kernels with producer-consumer pipelines and optimization strategies.

## Building the Exercises

### Prerequisites
- CUDA Toolkit with sm_89 support (or modify CMakeLists.txt for your architecture)
- CUTLASS library with CuTe headers

### Build Instructions

```bash
cd exercises
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Exercises

### Exercise 01: Producer-Consumer Pipeline
**File:** `ex01_producer_consumer.cu`

Learn pipeline fundamentals:
- Producer-consumer separation
- Pipeline stages
- Throughput calculation

**Concepts:** Pipeline, overlap, throughput

---

### Exercise 02: Collective Copy Operations
**File:** `ex02_collective_copy.cu`

Understand cooperative copying:
- Thread cooperation
- TiledCopy abstraction
- Efficiency analysis

**Concepts:** Collective, cooperation, TiledCopy

---

### Exercise 03: Multi-Stage Pipeline
**File:** `ex03_multi_stage_pipeline.cu`

Design multi-stage pipelines:
- 2-stage vs 3-stage vs 4-stage
- Pipeline hazards
- Optimal stage count

**Concepts:** Multi-stage, hazards, optimization

---

### Exercise 04: Thread Block Cooperation
**File:** `ex04_thread_block_cooperation.cu`

Master block-level cooperation:
- Block organization
- Work division
- Cooperation patterns

**Concepts:** Blocks, grid, cooperation

---

### Exercise 05: Mainloop Scheduling
**File:** `ex05_mainloop_scheduling.cu`

Schedule GEMM mainloops:
- K-dimension iteration
- Occupancy considerations
- Optimal scheduling

**Concepts:** Mainloop, scheduling, occupancy

---

### Exercise 06: Double Buffering
**File:** `ex06_double_buffering.cu`

Implement double buffering:
- Ping-pong buffers
- Latency hiding
- Multi-buffering extension

**Concepts:** Double buffering, latency, ping-pong

---

### Exercise 07: Complete GEMM Mainloop
**File:** `ex07_complete_gemm_mainloop.cu`

Integrate all concepts:
- Complete kernel structure
- Data flow
- Performance regimes

**Concepts:** Integration, complete kernel, GEMM

---

### Exercise 08: Performance Optimization
**File:** `ex08_performance_optimization.cu`

Optimize kernel performance:
- Performance metrics
- Roofline model
- Optimization workflow

**Concepts:** TFLOPS, roofline, profiling

---

## Learning Path

1. **Exercise 01** - Producer-consumer basics
2. **Exercise 02** - Collective copy
3. **Exercise 03** - Multi-stage pipeline
4. **Exercise 04** - Block cooperation
5. **Exercise 05** - Mainloop scheduling
6. **Exercise 06** - Double buffering
7. **Exercise 07** - Complete GEMM
8. **Exercise 08** - Performance optimization

## Pipeline Comparison

| Stages | Throughput | Complexity | Use Case |
|--------|------------|------------|----------|
| 1 (Sequential) | 1x | Low | Baseline |
| 2 | ~1.5-2x | Medium | Common |
| 3 | ~2-2.5x | High | Performance |
| 4 | ~2.5-3x | Very High | Maximum |

## Performance Targets

### A100 GPU Targets

| Metric | Target | Peak |
|--------|--------|------|
| FP16 TFLOPS | >250 | 312 |
| Memory Bandwidth | >1000 GB/s | 1555 GB/s |
| Occupancy | >50% | 100% |
| Tensor Core Util | >80% | 100% |

## Key Formulas

### GEMM Complexity
```
FLOPs = 2 × M × N × K
```

### Arithmetic Intensity
```
AI = FLOPs / Bytes
GEMM AI ≈ 2 × K / element_size
```

### Pipeline Throughput
```
Sequential time = n × (load + compute)
Pipelined time = load + n × compute
Speedup = Sequential / Pipelined
```

### Occupancy
```
Occupancy = active_warps / max_warps
Limited by: registers, shared memory, threads
```

## Common Patterns

```cpp
// Complete GEMM mainloop pattern
template<typename Problem, typename Tile>
__global__ void gemm_kernel(Problem problem, Tile tile) {
    // Shared memory
    extern __shared__ uint8_t smem[];
    
    // Accumulator
    Tensor accum = make_accum();
    clear(accum);
    
    // Prologue
    copy_A(A, sA, 0);
    copy_B(B, sB, 0);
    cp_async_fence();
    
    // Mainloop
    for (int k = 1; k < K / TILE_K; ++k) {
        cp_async_wait<0>();
        mma(accum, sA, sB);
        copy_A(A, sA, k);
        copy_B(B, sB, k);
        cp_async_fence();
    }
    
    // Epilogue
    cp_async_wait<0>();
    mma(accum, sA, sB);
    store(C, accum);
}

// Launch
gemm_kernel<<<grid, block, smem_size>>>(problem, tile);
```

## Optimization Checklist

### Memory Access
- [ ] Coalesced global memory access
- [ ] Vectorized loads (128-bit)
- [ ] No bank conflicts
- [ ] Minimal redundant loads

### Compute
- [ ] Tensor Core utilization >80%
- [ ] Pipeline depth 2-4 stages
- [ ] Loop unrolling
- [ ] Minimal overhead

### Resources
- [ ] Occupancy >50%
- [ ] Registers <64 per thread
- [ ] Shared memory <192 KB/SM
- [ ] No register spilling

## Tips for Success

1. **Start simple** - Get correctness first
2. **Profile early** - Find real bottlenecks
3. **Optimize iteratively** - One change at a time
4. **Verify correctness** - Performance means nothing if wrong
5. **Study CUTLASS** - Learn from optimized implementations

## Completion Certificate

Congratulations on completing all 6 modules of CuTe learning!

You have now learned:
1. Layout Algebra - Memory organization
2. CuTe Tensors - Safe memory access
3. Tiled Copy - Efficient data movement
4. MMA Atoms - Tensor Core operations
5. Shared Memory Swizzling - Bank conflict avoidance
6. Collective Mainloops - Complete kernel construction

You are now ready to:
- Read and understand CUTLASS 3.x code
- Write custom CuTe-based kernels
- Optimize GEMM and related operations
- Contribute to high-performance CUDA projects

## Additional Resources

- Module 06 README.md - Concept overview
- `producer_consumer_pipeline.cu` - Reference implementation
- CUTLASS Documentation - https://github.com/NVIDIA/cutlass
- NVIDIA Developer Blog - CUDA optimization tips
- Nsight Compute Documentation - Profiling guide
