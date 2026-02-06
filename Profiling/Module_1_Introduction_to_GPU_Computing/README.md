# Module 1: Introduction to GPU Computing and Profiling Concepts

## Overview

Welcome to GPU computing! This module introduces you to the fundamental concepts of GPU architecture, how it differs from CPU computing, and why profiling is crucial for performance optimization.

## Learning Objectives

By the end of this module, you will:
- Understand the basics of GPU architecture and parallel computing
- Know the difference between CPU and GPU computing paradigms
- Recognize why profiling is essential for GPU optimization
- Identify common performance bottlenecks in GPU applications
- Learn the terminology used in GPU profiling

## GPU Architecture Fundamentals

### CPU vs GPU: Key Differences

Traditional CPUs are designed for sequential processing with a few powerful cores optimized for single-threaded performance. GPUs, on the other hand, have hundreds or thousands of smaller, more efficient cores designed to handle multiple tasks simultaneously.

| Aspect | CPU | GPU |
|--------|-----|-----|
| Cores | 2-32 high-performance cores | Hundreds to thousands of smaller cores |
| Architecture | Optimized for sequential tasks | Optimized for parallel tasks |
| Memory | Large cache, low latency | High bandwidth memory |
| Best for | Complex, branching algorithms | Data-parallel computations |

### GPU Programming Model

Modern GPUs follow a Single Instruction Multiple Thread (SIMT) model where many threads execute the same instruction on different data elements simultaneously. This model is ideal for problems that can be broken down into many parallel tasks.

## Why GPU Profiling Matters

GPU applications can suffer from various performance bottlenecks that are difficult to detect without proper profiling tools. Unlike CPU applications, GPU performance depends heavily on:
- Memory bandwidth utilization
- Thread occupancy
- Divergent branching
- Latency hiding mechanisms

Without profiling, optimizing GPU code becomes guesswork, often leading to wasted effort on non-critical sections.

## Common GPU Performance Bottlenecks

1. **Memory Bandwidth Bound**: Kernels spend most of their time waiting for memory operations
2. **Compute Bound**: Kernels are limited by computational throughput
3. **Occupancy Limited**: Not enough active threads to hide memory latency
4. **Divergent Branching**: Threads in a warp take different execution paths
5. **Resource Conflicts**: Shared resources like registers or shared memory are oversubscribed

## GPU Profiling Terminology

- **Kernel**: A function that executes on the GPU
- **Thread**: An individual execution unit within a kernel
- **Warp/Wavefront**: Group of threads that execute in lockstep (32 for NVIDIA, 64 for AMD)
- **Block/Grid**: Hierarchical organization of threads
- **Occupancy**: Ratio of active warps to maximum possible warps
- **Throughput**: Amount of work done per unit time
- **Latency**: Time taken to complete an operation

## Hands-On Exercise

Let's look at a simple example to understand the difference between CPU and GPU computation:

```cpp
// Simple vector addition example

#include <iostream>
#include <vector>
#include <chrono>

// CPU version
void cpu_vector_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
    for(size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const size_t N = 1000000;
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N);

    // Time the CPU version
    auto start = std::chrono::high_resolution_clock::now();
    cpu_vector_add(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU Vector Addition took: " << duration.count() << " microseconds" << std::endl;

    // On a GPU, all additions could happen in parallel
    // This is the power of parallel computing!

    return 0;
}
```

## Key Takeaways

- GPUs excel at parallel processing tasks
- GPU optimization requires different approaches than CPU optimization
- Profiling is essential to identify actual bottlenecks
- Understanding GPU architecture is crucial for effective optimization

## Next Steps

In the next module, we'll set up your environment with the necessary profiling tools and get everything ready for hands-on practice.