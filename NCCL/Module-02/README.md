# Module 2: Basic Collective Operations - AllReduce, Broadcast

## Overview

In this module, we'll explore the most commonly used NCCL collective operations: AllReduce and Broadcast. These operations form the foundation of many distributed computing algorithms, especially in deep learning training.

## Learning Objectives

By the end of this module, you will:
- Understand the AllReduce operation and its applications
- Understand the Broadcast operation and its use cases
- Learn how to implement these operations using NCCL
- Gain hands-on experience with basic NCCL programming

## AllReduce Operation

### What is AllReduce?

AllReduce is a collective operation that combines data from all participating GPUs using a reduction operation (like sum, max, min) and then broadcasts the result to all GPUs.

```
GPU 0: [a]     →
GPU 1: [b]     →  AllReduce(SUM) → [a+b+c+d] 
GPU 2: [c]     →                 [a+b+c+d]
GPU 3: [d]     →                 [a+b+c+d]
                (on all GPUs)
```

### Why is AllReduce Important?

AllReduce is crucial in distributed machine learning for gradient synchronization. During training, each GPU computes gradients on its portion of the data. AllReduce combines these gradients efficiently across all GPUs.

### AllReduce Reduction Operations

NCCL supports several reduction operations:
- `ncclSum`: Sum of all values
- `ncclProd`: Product of all values  
- `ncclMax`: Maximum value
- `ncclMin`: Minimum value
- `ncclAvg`: Average (not directly supported, computed as sum followed by division)

## Broadcast Operation

### What is Broadcast?

Broadcast sends data from one designated "root" GPU to all other participating GPUs.

```
GPU 0: [data]  → [data] (unchanged if root=0)
GPU 1: [x]     → [data] (received from root)
GPU 2: [y]     → [data] (received from root) 
GPU 3: [z]     → [data] (received from root)
```

### Why is Broadcast Important?

Broadcast is used to distribute initial parameters, hyperparameters, or any data that needs to be identical across all GPUs in a distributed system.

## NCCL Data Types

NCCL supports various data types:
- `ncclInt8`, `ncclUint8`
- `ncclInt32`, `ncclUint32` 
- `ncclInt64`, `ncclUint64`
- `ncclFloat16`, `ncclFloat32`, `ncclFloat64`
- `ncclBfloat16` (for newer GPUs)

## Basic NCCL Programming Pattern

Most NCCL programs follow this pattern:

1. **Initialize NCCL**: Call `ncclCommInitAll()` or `ncclCommInitRank()`
2. **Allocate GPU memory**: Use `cudaMalloc()` for input/output buffers
3. **Execute collective operation**: Call the appropriate NCCL function
4. **Synchronize**: Use `cudaStreamSynchronize()` to ensure completion
5. **Clean up**: Free memory and destroy communicators

## Hands-On Practice

In the code-practice directory, you'll find examples demonstrating:
- AllReduce with different data types and operations
- Broadcast from different root GPUs
- Proper error handling in NCCL programs

## Common Pitfalls and Best Practices

### Pitfalls:
- Not synchronizing CUDA streams after NCCL operations
- Mismatched data types between host and device
- Incorrect buffer sizes
- Improper communicator initialization

### Best Practices:
- Always check return values from NCCL functions
- Synchronize streams after collective operations
- Use consistent data types across all participating GPUs
- Initialize communicators properly before use

## Next Steps

After mastering these basic operations, Module 3 will introduce more advanced collectives like Reduce, AllGather, and Scatter, expanding your understanding of distributed computing patterns.