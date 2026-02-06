# Module 4: Multi-GPU Programming with NCCL

## Overview

In this module, we'll explore how to implement multi-GPU programming patterns using NCCL. We'll combine all the collective operations we've learned into practical applications and discuss strategies for organizing code across multiple GPUs.

## Learning Objectives

By the end of this module, you will:
- Understand multi-GPU programming patterns and paradigms
- Learn how to organize code for multi-GPU execution
- Master the use of NCCL in real-world scenarios
- Understand GPU thread management and synchronization
- Learn how to combine multiple collective operations efficiently

## Multi-GPU Programming Paradigms

### Data Parallelism
In data parallelism, the same model is replicated across multiple GPUs, with each GPU processing a different subset of the data. Gradients are synchronized using AllReduce.

### Model Parallelism
In model parallelism, different parts of the model are placed on different GPUs, requiring operations like AllGather and Scatter to move data between GPUs.

### Hybrid Parallelism
Combines both approaches, using data parallelism within model-parallel groups.

## Key Programming Patterns

### 1. Initialization Pattern
```c
// Initialize CUDA devices
for (int i = 0; i < nGPUs; i++) {
    cudaSetDevice(i);
    // Allocate memory and initialize data on each GPU
}

// Initialize NCCL communicators
ncclComm_t* comms = (ncclComm_t*)malloc(nGPUs * sizeof(ncclComm_t));
ncclCommInitAll(comms, nGPUs, NULL);
```

### 2. Computation-Synchronization Pattern
```c
// Compute on each GPU independently
compute_on_gpu(current_gpu_id);

// Synchronize gradients across all GPUs
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat32, ncclSum, comm, stream);
cudaStreamSynchronize(stream);
```

### 3. Communication Grouping Pattern
```c
ncclGroupStart();
ncclAllReduce(sendbuff1, recvbuff1, count, ncclFloat32, ncclSum, comm, stream);
ncclAllReduce(sendbuff2, recvbuff2, count, ncclFloat32, ncclSum, comm, stream);
ncclGroupEnd();  // Execute all operations together
```

## Memory Management Across Multiple GPUs

### Unified Memory vs. Explicit Memory Management
- **Explicit**: Allocate memory separately on each GPU, copy data explicitly
- **Unified Memory**: Single memory space accessible from all GPUs (requires compatible hardware)

### Memory Pool Strategies
- Pre-allocate large buffers to reduce allocation overhead
- Reuse buffers across iterations
- Consider pinned memory for host-device transfers

## Synchronization Strategies

### Stream Synchronization
Each GPU typically uses its own CUDA stream for overlapping computation and communication:
```c
cudaStream_t stream;
cudaStreamCreate(&stream);

// Launch computation
kernel<<<blocks, threads, 0, stream>>>();

// Launch communication
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat32, ncclSum, comm, stream);

// Synchronize
cudaStreamSynchronize(stream);
```

### Event-Based Synchronization
Use CUDA events for more fine-grained control over synchronization points.

## Combining Multiple Operations

### Operation Sequencing
Operations can be sequenced to achieve complex communication patterns:
1. AllGather to collect data from all GPUs
2. Local computation on concatenated data
3. AllReduce to synchronize results

### Conditional Communication
Sometimes communication patterns depend on runtime conditions, requiring dynamic selection of collective operations.

## Performance Considerations

### Bandwidth Utilization
- Use larger messages to achieve higher bandwidth utilization
- Consider message aggregation to reduce overhead

### Topology Awareness
- NCCL automatically optimizes for GPU interconnect topology
- Understanding your hardware's topology can inform algorithm design

### Latency vs. Throughput
- Small messages are latency-bound
- Large messages are bandwidth-bound

## Error Handling in Multi-GPU Programs

Multi-GPU programs require robust error handling:
- Check return values from all NCCL and CUDA calls
- Handle device-specific errors appropriately
- Implement recovery mechanisms when possible

## Practical Example: Distributed Training Loop

A typical distributed training loop includes:
1. Load batch of data onto each GPU
2. Forward pass on each GPU
3. Backward pass to compute gradients
4. AllReduce to synchronize gradients
5. Update model parameters locally
6. Repeat

## Hands-On Practice

In the code-practice directory, you'll find examples demonstrating:
- Complete multi-GPU training simulation
- Proper resource management across GPUs
- Efficient combination of multiple collective operations
- Error handling in multi-GPU contexts

## Common Pitfalls and Best Practices

### Pitfalls:
- Inconsistent data distribution across GPUs
- Improper synchronization leading to race conditions
- Memory leaks due to improper cleanup
- Ignoring device affinity in multi-process setups

### Best Practices:
- Always validate that all GPUs have consistent data initially
- Use ncclGroupStart/ncclGroupEnd for multiple operations
- Implement proper cleanup in error paths
- Profile memory usage and communication patterns
- Test with different numbers of GPUs

## Next Steps

After mastering multi-GPU programming patterns, Module 5 will focus on performance optimization techniques to maximize the efficiency of your NCCL-based applications.