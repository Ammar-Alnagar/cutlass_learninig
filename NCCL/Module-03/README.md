# Module 3: Advanced Collective Operations - Reduce, AllGather, Scatter

## Overview

In this module, we'll explore three additional important NCCL collective operations: Reduce, AllGather, and Scatter. These operations complement the basic operations we learned in Module 2 and enable more sophisticated distributed computing patterns.

## Learning Objectives

By the end of this module, you will:
- Understand the Reduce operation and its applications
- Understand the AllGather operation and its use cases
- Understand the Scatter operation and its applications
- Learn how to implement these operations using NCCL
- Gain hands-on experience with advanced NCCL programming

## Reduce Operation

### What is Reduce?

Reduce is a collective operation that combines data from all participating GPUs using a reduction operation and stores the result on a designated "root" GPU. Unlike AllReduce, only the root GPU gets the result.

```
GPU 0: [a]     →
GPU 1: [b]     →  Reduce(SUM, root=2) → [a+b+c+d] (only on GPU 2)
GPU 2: [c]     →                      [c] (unchanged on other GPUs)
GPU 3: [d]     →                      [d] (unchanged on other GPUs)
```

### Why is Reduce Important?

Reduce is useful when you need to aggregate data but only one process needs the result. For example, collecting statistics from all workers but only the master needs to make decisions based on them.

## AllGather Operation

### What is AllGather?

AllGather collects data from all participating GPUs and concatenates it, distributing the combined result to all GPUs.

```
GPU 0: [a, b]     → [a, b, c, d, e, f, g, h]
GPU 1: [c, d]     → [a, b, c, d, e, f, g, h] 
GPU 2: [e, f]     → [a, b, c, d, e, f, g, h]
GPU 3: [g, h]     → [a, b, c, d, e, f, g, h]
```

### Why is AllGather Important?

AllGather is useful for combining partitioned data across all processes. For example, gathering predictions from all workers to compute global metrics, or collecting model parameters from sharded models.

## Scatter Operation

### What is Scatter?

Scatter distributes data from a designated "root" GPU to all participating GPUs. The root GPU divides its data among all participants.

```
GPU 0: [a, b, c, d] → [a] (first quarter)
GPU 1: [x, y, z, w] → [b] (second quarter, received from root)
GPU 2: [x, y, z, w] → [c] (third quarter, received from root)  
GPU 3: [x, y, z, w] → [d] (fourth quarter, received from root)
                    (assuming root=0)
```

### Why is Scatter Important?

Scatter is useful for distributing work or data among processes. For example, sending different batches of data to different GPUs for parallel processing.

## Comparison of Collective Operations

| Operation | Input | Output | Use Case |
|-----------|-------|--------|----------|
| AllReduce | Each GPU has data | All GPUs get reduced result | Gradient synchronization |
| Reduce | Each GPU has data | Only root GPU gets result | Aggregation to master |
| Broadcast | One GPU has data | All GPUs get same data | Parameter distribution |
| AllGather | Each GPU has data | All GPUs get concatenated data | Data collection |
| Scatter | Root GPU has data | Each GPU gets portion | Work distribution |

## Memory Layout Considerations

For operations like AllGather and Scatter, understanding memory layout is crucial:
- **AllGather**: Each GPU contributes `sendcount` elements, and receives `sendcount * nDevices` elements total
- **Scatter**: Root GPU has `sendcount * nDevices` elements, each GPU receives `sendcount` elements

## Synchronization Patterns

Remember that NCCL operations are asynchronous. Always synchronize the CUDA stream after NCCL calls:

```c
ncclGroupStart();
ncclAllGather(sendbuff, recvbuff, sendcount, ncclFloat, comm, stream);
// ... other NCCL calls ...
ncclGroupEnd();

cudaStreamSynchronize(stream);  // Important!
```

## Hands-On Practice

In the code-practice directory, you'll find examples demonstrating:
- Reduce operation with different root GPUs
- AllGather combining data from all participants
- Scatter distributing data from root to all
- Proper grouping of operations for efficiency

## Common Pitfalls and Best Practices

### Pitfalls:
- Confusing sendcount vs total element counts
- Forgetting to synchronize after operations
- Incorrect root specification
- Mismatched buffer sizes

### Best Practices:
- Use `ncclGroupStart()` and `ncclGroupEnd()` for multiple operations
- Validate buffer sizes mathematically before operations
- Test with different root values to ensure correctness
- Profile different operation sequences for performance

## Next Steps

After mastering these advanced operations, Module 4 will teach you how to implement multi-GPU programming patterns using NCCL, combining all the operations you've learned into practical applications.