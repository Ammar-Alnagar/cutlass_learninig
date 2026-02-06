# Module 6: Troubleshooting and Best Practices

## Overview

In this module, we'll cover common issues encountered when working with NCCL, how to troubleshoot them, and establish best practices for developing robust NCCL applications. This module is essential for deploying NCCL applications in production environments.

## Learning Objectives

By the end of this module, you will:
- Identify and resolve common NCCL issues
- Understand debugging strategies for NCCL applications
- Learn best practices for robust NCCL programming
- Master error handling and recovery techniques
- Understand deployment considerations for production systems

## Common NCCL Issues and Solutions

### 1. Device Visibility Issues
**Problem**: NCCL cannot see all GPUs or sees incorrect topology
**Solution**: 
- Ensure all GPUs are visible to the process (`CUDA_VISIBLE_DEVICES`)
- Check driver compatibility
- Verify sufficient permissions for NVML access

### 2. Memory Allocation Failures
**Problem**: `cudaMalloc` fails during NCCL initialization
**Solution**:
- Ensure sufficient GPU memory is available
- Check for memory leaks in your application
- Consider memory fragmentation

### 3. Communication Timeouts
**Problem**: NCCL operations hang or timeout
**Solution**:
- Increase timeout with `NCCL_TIMEOUT` environment variable
- Check network connectivity for multi-node setups
- Verify all processes are participating correctly

### 4. Topology Detection Problems
**Problem**: NCCL selects suboptimal communication paths
**Solution**:
- Use `NCCL_TOPO_FILE` to provide custom topology
- Set `NCCL_IB_DISABLE=1` to disable InfiniBand if problematic
- Use `NCCL_DEBUG=INFO` to see topology detection

## Debugging Strategies

### 1. Enable NCCL Debugging
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### 2. Use Verbose Output
```bash
export NCCL_DEBUG=WARN  # Less verbose
export NCCL_DEBUG=TRACE # More verbose (performance impact)
```

### 3. Monitor System Resources
- GPU utilization and memory usage
- Network traffic (for multi-node)
- CPU usage during communication phases

### 4. Isolate the Problem
- Test with minimal working example
- Test with different numbers of GPUs
- Test with different message sizes

## Error Handling Best Practices

### 1. Comprehensive Error Checking
Always check return values from NCCL functions:

```c
ncclResult_t ret = ncclAllReduce(sendbuff, recvbuff, count, ncclFloat32, ncclSum, comm, stream);
if (ret != ncclSuccess) {
    fprintf(stderr, "NCCL Error: %s\n", ncclGetErrorString(ret));
    // Handle error appropriately
}
```

### 2. Graceful Degradation
Implement fallback mechanisms when NCCL operations fail:

```c
ncclResult_t result = ncclAllReduce(sendbuff, recvbuff, count, ncclFloat32, ncclSum, comm, stream);
if (result != ncclSuccess) {
    fprintf(stderr, "NCCL failed, falling back to CPU implementation\n");
    cpu_allreduce_fallback(sendbuff_host, recvbuff_host, count);
}
```

### 3. Resource Cleanup
Always clean up resources even when errors occur:

```c
// Always destroy communicators
for (int i = 0; i < nGPUs; i++) {
    if (comms[i] != NULL) {
        ncclCommDestroy(comms[i]);
    }
}
```

## Robust Programming Patterns

### 1. Proper Initialization Sequence
```c
// Initialize CUDA context on each device first
for (int i = 0; i < nGPUs; i++) {
    cudaSetDevice(i);
    cudaFree(0);  // Initialize context
}

// Then initialize NCCL
ncclCommInitAll(comms, nGPUs, NULL);
```

### 2. Consistent State Across GPUs
Ensure all GPUs have consistent data before collective operations:

```c
// Verify data consistency (in debug builds)
#ifdef DEBUG
verify_data_consistency_across_gpus();
#endif
```

### 3. Timeout Handling
Handle timeouts gracefully in distributed environments:

```c
// Set appropriate timeout (in seconds)
setenv("NCCL_TIMEOUT", "300", 1);  // 5 minutes
```

## Production Deployment Considerations

### 1. Resource Management
- Monitor GPU memory usage
- Implement memory pooling to reduce allocation overhead
- Consider memory limits in containerized environments

### 2. Process Coordination
- Ensure all processes start and participate in collectives
- Implement health checks for long-running processes
- Handle process failures gracefully

### 3. Configuration Management
- Use environment variables for tuning
- Document required configurations
- Implement configuration validation

## Performance Monitoring in Production

### 1. Key Metrics to Track
- Communication bandwidth utilization
- Operation latency
- GPU utilization during communication
- Memory usage patterns

### 2. Logging and Alerting
- Log NCCL performance metrics
- Set up alerts for performance degradation
- Monitor for communication errors

## Multi-Node Considerations

### 1. Network Configuration
- Ensure RDMA/InfiniBand is properly configured
- Check firewall rules for required ports
- Verify network interface binding

### 2. Process Management
- Use process managers like SLURM or MPI
- Ensure consistent environment across nodes
- Coordinate GPU assignment across nodes

## Testing Strategies

### 1. Unit Testing
Test individual NCCL operations with known inputs and expected outputs.

### 2. Integration Testing
Test complete workflows with multiple GPUs and operations.

### 3. Stress Testing
Test with maximum message sizes and longest running times.

### 4. Failure Testing
Test error handling and recovery mechanisms.

## Security Considerations

### 1. Privilege Requirements
- NCCL may require elevated privileges for topology detection
- Consider security implications of required permissions

### 2. Data Protection
- Ensure sensitive data is handled appropriately during communication
- Consider encryption for network communication

## Common Anti-Patterns to Avoid

### 1. Inadequate Error Handling
Never ignore NCCL return values in production code.

### 2. Improper Synchronization
Always synchronize streams after NCCL operations when needed.

### 3. Resource Leaks
Always clean up communicators, streams, and memory.

### 4. Assumptions About Hardware
Don't assume specific hardware configurations; make code adaptable.

## Troubleshooting Checklist

Before deploying NCCL applications, verify:
- [ ] All GPUs are visible and accessible
- [ ] Correct drivers and CUDA versions are installed
- [ ] NCCL library is properly installed
- [ ] Network connectivity (for multi-node)
- [ ] Sufficient GPU memory available
- [ ] Proper permissions for topology detection
- [ ] Error handling is implemented
- [ ] Resource cleanup is in place

## Hands-On Practice

In the code-practice directory, you'll find examples demonstrating:
- Error handling patterns
- Debugging techniques
- Robust initialization sequences
- Resource management best practices

## Next Steps

After mastering troubleshooting and best practices, Module 7 will explore real-world applications and case studies, showing how NCCL is used in production systems like deep learning frameworks.