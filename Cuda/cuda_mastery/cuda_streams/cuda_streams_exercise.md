# CUDA Streams Hands-On Exercise

## Objective
Complete the asynchronous execution kernel using CUDA streams.

## Code to Complete
```cuda
int main() {
    // ... initialization code ...

    // TODO: Create CUDA streams for asynchronous execution
    // Hint: Use cudaStreamCreate()
    /* YOUR STREAM DECLARATIONS AND CREATION HERE */;
    
    // TODO: Launch asynchronous memory copies and kernel executions
    // Hint: Use cudaMemcpyAsync() and kernel<<<...>>>() with stream parameter
    // Example: cudaMemcpyAsync(d_chunk1, h_chunk1, chunk_size_bytes, cudaMemcpyHostToDevice, stream1);
    //          asyncComputation<<<blocks, threads, 0, stream1>>>(d_chunk1, CHUNK_SIZE, 2.0f);
    
    // Launch operations on different streams
    /* YOUR ASYNC OPERATIONS FOR CHUNK 1 */;
    /* YOUR ASYNC OPERATIONS FOR CHUNK 2 */;
    /* YOUR ASYNC OPERATIONS FOR CHUNK 3 */;
    /* YOUR ASYNC OPERATIONS FOR CHUNK 4 */;
    
    // TODO: Synchronize all streams
    // Hint: Use cudaStreamSynchronize() for each stream
    /* YOUR SYNCHRONIZATION CODE */;
    
    // ... result copying and cleanup ...
    
    // TODO: Destroy streams
    // Hint: Use cudaStreamDestroy()
    /* YOUR STREAM DESTRUCTION CODE */;
}
```

## Solution Guidance
- Declare streams: `cudaStream_t stream1, stream2, stream3, stream4;`
- Create streams: `cudaStreamCreate(&stream1);` etc.
- Launch async operations: `cudaMemcpyAsync(d_chunk1, h_chunk1, chunk_size_bytes, cudaMemcpyHostToDevice, stream1);`
- Launch kernels with streams: `asyncComputation<<<gridSize, blockSize, 0, stream1>>>(d_chunk1, CHUNK_SIZE, 2.0f);`
- Synchronize: `cudaStreamSynchronize(stream1);` etc.
- Destroy streams: `cudaStreamDestroy(stream1);` etc.

## Key Concepts Practiced
- CUDA streams for asynchronous execution
- Concurrent memory transfers and computations
- Stream management
- Overlapping computation with memory transfers

## Verification
After completing the code:
1. Ensure the code compiles without errors
2. Check that all operations complete successfully
3. Understand how streams enable concurrent execution
4. Observe potential performance improvements from overlapping operations