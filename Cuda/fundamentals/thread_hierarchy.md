# CUDA Thread Hierarchy

## Concept Overview

CUDA organizes computation into a hierarchical structure of threads that execute in parallel on the GPU. Understanding this hierarchy is fundamental to writing efficient CUDA programs.

The thread hierarchy consists of three levels:
1. **Grid**: The highest level containing all thread blocks for a kernel launch
2. **Block**: A group of threads that can cooperate through shared memory and synchronization
3. **Thread**: The individual execution unit that runs the kernel code

## Visual Representation

```
Grid (entire kernel launch)
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   ├── Thread 2
│   └── ...
├── Block 1
│   ├── Thread 0
│   ├── Thread 1
│   ├── Thread 2
│   └── ...
├── Block 2
│   ├── Thread 0
│   ├── Thread 1
│   ├── Thread 2
│   └── ...
└── ...
```

## Key Components

### Thread Indexing Variables

CUDA provides built-in variables to identify each thread's position in the hierarchy:

- `threadIdx.x/y/z`: Thread index within a block (0 to blockDim.x/y/z - 1)
- `blockIdx.x/y/z`: Block index within the grid (0 to gridDim.x/y/z - 1)
- `blockDim.x/y/z`: Number of threads in each dimension of a block
- `gridDim.x/y/z`: Number of blocks in each dimension of the grid

### Warp Concept

- A **warp** is a group of 32 consecutive threads that execute in lockstep
- All threads in a warp execute the same instruction at the same time (SIMT - Single Instruction, Multiple Thread)
- Understanding warps is crucial for performance optimization

## Practical Example

```cuda
// Kernel that prints thread and block information
__global__ void printThreadInfo(int n) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        printf("Thread %d in Block %d, Global ID: %d\n", 
               threadIdx.x, blockIdx.x, tid);
    }
}

int main() {
    int n = 256;  // Total number of elements
    dim3 blockSize(64);  // 64 threads per block
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);  // Enough blocks to cover n elements
    
    printThreadInfo<<<gridSize, blockSize>>>(n);
    cudaDeviceSynchronize();
    
    return 0;
}
```

## Mapping Algorithms to Thread Organization

### 1. Element-wise Operations
For operations like vector addition where each element is processed independently:
- Assign one thread per element
- Use 1D indexing: `int idx = blockIdx.x * blockDim.x + threadIdx.x;`

### 2. Matrix Operations
For 2D operations like matrix multiplication:
- Use 2D indexing: `int row = blockIdx.y * blockDim.y + threadIdx.y;`
- `int col = blockIdx.x * blockDim.x + threadIdx.x;`

### 3. Reduction Operations
For operations that combine values (sum, max, min):
- Organize threads hierarchically to perform partial reductions
- Use shared memory for cooperation within blocks

## Best Practices

1. **Choose appropriate block sizes**: Powers of 2 (32, 64, 128, 256, 512) often work best
2. **Consider warp size**: Block sizes should typically be multiples of 32
3. **Balance occupancy**: Too few threads per block reduce occupancy; too many may exhaust resources
4. **Plan for your problem size**: Ensure your grid configuration covers all data elements

## Common Patterns

### 1D Grid-Stride Loop
```cuda
__global__ void kernel(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride) {
        // Process data[i]
        data[i] *= 2.0f;
    }
}
```

### 2D Processing
```cuda
__global__ void matrixKernel(float* matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        int idx = row * width + col;
        // Process matrix[idx]
    }
}
```

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Map any parallel algorithm to GPU thread organization efficiently
- Calculate appropriate grid and block dimensions for different problem sizes
- Understand how thread indexing works and how to use it effectively
- Design thread organizations that maximize parallelism and minimize resource conflicts

## Hands-on Tutorial

See the `thread_hierarchy_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.