# Memory Coalescing

## Concept Overview

Memory coalescing is a critical optimization technique in CUDA that significantly impacts memory bandwidth utilization. When threads in a warp access consecutive memory addresses, the hardware can combine these accesses into fewer, wider transactions, dramatically improving performance.

## Understanding Coalesced Access

### What is Coalescing?

In a coalesced access pattern, consecutive threads in a warp access consecutive memory addresses. Modern GPUs can combine these accesses into a single wide transaction, maximizing memory bandwidth utilization.

### Warp Access Patterns

A warp consists of 32 consecutive threads (thread 0-31, 32-63, etc.). For optimal coalescing:
- Threads 0-31 access addresses 0-31, 32-63, 64-95, etc.
- This creates a single wide transaction instead of 32 individual transactions

## Coalescing Examples

### Good: Coalesced Access
```cuda
// Vector addition - coalesced access
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        // Consecutive threads access consecutive memory addresses
        C[i] = A[i] + B[i];  // Coalesced access
    }
}
```

### Bad: Strided Access
```cuda
// Strided access - poor coalescing
__global__ void stridedAccess(float* data, int N, int stride) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        // If stride is large, threads access non-consecutive addresses
        data[tid * stride] = tid;  // Uncoalesced access
    }
}
```

### Bad: Reverse Access
```cuda
// Reverse access pattern - uncoalesced
__global__ void reverseAccess(float* data, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N - 1) {
        // Threads access memory in reverse order
        data[N - 1 - i] = data[i];  // Uncoalesced access
    }
}
```

## Memory Transaction Sizes

Different GPU architectures have different optimal transaction sizes:

### Modern GPUs (Compute Capability 2.x and later)
- **Worst case**: 32 separate transactions for 32 threads
- **Best case**: 1 transaction for 32 consecutive addresses
- **Optimal alignment**: Addresses should be aligned to transaction size

### Transaction Alignment
- 32-byte, 64-byte, or 128-byte aligned transactions depending on access pattern
- Misaligned accesses may require multiple transactions

## Practical Coalescing Techniques

### 1. Array-of-Structures vs Structure-of-Arrays

**Poor for coalescing:**
```cuda
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

__global__ void processParticles(Particle* particles, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        // Each thread accesses different struct members
        particles[i].x += particles[i].vx;
        particles[i].y += particles[i].vy;
        particles[i].z += particles[i].vz;
    }
}
```

**Better for coalescing:**
```cuda
struct ParticlesSoA {
    float* x, *y, *z;
    float* vx, *vy, *vz;
};

__global__ void processParticlesSoA(ParticlesSoA p, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        // Consecutive threads access consecutive elements in each array
        p.x[i] += p.vx[i];
        p.y[i] += p.vy[i];
        p.z[i] += p.vz[i];
    }
}
```

### 2. Matrix Transposition Example

**Uncoalesced transpose:**
```cuda
__global__ void transposeNaive(float* input, float* output, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < N && y < N) {
        // Input access is coalesced, but output is not
        output[y*N + x] = input[x*N + y];  // Uncoalesced write
    }
}
```

**Coalesced transpose with shared memory:**
```cuda
#define TILE_SIZE 32
__global__ void transposeCoalesced(float* input, float* output, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE+1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Read input in coalesced pattern
    for (int j = 0; j < TILE_SIZE; j += blockDim.y) {
        if (y+j < N && x < N) {
            tile[threadIdx.y+j][threadIdx.x] = input[(y+j)*N + x];
        }
    }
    __syncthreads();
    
    // Write output in coalesced pattern
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    for (int j = 0; j < TILE_SIZE; j += blockDim.y) {
        if (y+j < N && x < N) {
            output[(y+j)*N + x] = tile[threadIdx.x][threadIdx.y+j];
        }
    }
}
```

## Measuring Coalescing Effectiveness

### Bandwidth Utilization
- Use profiling tools to measure achieved vs. peak bandwidth
- Well-coalesced kernels often achieve 80%+ of peak bandwidth
- Poorly coalesced kernels may achieve <20% of peak bandwidth

### Common Metrics
- **Global Memory Throughput**: GB/s achieved vs. peak
- **Coalesced Access Rate**: Percentage of coalesced transactions
- **Memory Efficiency**: Effective bandwidth utilization

## Performance Impact

The performance difference between coalesced and uncoalesced access can be dramatic:
- **Coalesced**: Close to peak memory bandwidth
- **Uncoalesced**: Often 10x slower due to multiple memory transactions

## Expected Knowledge Outcome

After mastering this concept, you should be able to:
- Structure memory access patterns to maximize bandwidth utilization
- Identify and fix uncoalesced access patterns in existing code
- Design algorithms that naturally lead to coalesced access patterns
- Understand the relationship between data layout and memory performance

## Hands-on Tutorial

See the `coalescing_tutorial.cu` file in this directory for practical exercises that reinforce these concepts.