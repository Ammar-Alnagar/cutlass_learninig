#include <iostream>
#include <vector>
#include <type_traits>
#include <chrono>
#include <cmath>

// Module 5: CUDA and GPU Programming Fundamentals Practice
// Hands-on tutorial for GPU programming concepts necessary for CUTLASS
// TODO: Implement each exercise to learn CUDA fundamentals

/*
 * EXERCISE 1: GPU ARCHITECTURE BASICS
 * Understanding GPU vs CPU architecture and SIMT model
 * 
 * LEARNING OBJECTIVES:
 * - Understand the fundamental differences between GPU and CPU architectures
 * - Learn about the SIMT (Single Instruction, Multiple Thread) execution model
 * - See why GPUs are well-suited for parallelizable workloads like matrix operations
 */
// TODO: Implement this function to explain GPU architecture basics
void exercise_gpu_architecture_basics() {
    std::cout << "\n=== Exercise 1: GPU Architecture Basics ===" << std::endl;

    // TODO: Fill in the characteristics of CPUs vs GPUs
    std::cout << "GPU vs CPU characteristics:" << std::endl;
    std::cout << "CPU:" << std::endl;
    std::cout << "- Few powerful cores (4-64)" << std::endl;
    std::cout << "- High clock speed (3-5 GHz)" << std::endl;
    std::cout << "- Complex control logic" << std::endl;
    std::cout << "- Large caches" << std::endl;
    std::cout << "- Optimized for sequential code" << std::endl;

    std::cout << "\nGPU:" << std::endl;
    std::cout << "- Thousands of simpler cores (1000s)" << std::endl;
    std::cout << "- Lower clock speed (1-2 GHz)" << std::endl;
    std::cout << "- Simpler control logic per core" << std::endl;
    std::cout << "- Smaller caches per core" << std::endl;
    std::cout << "- Optimized for parallel execution" << std::endl;

    // TODO: Explain the SIMT model and how it differs from SIMD
    std::cout << "\nSIMT (Single Instruction, Multiple Thread) model:" << std::endl;
    std::cout << "- Unlike SIMD, SIMT executes the same instruction on different data" << std::endl;
    std::cout << "- Threads in a warp execute in lockstep" << std::endl;
    std::cout << "- Allows for more flexible control flow than SIMD" << std::endl;
}

/*
 * EXERCISE 2: CUDA PROGRAMMING MODEL
 * Understanding the basic CUDA programming model
 */
void exercise_cuda_programming_model() {
    std::cout << "\n=== Exercise 2: CUDA Programming Model ===" << std::endl;

    std::cout << "CUDA Function Type Qualifiers:" << std::endl;
    std::cout << "__global__: Runs on device, called from host" << std::endl;
    std::cout << "__device__: Runs on device, called from device" << std::endl;
    std::cout << "__host__: Runs on host, called from host (default)" << std::endl;
    std::cout << "__host__ __device__: Runs on both host and device" << std::endl;

    // Simulated example of CUDA-like structure
    std::cout << "\nSimulated CUDA kernel execution:" << std::endl;
    std::cout << "1. Allocate memory on host and device" << std::endl;
    std::cout << "2. Copy data from host to device" << std::endl;
    std::cout << "3. Launch kernel on GPU" << std::endl;
    std::cout << "4. Wait for kernel completion" << std::endl;
    std::cout << "5. Copy results back to host" << std::endl;
    std::cout << "6. Free allocated memory" << std::endl;
}

/*
 * EXERCISE 3: THREAD ORGANIZATION
 * Understanding grid, block, and thread hierarchy
 */
void exercise_thread_organization() {
    std::cout << "\n=== Exercise 3: Thread Organization ===" << std::endl;

    // Simulate thread indexing calculations
    int blockSize = 256;
    int gridSize = 4;
    int blockId = 2;
    int threadId = 100;

    int globalThreadId = blockId * blockSize + threadId;

    std::cout << "Thread organization example:" << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;
    std::cout << "Grid size: " << gridSize << std::endl;
    std::cout << "Block ID: " << blockId << std::endl;
    std::cout << "Thread ID within block: " << threadId << std::endl;
    std::cout << "Global thread ID: " << globalThreadId << std::endl;

    // Simulate warp calculations
    int warpSize = 32;
    int warpId = threadId / warpSize;
    int laneId = threadId % warpSize;

    std::cout << "\nWarp calculations:" << std::endl;
    std::cout << "Warp size: " << warpSize << std::endl;
    std::cout << "Warp ID within block: " << warpId << std::endl;
    std::cout << "Lane ID within warp: " << laneId << std::endl;

    // 2D grid example
    std::cout << "\n2D grid organization:" << std::endl;
    std::cout << "For matrix operations, 2D blocks are often used" << std::endl;
    std::cout << "dim3 blockSize(16, 16) creates 16x16 = 256 threads per block" << std::endl;
    std::cout << "This is efficient for processing 2D data like matrices" << std::endl;
}

/*
 * EXERCISE 4: MEMORY HIERARCHIES
 * Understanding different types of GPU memory
 */
void exercise_memory_hierarchies() {
    std::cout << "\n=== Exercise 4: Memory Hierarchies ===" << std::endl;

    std::cout << "GPU Memory Types:" << std::endl;
    std::cout << "\n1. Global Memory:" << std::endl;
    std::cout << "   - Largest capacity" << std::endl;
    std::cout << "   - Slowest access" << std::endl;
    std::cout << "   - Accessible by all threads" << std::endl;
    std::cout << "   - Cached in L1/L2 cache" << std::endl;

    std::cout << "\n2. Shared Memory:" << std::endl;
    std::cout << "   - Faster than global memory" << std::endl;
    std::cout << "   - Shared among threads in a block" << std::endl;
    std::cout << "   - Limited capacity (~48KB-164KB per SM)" << std::endl;
    std::cout << "   - Programmable cache" << std::endl;

    std::cout << "\n3. Registers:" << std::endl;
    std::cout << "   - Fastest memory" << std::endl;
    std::cout << "   - Private to each thread" << std::endl;
    std::cout << "   - Limited capacity (~65K 32-bit registers per SM)" << std::endl;

    std::cout << "\n4. Constant Memory:" << std::endl;
    std::cout << "   - Read-only" << std::endl;
    std::cout << "   - Cached" << std::endl;
    std::cout << "   - Optimized for broadcast to threads" << std::endl;

    std::cout << "\n5. Texture Memory:" << std::endl;
    std::cout << "   - Cached" << std::endl;
    std::cout << "   - Optimized for spatial locality" << std::endl;
    std::cout << "   - Hardware interpolation" << std::endl;

    std::cout << "\nMemory access performance hierarchy:" << std::endl;
    std::cout << "Registers > Shared Memory > L1 Cache > L2 Cache > Global Memory" << std::endl;
}

/*
 * EXERCISE 5: COALESCED MEMORY ACCESS
 * Understanding memory access patterns and coalescing
 */
void exercise_coalesced_memory_access() {
    std::cout << "\n=== Exercise 5: Coalesced Memory Access ===" << std::endl;

    std::cout << "Coalesced Access Pattern:" << std::endl;
    std::cout << "When consecutive threads access consecutive memory locations," << std::endl;
    std::cout << "the memory transactions are combined efficiently." << std::endl;
    std::cout << "\nGood: Threads 0,1,2,3 access memory addresses 0,1,2,3" << std::endl;

    std::cout << "\nUncoalesced Access Pattern:" << std::endl;
    std::cout << "When threads access scattered memory locations," << std::endl;
    std::cout << "multiple memory transactions are required." << std::endl;
    std::cout << "Bad: Threads 0,1,2,3 access memory addresses 0,100,200,300" << std::endl;

    // Simulate memory access patterns
    int numThreads = 32; // Warp size
    std::cout << "\nExample with " << numThreads << " threads (one warp):" << std::endl;

    std::cout << "Coalesced access - consecutive addresses:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << "Thread " << i << " accesses address " << i << std::endl;
    }

    std::cout << "\nUncoalesced access - strided addresses:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << "Thread " << i << " accesses address " << (i * 4) << std::endl;
    }

    std::cout << "\nFor optimal performance:" << std::endl;
    std::cout << "- Organize data access so consecutive threads access consecutive memory" << std::endl;
    std::cout << "- Use appropriate data layouts (AOS vs SOA)" << std::endl;
    std::cout << "- Consider memory padding to avoid bank conflicts" << std::endl;
}

/*
 * EXERCISE 6: OCCUPANCY AND PERFORMANCE
 * Understanding occupancy and performance factors
 */
void exercise_occupancy_performance() {
    std::cout << "\n=== Exercise 6: Occupancy and Performance ===" << std::endl;

    std::cout << "Occupancy refers to the ratio of active warps to the maximum" << std::endl;
    std::cout << "possible warps on a streaming multiprocessor (SM)." << std::endl;

    std::cout << "\nFactors affecting occupancy:" << std::endl;
    std::cout << "1. Number of registers used per thread" << std::endl;
    std::cout << "2. Amount of shared memory used per block" << std::endl;
    std::cout << "3. Number of threads per block" << std::endl;

    // Simulate occupancy calculation
    int registersPerThread = 32;
    int sharedMemPerBlock = 16384; // 16KB
    int threadsPerBlock = 256;

    std::cout << "\nExample calculation:" << std::endl;
    std::cout << "Registers per thread: " << registersPerThread << std::endl;
    std::cout << "Shared memory per block: " << sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Threads per block: " << threadsPerBlock << std::endl;

    // These are theoretical limits (actual values vary by GPU architecture)
    int maxRegistersPerSM = 65536; // 64K registers per SM
    int maxSharedMemPerSM = 163840; // 160KB shared memory per SM
    int maxThreadsPerSM = 2048; // 2048 threads per SM

    int blocksLimitedByRegisters = maxRegistersPerSM / (registersPerThread * threadsPerBlock);
    int blocksLimitedBySharedMem = maxSharedMemPerSM / sharedMemPerBlock;
    int blocksLimitedByThreads = maxThreadsPerSM / threadsPerBlock;

    int theoreticalMaxBlocks = std::min({blocksLimitedByRegisters,
                                        blocksLimitedBySharedMem,
                                        blocksLimitedByThreads});

    std::cout << "\nTheoretical max blocks per SM limited by:" << std::endl;
    std::cout << "Registers: " << blocksLimitedByRegisters << std::endl;
    std::cout << "Shared memory: " << blocksLimitedBySharedMem << std::endl;
    std::cout << "Threads: " << blocksLimitedByThreads << std::endl;
    std::cout << "Actual max blocks per SM: " << theoreticalMaxBlocks << std::endl;

    int actualActiveWarps = (theoreticalMaxBlocks * threadsPerBlock) / 32;
    int maxPossibleWarps = maxThreadsPerSM / 32;
    float occupancy = static_cast<float>(actualActiveWarps) / maxPossibleWarps;

    std::cout << "Occupancy: " << (occupancy * 100) << "%" << std::endl;

    std::cout << "\nOptimization strategies:" << std::endl;
    std::cout << "- Minimize register usage per thread" << std::endl;
    std::cout << "- Use appropriate block sizes (multiples of warp size)" << std::endl;
    std::cout << "- Balance shared memory usage" << std::endl;
    std::cout << "- Consider using occupancy calculator tools" << std::endl;
}

/*
 * EXERCISE 7: MEMORY BANDWIDTH OPTIMIZATION
 * Understanding bank conflicts and memory optimization
 */
void exercise_memory_bandwidth_optimization() {
    std::cout << "\n=== Exercise 7: Memory Bandwidth Optimization ===" << std::endl;

    std::cout << "Shared Memory Banks:" << std::endl;
    std::cout << "- GPU shared memory is divided into banks (typically 32 banks)" << std::endl;
    std::cout << "- Each bank can service one access per cycle" << std::endl;
    std::cout << "- Multiple accesses to the same bank cause serialization" << std::endl;

    std::cout << "\nBank Conflict Examples:" << std::endl;
    std::cout << "Good: Threads access different banks" << std::endl;
    std::cout << "Bad: Multiple threads access the same bank simultaneously" << std::endl;

    std::cout << "\nAvoiding bank conflicts:" << std::endl;
    std::cout << "1. Use appropriate indexing patterns" << std::endl;
    std::cout << "2. Add padding to arrays (e.g., [32][33] instead of [32][32])" << std::endl;
    std::cout << "3. Restructure algorithms to avoid conflicting access patterns" << std::endl;

    std::cout << "\nMemory optimization techniques:" << std::endl;
    std::cout << "- Use coalesced access patterns" << std::endl;
    std::cout << "- Maximize reuse of data in shared memory" << std::endl;
    std::cout << "- Minimize global memory transactions" << std::endl;
    std::cout << "- Use appropriate data structures for GPU access patterns" << std::endl;
}

/*
 * HANDS-ON CHALLENGES
 * Try implementing these CUDA concepts in practice
 */

// Challenge 1: Matrix Multiplication Simulation
void simulate_matrix_multiplication() {
    std::cout << "\nChallenge 1 - Matrix Multiplication Simulation:" << std::endl;

    int M = 64, N = 64, K = 64; // Matrix dimensions
    std::cout << "Simulating C[MxN] = A[MxK] * B[KxN]" << std::endl;
    std::cout << "Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    std::cout << "\nMemory access considerations:" << std::endl;
    std::cout << "A: accessed by [row][k] - good for row-major if processing by rows" << std::endl;
    std::cout << "B: accessed by [k][col] - good for row-major if processing by columns" << std::endl;
    std::cout << "C: accessed by [row][col] - write once, good pattern" << std::endl;

    std::cout << "\nOptimization strategies:" << std::endl;
    std::cout << "1. Use tiling to improve data reuse" << std::endl;
    std::cout << "2. Ensure coalesced access for all matrices" << std::endl;
    std::cout << "3. Use shared memory to stage tiles" << std::endl;
}

// Challenge 2: Reduction Operation Simulation
void simulate_reduction_operation() {
    std::cout << "\nChallenge 2 - Reduction Operation Simulation:" << std::endl;

    int n = 1024;
    std::cout << "Simulating parallel reduction of " << n << " elements" << std::endl;

    std::cout << "\nNaive approach complexity: O(n) time, O(log n) steps" << std::endl;
    std::cout << "GPU-optimized approach:" << std::endl;
    std::cout << "1. Use shared memory to store intermediate results" << std::endl;
    std::cout << "2. Perform reduction within a block" << std::endl;
    std::cout << "3. Use multiple blocks for large arrays" << std::endl;
    std::cout << "4. Handle final reduction on CPU or with another kernel" << std::endl;

    std::cout << "\nWarp-level optimizations:" << std::endl;
    std::cout << "1. Use warp shuffle operations for final 32 elements" << std::endl;
    std::cout << "2. Avoid divergence in reduction tree" << std::endl;
    std::cout << "3. Use proper synchronization (__syncthreads)" << std::endl;
}

// Challenge 3: Memory Access Pattern Analysis
void analyze_memory_access_patterns() {
    std::cout << "\nChallenge 3 - Memory Access Pattern Analysis:" << std::endl;

    std::cout << "Scenario: Processing a 2D array in different ways" << std::endl;

    std::cout << "\nRow-major traversal (good coalescing):" << std::endl;
    std::cout << "for(row) for(col) process(array[row][col]);" << std::endl;
    std::cout << "Consecutive threads access consecutive memory -> GOOD" << std::endl;

    std::cout << "\nColumn-major traversal (poor coalescing):" << std::endl;
    std::cout << "for(col) for(row) process(array[row][col]);" << std::endl;
    std::cout << "Consecutive threads access memory with stride -> BAD" << std::endl;

    std::cout << "\nSolutions:" << std::endl;
    std::cout << "1. Change algorithm to use row-major access" << std::endl;
    std::cout << "2. Use tiling to improve access locality" << std::endl;
    std::cout << "3. Transpose data if access pattern is fixed" << std::endl;
}

// CUTLASS Connection Example
void cutlass_connection_example() {
    std::cout << "\nCUTLASS Connection:" << std::endl;
    std::cout << "CUTLASS uses these CUDA fundamentals:" << std::endl;
    std::cout << "1. Tiled GEMM operations for optimal memory access" << std::endl;
    std::cout << "2. Tensor cores for accelerated math operations" << std::endl;
    std::cout << "3. Careful memory layout and swizzling for coalescing" << std::endl;
    std::cout << "4. Warp-level primitives for efficient computation" << std::endl;
    std::cout << "5. Template metaprogramming for compile-time optimization" << std::endl;
}

int main() {
    std::cout << "Module 5: CUDA and GPU Programming Fundamentals Practice - Hands-on Tutorial" << std::endl;

    // Run all exercises
    exercise_gpu_architecture_basics();
    exercise_cuda_programming_model();
    exercise_thread_organization();
    exercise_memory_hierarchies();
    exercise_coalesced_memory_access();
    exercise_occupancy_performance();
    exercise_memory_bandwidth_optimization();

    // Try the challenges
    std::cout << "\n=== Challenge Solutions ===" << std::endl;
    simulate_matrix_multiplication();
    simulate_reduction_operation();
    analyze_memory_access_patterns();

    cutlass_connection_example();

    std::cout << "\nSummary:" << std::endl;
    std::cout << "This module covered essential CUDA concepts that form the foundation" << std::endl;
    std::cout << "for understanding and working with high-performance GPU libraries like CUTLASS." << std::endl;
    std::cout << "Key takeaways include memory hierarchy, coalescing, occupancy, and optimization strategies." << std::endl;

    return 0;
}