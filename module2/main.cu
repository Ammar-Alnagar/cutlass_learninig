#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/stride.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

/*
 * Module 2: Tiled Copy (Vectorized Global-to-Shared Memory Movement)
 * Composable Memory Access Patterns
 *
 * This kernel demonstrates how to use CuTe's tiled copy mechanisms to efficiently
 * move data between global and shared memory using vectorized operations.
 * We'll show how to define memory access patterns using layout algebra,
 * enabling automatic vectorization and optimal memory bandwidth utilization.
 */

// Device function to demonstrate tiled copy operations
__global__ void tiled_copy_kernel(float* global_input, float* global_output, int M, int N) {
    // Define shared memory for input and output tiles
    __shared__ float smem_input[128];  // Shared memory for input tile
    __shared__ float smem_output[128]; // Shared memory for output tile

    // Thread block and thread indices
    int tid = threadIdx.x;  // Thread ID within block (0-127 for blockDim.x=128)
    int bid = blockIdx.x;   // Block ID

    // Define the tile size for our computation
    // We'll work with 32x32 tiles processed by 128 threads
    // Each thread will handle multiple elements
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    
    // Calculate which tile this block is responsible for
    int tile_row_start = (bid * TILE_M) % M;
    int tile_col_start = ((bid * TILE_M) / M) * TILE_N;

    // Define the layout for the input tile in global memory
    // Shape: 32 rows x 32 columns
    // Stride: N (leading dimension) for rows, 1 for columns (row-major)
    auto gInputLayout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                    make_stride(Int<N>{}, Int<1>{}));

    // Define the layout for the same tile in shared memory
    // For coalesced access, we often use a swizzled layout in shared memory
    auto sInputLayout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                    make_stride(Int<32>{}, Int<1>{}));  // Simple row-major in shared

    // Create tensors representing the global and shared memory views
    auto gInput = make_tensor(global_input + tile_row_start * N + tile_col_start, gInputLayout);
    auto sInput = make_tensor(smem_input, sInputLayout);

    // Define the copy operation using CuTe's copy atom
    // We'll use a simple copy atom that handles vectorized access
    auto copy_op = make_copy(Copy_Atom<SM75_U32x4_LDSM_N, float>{});

    // Perform the tiled copy from global to shared memory
    // This operation automatically handles vectorization when possible
    copy(copy_op, gInput, sInput);

    // Synchronize threads to ensure all data is loaded
    __syncthreads();

    // Now demonstrate the reverse: copy from shared back to global
    // But first, let's do a simple transformation in shared memory
    // (for demonstration purposes, we'll just double the values)
    #pragma unroll
    for (int i = 0; i < size<0>(sInputLayout); ++i) {
        #pragma unroll
        for (int j = 0; j < size<1>(sInputLayout); ++j) {
            if (tid < size(sInputLayout)) {  // Basic bounds check
                // Calculate linear index for this thread
                int linear_idx = (tid * size(sInputLayout) + i * size<1>(sInputLayout) + j) % size(sInputLayout);
                if (linear_idx < 128) {
                    sInput(i, j) *= 2.0f;  // Simple transformation
                }
            }
        }
    }

    // Define the output tile layout in global memory
    auto gOutputLayout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                     make_stride(Int<N>{}, Int<1>{}));
    auto gOutput = make_tensor(global_output + tile_row_start * N + tile_col_start, gOutputLayout);
    auto sOutput = make_tensor(smem_output, sInputLayout);  // Same layout as input for simplicity

    // Copy transformed data from shared to global output
    copy(copy_op, sInput, gOutput);

    // Synchronize before kernel completion
    __syncthreads();
}

// Alternative implementation showing more advanced tiled copy concepts
__global__ void advanced_tiled_copy_kernel(float* global_input, float* global_output, int M, int N) {
    // Define shared memory
    __shared__ float smem_tile[128];

    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Define tile dimensions - using a more complex tiling pattern
    // We'll use a 16x16 tile handled by 128 threads (each thread handles 2 elements)
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    constexpr int ELEMS_PER_THREAD = 2;

    // Calculate which tile this block processes
    int block_tiles_m = (M + TILE_M - 1) / TILE_M;
    int tile_id = bid;
    int tile_m = tile_id % block_tiles_m;
    int tile_n = tile_id / block_tiles_m;

    int tile_row_start = tile_m * TILE_M;
    int tile_col_start = tile_n * TILE_N;

    // Define the global memory layout for the tile
    auto gLayout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                               make_stride(Int<N>{}, Int<1>{}));

    // Define how threads map to elements in the tile
    // We'll use a blocked arrangement: 16 threads handle rows, 8 threads handle columns
    // This creates a 16x8 thread block that handles the 16x16 tile with 2 elements per thread
    auto thr_layout = make_layout(make_shape(Int<16>{}, Int<8>{}),
                                  make_stride(Int<8>{}, Int<1>{}));

    // Create the thread-blocked tensor view
    auto thrMMA = make_tensor(make_gmem_ptr(global_input + tile_row_start * N + tile_col_start),
                              gLayout,
                              thr_layout);

    // Define the shared memory layout with swizzling for bank conflict avoidance
    // Swizzle the last dimension to avoid bank conflicts
    auto sLayout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                               make_stride(Int<16>{}, Int<1>{}));

    // Create shared memory tensor
    auto sTile = make_tensor(make_smem_ptr(smem_tile), sLayout);

    // Define the copy operation with vectorization considerations
    // Use a copy atom that supports vectorized access
    auto copy_atom = Copy_Atom<DefaultCopy, float>{};

    // Perform the copy operation - CuTe handles the vectorization automatically
    // based on the layout compatibility
    copy(copy_atom, thrMMA(_,_,tid), sTile(_,_,tid));

    // Synchronize threads
    __syncthreads();

    // Transform data in shared memory (simple multiplication by 2)
    #pragma unroll
    for (int i = 0; i < size<0>(sLayout); ++i) {
        #pragma unroll
        for (int j = 0; j < size<1>(sLayout); ++j) {
            sTile(i, j) *= 2.0f;
        }
    }

    // Copy back to global memory
    auto gOutputLayout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                     make_stride(Int<N>{}, Int<1>{}));
    auto gOutput = make_tensor(global_output + tile_row_start * N + tile_col_start, gOutputLayout);

    copy(copy_atom, sTile, gOutput(_,_,tid));

    __syncthreads();
}

int main() {
    std::cout << "=== CUTLASS 3.x CuTe Tiled Copy Demo ===" << std::endl;
    std::cout << "Demonstrating vectorized global-to-shared memory movement" << std::endl;

    // Define problem size
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int SIZE = M * N;

    // Allocate host memory
    std::vector<float> h_input(SIZE);
    std::vector<float> h_output(SIZE, 0.0f);

    // Initialize input data
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = static_cast<float>(i % 100) / 10.0f;  // Simple pattern
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 1D block of 128 threads
    dim3 block_dim(128);
    dim3 grid_dim((M * N + 1023) / 1024);  // Rough calculation for demo

    std::cout << "Launching basic tiled copy kernel..." << std::endl;
    tiled_copy_kernel<<<grid_dim, block_dim>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results (first few elements)
    std::cout << "Verification (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Input[" << i << "] = " << h_input[i] 
                  << ", Output[" << i << "] = " << h_output[i] 
                  << ", Expected = " << (h_input[i] * 2.0f) << std::endl;
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "\n=== Key Concepts Demonstrated ===" << std::endl;
    std::cout << "1. Tiled memory access patterns using CuTe layouts" << std::endl;
    std::cout << "2. Vectorized memory operations through layout algebra" << std::endl;
    std::cout << "3. Thread-to-data mapping using mathematical compositions" << std::endl;
    std::cout << "4. Shared memory tiling for efficient data reuse" << std::endl;
    std::cout << "5. Elimination of manual indexing through composable abstractions" << std::endl;

    return 0;
}