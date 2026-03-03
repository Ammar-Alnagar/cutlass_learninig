/**
 * Project 01: Vector Add using CuTe
 * 
 * Objective: Implement C = A + B using CuTe tensor abstractions
 * 
 * Instructions:
 * 1. Read the README.md for theory and guidance
 * 2. Complete all TODO sections in this file
 * 3. Build with: make project_01_vector_add
 * 4. Run and verify correctness
 * 
 * Key CuTe Concepts:
 * - Layout: Maps logical indices to physical memory
 * - Tensor: Pointer + Layout = Typed access
 * - Partition: Distributing work among threads
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

/**
 * TODO: Implement the CuTe vector add kernel
 * 
 * Steps:
 * 1. Create a 1D layout representing the vector size
 * 2. Wrap raw pointers (A, B, C) in CuTe tensors
 * 3. Calculate this thread's starting index and stride
 * 4. Loop through assigned elements and compute C[i] = A[i] + B[i]
 * 
 * Hints:
 * - Use make_layout(N) to create a 1D layout
 * - Use make_tensor(make_gmem_ptr(ptr), layout) to create tensors
 * - Use threadIdx.x and blockDim.x for thread indexing
 * - Remember to check bounds!
 */
__global__ void vector_add_cute_kernel(float* A, float* B, float* C, int N) {
    // TODO 1: Create a 1D layout for vectors of size N
    // Hint: auto layout = make_layout(N);
    
    // TODO 2: Wrap raw pointers in CuTe tensors
    // Hint: auto tensor_A = make_tensor(make_gmem_ptr(A), layout);
    //       auto tensor_B = make_tensor(make_gmem_ptr(B), layout);
    //       auto tensor_C = make_tensor(make_gmem_ptr(C), layout);
    
    // TODO 3: Calculate thread index and stride
    // Each thread processes elements at: thread_idx, thread_idx + stride, thread_idx + 2*stride, ...
    // Hint: int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    //       int stride = blockDim.x * gridDim.x;
    
    // TODO 4: Implement the element-wise addition loop
    // for (int i = thread_idx; i < N; i += stride) {
    //     tensor_C(i) = tensor_A(i) + tensor_B(i);
    // }
    
    // Suppress unused parameter warnings (remove after implementing)
    (void)A; (void)B; (void)C; (void)N;
}

} // namespace cute

// ============================================================================
// Host Code - Setup, Launch, and Verification (No changes needed)
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Initialize vector with sequential values
 */
void init_vector(std::vector<float>& vec, float start = 1.0f, float step = 1.0f) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = start + i * step;
    }
}

/**
 * Print first N elements of a vector
 */
void print_vector(const std::vector<float>& vec, size_t n = 5) {
    for (size_t i = 0; i < std::min(n, vec.size()); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

/**
 * Verify result: C should equal A + B
 */
bool verify_result(const std::vector<float>& A, 
                   const std::vector<float>& B,
                   const std::vector<float>& C,
                   float tolerance = 1e-5) {
    for (size_t i = 0; i < A.size(); ++i) {
        float expected = A[i] + B[i];
        if (std::abs(C[i] - expected) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected 
                      << ", got " << C[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Configuration
    const int N = 10000;  // Vector size
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = 32;
    
    std::cout << "=== Project 01: Vector Add with CuTe ===" << std::endl;
    std::cout << "Vector size: " << N << std::endl;
    std::cout << "Launch config: " << GRID_SIZE << " blocks x " << BLOCK_SIZE << " threads" << std::endl;
    std::cout << std::endl;
    
    // Allocate host memory
    std::vector<float> h_A(N), h_B(N), h_C(N);
    
    // Initialize input vectors
    init_vector(h_A, 1.0f, 1.0f);   // [1, 2, 3, ...]
    init_vector(h_B, 10.0f, 10.0f); // [10, 20, 30, ...]
    
    std::cout << "Vector A (first 5): ";
    print_vector(h_A, 5);
    std::cout << "Vector B (first 5): ";
    print_vector(h_B, 5);
    std::cout << std::endl;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    std::cout << "Launching CuTe kernel..." << std::endl;
    cute::vector_add_cute_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    
    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify result
    std::cout << "Verifying result..." << std::endl;
    std::cout << "Vector C (first 5): ";
    print_vector(h_C, 5);
    
    if (verify_result(h_A, h_B, h_C)) {
        std::cout << "\n[PASS] Vector Add: All elements match (max error: 0.000000)" << std::endl;
    } else {
        std::cout << "\n[FAIL] Vector Add: Result mismatch!" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "\n=== Project 01 Complete! ===" << std::endl;
    std::cout << "Next: Try the vectorized access challenge in the README." << std::endl;
    
    return EXIT_SUCCESS;
}
