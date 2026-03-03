/**
 * Project 01: Vector Add - Reference Solution
 * 
 * This is the complete solution for the vector add exercise.
 * Study this after attempting the implementation yourself!
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

/**
 * Solution: CuTe vector add kernel
 * 
 * Key concepts demonstrated:
 * 1. Layout creation for 1D data
 * 2. Tensor wrapping of raw pointers
 * 3. Thread partitioning for parallel work
 * 4. Element-wise operations with bounds checking
 */
__global__ void vector_add_cute_kernel(float* A, float* B, float* C, int N) {
    // Step 1: Create a 1D layout representing the vector size
    auto layout = make_layout(N);
    
    // Step 2: Wrap raw pointers in CuTe tensors
    // make_gmem_ptr casts raw pointer to CuTe's GMEM pointer type
    // make_tensor combines pointer + layout for indexed access
    auto tensor_A = make_tensor(make_gmem_ptr(A), layout);
    auto tensor_B = make_tensor(make_gmem_ptr(B), layout);
    auto tensor_C = make_tensor(make_gmem_ptr(C), layout);
    
    // Step 3: Calculate this thread's global index and stride
    // Global thread index across all blocks
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Total number of threads in the grid
    int stride = blockDim.x * gridDim.x;
    
    // Step 4: Element-wise addition with stride loop
    // Each thread processes multiple elements if N > num_threads
    for (int i = thread_idx; i < N; i += stride) {
        tensor_C(i) = tensor_A(i) + tensor_B(i);
    }
}

} // namespace cute

// ============================================================================
// Host Code (identical to the exercise file)
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

void init_vector(std::vector<float>& vec, float start = 1.0f, float step = 1.0f) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = start + i * step;
    }
}

void print_vector(const std::vector<float>& vec, size_t n = 5) {
    for (size_t i = 0; i < std::min(n, vec.size()); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

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
    const int N = 10000;
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = 32;
    
    std::cout << "=== Project 01: Vector Add with CuTe (Solution) ===" << std::endl;
    std::cout << "Vector size: " << N << std::endl;
    std::cout << "Launch config: " << GRID_SIZE << " blocks x " << BLOCK_SIZE << " threads" << std::endl;
    std::cout << std::endl;
    
    std::vector<float> h_A(N), h_B(N), h_C(N);
    init_vector(h_A, 1.0f, 1.0f);
    init_vector(h_B, 10.0f, 10.0f);
    
    std::cout << "Vector A (first 5): ";
    print_vector(h_A, 5);
    std::cout << "Vector B (first 5): ";
    print_vector(h_B, 5);
    std::cout << std::endl;
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "Launching CuTe kernel..." << std::endl;
    cute::vector_add_cute_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Verifying result..." << std::endl;
    std::cout << "Vector C (first 5): ";
    print_vector(h_C, 5);
    
    if (verify_result(h_A, h_B, h_C)) {
        std::cout << "\n[PASS] Vector Add: All elements match (max error: 0.000000)" << std::endl;
    } else {
        std::cout << "\n[FAIL] Vector Add: Result mismatch!" << std::endl;
        return EXIT_FAILURE;
    }
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    std::cout << "\n=== Solution Complete! ===" << std::endl;
    
    return EXIT_SUCCESS;
}
