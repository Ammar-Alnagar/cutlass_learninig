/**
 * Project 09: Vectorized Copy Kernel
 * 
 * Objective: Implement high-bandwidth memory copy using vectorized operations
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

struct VectorizedCopyConfig {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int VEC_SIZE = 4;  // float4 = 128 bits
};

/**
 * TODO: Implement vectorized copy kernel
 * 
 * Key concepts:
 * - Use float4 for 128-bit loads/stores
 * - Ensure proper alignment
 * - Handle remainder elements
 */
__global__ void vectorized_copy_kernel(const float* src, float* dst, int N) {
    using Config = VectorizedCopyConfig;
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // TODO: Implement vectorized copy
    // 1. Calculate vectorized index
    // 2. Load float4 from source
    // 3. Store float4 to destination
    // 4. Handle remainder elements
    
    (void)src; (void)dst; (void)N;
}

} // namespace cute

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1); } } while(0)

int main() {
    using Config = cute::VectorizedCopyConfig;
    
    const int N = 1000003;  // Non-multiple of 4
    const int block_size = Config::BLOCK_SIZE;
    const int grid_size = 1024;
    
    std::cout << "=== Project 09: Vectorized Copy Kernel ===" << std::endl;
    std::cout << "Elements: " << N << std::endl;
    
    std::vector<float> h_src(N), h_dst(N, 0), h_ref(N);
    for (int i = 0; i < N; i++) { h_src[i] = i * 0.5f; h_ref[i] = h_src[i]; }
    
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    
    cudaMemcpy(d_src, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    cute::vectorized_copy_kernel<<<grid_size, block_size>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (h_dst[i] != h_ref[i]) { pass = false; break; }
    }
    
    std::cout << (pass ? "[PASS] Vectorized Copy" : "[FAIL]") << std::endl;
    
    cudaFree(d_src); cudaFree(d_dst);
    return pass ? 0 : 1;
}
