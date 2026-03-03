/**
 * Project 09: Vectorized Copy - Reference Solution
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
    static constexpr int VEC_SIZE = 4;
};

__global__ void vectorized_copy_kernel(const float* src, float* dst, int N) {
    using Config = VectorizedCopyConfig;
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x * Config::VEC_SIZE;
    
    // Vectorized copy (float4 = 128 bits)
    int vec_idx = tid / Config::VEC_SIZE;
    int vec_stride = stride / Config::VEC_SIZE;
    
    const float4* src_vec = reinterpret_cast<const float4*>(src);
    float4* dst_vec = reinterpret_cast<float4*>(dst);
    
    for (int i = vec_idx * Config::VEC_SIZE; i + Config::VEC_SIZE - 1 < N; i += stride) {
        dst_vec[i / Config::VEC_SIZE] = src_vec[i / Config::VEC_SIZE];
    }
    
    // Handle remainder
    int rem_start = (N / Config::VEC_SIZE) * Config::VEC_SIZE;
    for (int i = rem_start + tid; i < N; i += stride) {
        dst[i] = src[i];
    }
}

} // namespace cute

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1); } } while(0)

int main() {
    using Config = cute::VectorizedCopyConfig;
    const int N = 1000003;
    
    std::cout << "=== Project 09: Vectorized Copy (Solution) ===" << std::endl;
    
    std::vector<float> h_src(N), h_dst(N, 0), h_ref(N);
    for (int i = 0; i < N; i++) { h_src[i] = i * 0.5f; h_ref[i] = h_src[i]; }
    
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    
    cudaMemcpy(d_src, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    cute::vectorized_copy_kernel<<<1024, Config::BLOCK_SIZE>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool pass = true;
    for (int i = 0; i < N; i++) if (h_dst[i] != h_ref[i]) { pass = false; break; }
    
    std::cout << (pass ? "[PASS] Vectorized Copy" : "[FAIL]") << std::endl;
    
    cudaFree(d_src); cudaFree(d_dst);
    return pass ? 0 : 1;
}
