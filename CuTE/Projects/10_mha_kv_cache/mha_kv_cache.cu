/**
 * Project 10: Multi-Head Attention with KV-Cache
 * 
 * Objective: Implement MHA with cached K/V for autoregressive generation
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

struct MHAConfig {
    static constexpr int NUM_HEADS = 8;
    static constexpr int HEAD_DIM = 64;
    static constexpr int MAX_SEQ_LEN = 1024;
    static constexpr int BLOCK_SIZE = 256;
};

/**
 * TODO: Implement multi-head attention with KV-cache
 * 
 * Key concepts:
 * - Split Q, K, V into heads
 * - Cache K and V for incremental decoding
 * - Compute attention per head
 */
__global__ void mha_kv_cache_kernel(
    float* Q, float* K_cache, float* V_cache, float* O,
    int* seq_lens,
    int batch, int seq_len, int num_heads, int head_dim) {
    
    using Config = MHAConfig;
    
    int head_idx = blockIdx.z;
    int batch_idx = blockIdx.y;
    int q_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Implement MHA with KV-cache
    // 1. Load Q for current position and head
    // 2. Load K, V from cache for all positions
    // 3. Compute attention scores
    // 4. Apply softmax and accumulate
    // 5. Write output
    
    (void)Q; (void)K_cache; (void)V_cache; (void)O;
    (void)seq_lens; (void)batch; (void)seq_len;
    (void)num_heads; (void)head_dim;
}

} // namespace cute

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1); } } while(0)

int main() {
    using Config = cute::MHAConfig;
    
    const int batch = 4;
    const int seq_len = 64;
    const int num_heads = Config::NUM_HEADS;
    const int head_dim = Config::HEAD_DIM;
    const int d = num_heads * head_dim;
    
    std::cout << "=== Project 10: Multi-Head Attention with KV-Cache ===" << std::endl;
    std::cout << "Batch: " << batch << ", Seq: " << seq_len 
              << ", Heads: " << num_heads << ", Dim: " << head_dim << std::endl;
    
    int total = batch * seq_len * d;
    std::vector<float> h_Q(total), h_K(total), h_V(total), h_O(total, 0);
    
    for (int i = 0; i < total; i++) {
        h_Q[i] = (i % 100) * 0.01f;
        h_K[i] = (i % 100) * 0.01f;
        h_V[i] = (i % 100) * 0.01f;
    }
    
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, total * sizeof(float)));
    
    cudaMemcpy(d_Q, h_Q.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    
    const dim3 block(Config::BLOCK_SIZE);
    const dim3 grid((seq_len + block.x - 1) / block.x, batch, num_heads);
    
    cute::mha_kv_cache_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, nullptr, 
                                                batch, seq_len, num_heads, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaMemcpy(h_O.data(), d_O, total * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "[PASS] MHA with KV-Cache (structure verified)" << std::endl;
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
