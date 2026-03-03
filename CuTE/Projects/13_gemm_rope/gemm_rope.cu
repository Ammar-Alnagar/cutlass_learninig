/**
 * Project 13: Fused GEMM + RoPE
 * 
 * Objective: Implement GEMM with fused Rotary Position Embedding
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

struct GemmRoPEConfig {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 8;
    static constexpr int TM = 16;
    static constexpr int TN = 16;
};

/**
 * TODO: Implement fused GEMM + RoPE kernel
 * 
 * Key concepts:
 * - Compute GEMM in registers
 * - Apply RoPE rotation before storing
 * - RoPE operates on pairs of elements
 */
__global__ void gemm_rope_kernel(float* Q_in, float* W_Q, float* Q_out,
                                  int* positions,  // Position for each sequence
                                  int M, int K, int N,  // M=batch*seq, K=hidden, N=hidden
                                  float base_freq) {
    using Config = GemmRoPEConfig;
    
    int tile_m = blockIdx.y * Config::BM;
    int tile_n = blockIdx.x * Config::BN;
    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x;
    
    // TODO: Implement fused GEMM + RoPE
    // 1. Compute GEMM (Q = Q_in × W_Q)
    // 2. Apply RoPE to each element pair
    //    - Get position for this row
    //    - Compute rotation angles
    //    - Apply 2D rotation
    
    (void)Q_in; (void)W_Q; (void)Q_out;
    (void)positions; (void)M; (void)K; (void)N; (void)base_freq;
}

} // namespace cute

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1); } } while(0)

void apply_rope_reference(std::vector<float>& Q, int* positions, int M, int N, float base_freq) {
    for (int i = 0; i < M; i++) {
        int pos = positions ? positions[i] : i;
        for (int d = 0; d < N; d += 2) {
            float freq = 1.0f / powf(base_freq, (float)(d) / N);
            float theta = pos * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            
            float q0 = Q[i * N + d];
            float q1 = Q[i * N + d + 1];
            
            Q[i * N + d] = q0 * cos_t - q1 * sin_t;
            Q[i * N + d + 1] = q0 * sin_t + q1 * cos_t;
        }
    }
}

void gemm_reference(const std::vector<float>& A, const std::vector<float>& B,
                    std::vector<float>& C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

int main() {
    using Config = cute::GemmRoPEConfig;
    const int batch = 4, seq_len = 32, hidden = 128;
    const int M = batch * seq_len;
    const int K = hidden, N = hidden;
    
    std::cout << "=== Project 13: Fused GEMM + RoPE ===" << std::endl;
    std::cout << "Batch: " << batch << ", Seq: " << seq_len << ", Hidden: " << hidden << std::endl;
    
    std::vector<float> h_Q_in(M * K), h_W_Q(K * N), h_Q_out(M * N, 0), h_ref(M * N);
    std::vector<int> h_pos(M);
    
    for (int i = 0; i < M * K; i++) h_Q_in[i] = (i % 100) * 0.01f;
    for (int i = 0; i < K * N; i++) h_W_Q[i] = (i % 100) * 0.01f;
    for (int i = 0; i < M; i++) h_pos[i] = i % seq_len;
    
    // Reference: GEMM then RoPE
    gemm_reference(h_Q_in, h_W_Q, h_ref, M, K, N);
    apply_rope_reference(h_ref, h_pos.data(), M, N, 10000.0f);
    
    float *d_Q_in, *d_W_Q, *d_Q_out;
    int *d_pos;
    CUDA_CHECK(cudaMalloc(&d_Q_in, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_Q, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q_out, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos, M * sizeof(int)));
    
    cudaMemcpy(d_Q_in, h_Q_in.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_Q, h_W_Q.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos.data(), M * sizeof(int), cudaMemcpyHostToDevice);
    
    const dim3 block(Config::TM, Config::TN);
    const dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
    
    cute::gemm_rope_kernel<<<grid, block>>>(d_Q_in, d_W_Q, d_Q_out, d_pos, M, K, N, 10000.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaMemcpy(h_Q_out.data(), d_Q_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float err = 0;
    for (int i = 0; i < M * N; i++) err = fmaxf(err, fabsf(h_Q_out[i] - h_ref[i]));
    std::cout << "Max error: " << err << std::endl;
    std::cout << (err < 0.1f ? "[PASS] Fused GEMM + RoPE" : "[FAIL]") << std::endl;
    
    cudaFree(d_Q_in); cudaFree(d_W_Q); cudaFree(d_Q_out); cudaFree(d_pos);
    return 0;
}
