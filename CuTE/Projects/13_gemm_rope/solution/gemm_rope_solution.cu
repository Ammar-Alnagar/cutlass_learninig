/**
 * Project 13: Fused GEMM + RoPE - Reference Solution
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

__global__ void gemm_rope_kernel(float* Q_in, float* W_Q, float* Q_out,
                                  int* positions, int M, int K, int N,
                                  float base_freq) {
    using Config = GemmRoPEConfig;
    
    int tile_m = blockIdx.y * Config::BM;
    int tile_n = blockIdx.x * Config::BN;
    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x;
    
    if (tile_m >= M || tile_n >= N) return;
    
    __shared__ float As[Config::BM * Config::BK];
    __shared__ float Bs[Config::BK * Config::BN];
    
    auto layout_A = make_layout(make_shape(M, K), make_stride(K, 1));
    auto layout_B = make_layout(make_shape(K, N), make_stride(N, 1));
    auto layout_C = make_layout(make_shape(M, N), make_stride(N, 1));
    
    auto tensor_Q_in = make_tensor(make_gmem_ptr(Q_in), layout_A);
    auto tensor_W_Q = make_tensor(make_gmem_ptr(W_Q), layout_B);
    auto tensor_Q_out = make_tensor(make_gmem_ptr(Q_out), layout_C);
    
    constexpr int THREAD_M = Config::BM / Config::TM;
    constexpr int THREAD_N = Config::BN / Config::TN;
    float accum[THREAD_M][THREAD_N] = {{0.0f}};
    
    for (int k = 0; k < K; k += Config::BK) {
        for (int m = 0; m < THREAD_M; m++) {
            int row = tile_m + thread_m * THREAD_M + m;
            for (int bk = 0; bk < Config::BK; bk++) {
                if (row < M && k + bk < K)
                    As[(thread_m * THREAD_M + m) * Config::BK + bk] = tensor_Q_in(row, k + bk);
            }
        }
        for (int n = 0; n < THREAD_N; n++) {
            int col = tile_n + thread_n * THREAD_N + n;
            for (int bk = 0; bk < Config::BK; bk++) {
                if (k + bk < K && col < N)
                    Bs[bk * Config::BN + (thread_n * THREAD_N + n)] = tensor_W_Q(k + bk, col);
            }
        }
        
        __syncthreads();
        
        for (int kk = 0; kk < Config::BK; kk++) {
            for (int m = 0; m < THREAD_M; m++) {
                float a = As[(thread_m * THREAD_M + m) * Config::BK + kk];
                for (int n = 0; n < THREAD_N; n++) {
                    float b = Bs[kk * Config::BN + (thread_n * THREAD_N + n)];
                    accum[m][n] += a * b;
                }
            }
        }
        __syncthreads();
    }
    
    // Apply RoPE and store
    for (int m = 0; m < THREAD_M; m++) {
        int row = tile_m + thread_m * THREAD_M + m;
        if (row >= M) continue;
        
        int pos = positions ? positions[row] : row;
        
        for (int n = 0; n < THREAD_N; n += 2) {
            int col = tile_n + thread_n * THREAD_N + n;
            if (col + 1 >= N) break;
            
            float freq = 1.0f / powf(base_freq, (float)col / N);
            float theta = pos * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            
            float q0 = accum[m][n];
            float q1 = accum[m][n + 1];
            
            tensor_Q_out(row, col) = q0 * cos_t - q1 * sin_t;
            tensor_Q_out(row, col + 1) = q0 * sin_t + q1 * cos_t;
        }
    }
}

} // namespace cute

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1); } } while(0)

void apply_rope(std::vector<float>& Q, int* pos, int M, int N, float base) {
    for (int i = 0; i < M; i++) {
        int p = pos ? pos[i] : i;
        for (int d = 0; d < N; d += 2) {
            float f = 1.0f / powf(base, (float)d / N);
            float t = p * f, c = cosf(t), s = sinf(t);
            float q0 = Q[i * N + d], q1 = Q[i * N + d + 1];
            Q[i * N + d] = q0 * c - q1 * s;
            Q[i * N + d + 1] = q0 * s + q1 * c;
        }
    }
}

void gemm_ref(const std::vector<float>& A, const std::vector<float>& B,
              std::vector<float>& C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

int main() {
    using Config = cute::GemmRoPEConfig;
    const int batch = 4, seq_len = 32, hidden = 128;
    const int M = batch * seq_len, K = hidden, N = hidden;
    
    std::cout << "=== Project 13: Fused GEMM + RoPE (Solution) ===" << std::endl;
    
    std::vector<float> h_Q_in(M * K), h_W_Q(K * N), h_Q_out(M * N, 0), h_ref(M * N);
    std::vector<int> h_pos(M);
    for (int i = 0; i < M * K; i++) h_Q_in[i] = (i % 100) * 0.01f;
    for (int i = 0; i < K * N; i++) h_W_Q[i] = (i % 100) * 0.01f;
    for (int i = 0; i < M; i++) h_pos[i] = i % seq_len;
    
    gemm_ref(h_Q_in, h_W_Q, h_ref, M, K, N);
    apply_rope(h_ref, h_pos.data(), M, N, 10000.0f);
    
    float *d_Q_in, *d_W_Q, *d_Q_out; int *d_pos;
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
    std::cout << (err < 0.1f ? "[PASS]" : "[FAIL]") << std::endl;
    
    cudaFree(d_Q_in); cudaFree(d_W_Q); cudaFree(d_Q_out); cudaFree(d_pos);
    return 0;
}
