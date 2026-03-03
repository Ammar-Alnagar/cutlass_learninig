/**
 * Project 04: FlashAttention - Reference Solution
 * 
 * Implements tiled attention with online softmax.
 * This is a simplified single-batch version for educational purposes.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <limits>

#include <cute/tensor.hpp>
#include <cute/tensor.hpp>

namespace cute {

struct FlashAttentionConfig {
    static constexpr int Br = 32;
    static constexpr int Bc = 32;
    static constexpr int Bd = 8;
};

__global__ void flash_attention_cute_kernel(
    float* Q, float* K, float* V, float* O,
    int seq_len, int d) {
    
    using Config = FlashAttentionConfig;
    
    // Each block computes one row tile of output (Br queries)
    int tile_idx = blockIdx.x;
    int q_start = tile_idx * Config::Br;
    
    int tid = threadIdx.x;
    
    // Check bounds
    if (q_start >= seq_len) return;
    
    // Number of queries this block handles
    int num_queries = min(Config::Br, seq_len - q_start);
    
    // Create 2D layouts for input/output matrices
    auto layout_QKV = make_layout(make_shape(seq_len, d), make_stride(d, 1));
    
    // Create tensors from pointers
    auto tensor_Q = make_tensor(make_gmem_ptr(Q), layout_QKV);
    auto tensor_K = make_tensor(make_gmem_ptr(K), layout_QKV);
    auto tensor_V = make_tensor(make_gmem_ptr(V), layout_QKV);
    auto tensor_O = make_tensor(make_gmem_ptr(O), layout_QKV);
    
    // Allocate shared memory for tiles
    __shared__ float Q_tile[Config::Br * Config::Bd];
    __shared__ float K_tile[Config::Bc * Config::Bd];
    __shared__ float V_tile[Config::Bc * Config::Bd];
    
    // Precompute scale factor
    float scale = rsqrtf((float)d);
    
    // Per-query running statistics (stored in registers for simplicity)
    // For a full implementation, use arrays indexed by thread's query index
    float m_i[Config::Br];
    float l_i[Config::Br];
    float o_i[Config::Br * Config::Bd];
    
    // Initialize running statistics
    for (int i = 0; i < Config::Br; i++) {
        m_i[i] = -std::numeric_limits<float>::infinity();
        l_i[i] = 0.0f;
        for (int k = 0; k < Config::Bd; k++) {
            o_i[i * Config::Bd + k] = 0.0f;
        }
    }
    
    // Main loop over K, V tiles
    for (int k_start = 0; k_start < seq_len; k_start += Config::Bc) {
        int num_keys = min(Config::Bc, seq_len - k_start);
        
        // Phase 1: Load tiles from global to shared memory
        // Load Q tile
        for (int idx = tid; idx < num_queries * Config::Bd; idx += blockDim.x) {
            int i = idx / Config::Bd;
            int k = idx % Config::Bd;
            int row = q_start + i;
            if (k < d) {
                Q_tile[i * Config::Bd + k] = tensor_Q(row, k);
            } else {
                Q_tile[i * Config::Bd + k] = 0.0f;
            }
        }
        
        // Load K tile
        for (int idx = tid; idx < num_keys * Config::Bd; idx += blockDim.x) {
            int j = idx / Config::Bd;
            int k = idx % Config::Bd;
            int row = k_start + j;
            if (k < d) {
                K_tile[j * Config::Bd + k] = tensor_K(row, k);
            } else {
                K_tile[j * Config::Bd + k] = 0.0f;
            }
        }
        
        // Load V tile
        for (int idx = tid; idx < num_keys * Config::Bd; idx += blockDim.x) {
            int j = idx / Config::Bd;
            int k = idx % Config::Bd;
            int row = k_start + j;
            if (k < d) {
                V_tile[j * Config::Bd + k] = tensor_V(row, k);
            } else {
                V_tile[j * Config::Bd + k] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Phase 2-7: Compute attention for each query this block handles
        for (int qi = 0; qi < num_queries; qi++) {
            // Compute attention scores S = Q[qi] × K^T
            float scores[Config::Bc];
            float m_tile = -std::numeric_limits<float>::infinity();
            
            for (int kj = 0; kj < num_keys; kj++) {
                float score = 0.0f;
                for (int k = 0; k < Config::Bd && k < d; k++) {
                    score += Q_tile[qi * Config::Bd + k] * K_tile[kj * Config::Bd + k];
                }
                score *= scale;
                scores[kj] = score;
                m_tile = fmaxf(m_tile, score);
            }
            
            // Update running max
            float m_new = fmaxf(m_i[qi], m_tile);
            
            // Rescale accumulator
            float rescale = expf(m_i[qi] - m_new);
            for (int k = 0; k < Config::Bd && k < d; k++) {
                o_i[qi * Config::Bd + k] *= rescale;
            }
            
            // Compute probabilities and accumulate
            float l_new = l_i[qi] * rescale;
            
            for (int kj = 0; kj < num_keys; kj++) {
                float p = expf(scores[kj] - m_new);
                l_new += p;
                
                for (int k = 0; k < Config::Bd && k < d; k++) {
                    o_i[qi * Config::Bd + k] += p * V_tile[kj * Config::Bd + k];
                }
            }
            
            // Update running statistics
            m_i[qi] = m_new;
            l_i[qi] = l_new;
        }
        
        __syncthreads();
    }
    
    // Normalize and write output
    for (int idx = tid; idx < num_queries * Config::Bd; idx += blockDim.x) {
        int i = idx / Config::Bd;
        int k = idx % Config::Bd;
        int row = q_start + i;
        
        if (row < seq_len && k < d && l_i[i] > 0.0f) {
            tensor_O(row, k) = o_i[i * Config::Bd + k] / l_i[i];
        }
    }
}

} // namespace cute

// ============================================================================
// Host Code
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

void init_attention_input(std::vector<float>& Q, std::vector<float>& K, 
                          std::vector<float>& V, int seq_len, int d) {
    for (int i = 0; i < seq_len * d; ++i) {
        Q[i] = (float)(i % 100) / 100.0f - 0.5f;
        K[i] = (float)((i * 3) % 100) / 100.0f - 0.5f;
        V[i] = (float)((i * 7) % 100) / 100.0f - 0.5f;
    }
}

void attention_reference(const std::vector<float>& Q,
                         const std::vector<float>& K,
                         const std::vector<float>& V,
                         std::vector<float>& O,
                         int seq_len, int d) {
    float scale = 1.0f / sqrtf((float)d);
    
    for (int i = 0; i < seq_len; ++i) {
        std::vector<float> scores(seq_len);
        float max_score = -std::numeric_limits<float>::infinity();
        
        for (int j = 0; j < seq_len; ++j) {
            float score = 0.0f;
            for (int k = 0; k < d; ++k) {
                score += Q[i * d + k] * K[j * d + k];
            }
            score *= scale;
            scores[j] = score;
            max_score = fmaxf(max_score, score);
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            sum_exp += expf(scores[j] - max_score);
        }
        
        for (int k = 0; k < d; ++k) {
            float out_val = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                float prob = expf(scores[j] - max_score) / sum_exp;
                out_val += prob * V[j * d + k];
            }
            O[i * d + k] = out_val;
        }
    }
}

bool verify_attention(const std::vector<float>& O_gpu,
                      const std::vector<float>& O_cpu,
                      int seq_len, int d,
                      float tolerance = 1e-2f) {
    float max_error = 0.0f;
    int max_error_idx = 0;
    
    for (int i = 0; i < seq_len * d; ++i) {
        float error = std::abs(O_gpu[i] - O_cpu[i]);
        if (error > max_error) {
            max_error = error;
            max_error_idx = i;
        }
    }
    
    if (max_error > tolerance) {
        int row = max_error_idx / d;
        int col = max_error_idx % d;
        std::cerr << "Max error at (" << row << ", " << col << "): "
                  << "expected " << O_cpu[max_error_idx]
                  << ", got " << O_gpu[max_error_idx] << std::endl;
        return false;
    }
    
    std::cout << "Max error: " << max_error << " (tolerance: " << tolerance << ")" << std::endl;
    return true;
}

void print_output(const std::vector<float>& O, int seq_len, int d, 
                  int max_rows = 2, int max_cols = 8) {
    for (int i = 0; i < std::min(seq_len, max_rows); ++i) {
        std::cout << "  Row " << i << ": ";
        for (int j = 0; j < std::min(d, max_cols); ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) 
                      << O[i * d + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    using Config = cute::FlashAttentionConfig;
    
    const int seq_len = 128;
    const int d = 32;
    
    const int block_size = 256;
    const int grid_size = (seq_len + Config::Br - 1) / Config::Br;
    
    std::cout << "=== Project 04: FlashAttention with CuTe (Solution) ===" << std::endl;
    std::cout << "Sequence length: " << seq_len << std::endl;
    std::cout << "Dimension: " << d << std::endl;
    std::cout << std::endl;
    
    std::vector<float> h_Q(seq_len * d), h_K(seq_len * d), h_V(seq_len * d);
    std::vector<float> h_O(seq_len * d, 0.0f);
    
    init_attention_input(h_Q, h_K, h_V, seq_len, d);
    
    std::cout << "Computing CPU reference..." << std::endl;
    std::vector<float> h_O_ref(seq_len * d);
    attention_reference(h_Q, h_K, h_V, h_O_ref, seq_len, d);
    std::cout << "Reference output (first 2 rows, first 8 cols):" << std::endl;
    print_output(h_O_ref, seq_len, d, 2, 8);
    std::cout << std::endl;
    
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, seq_len * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, seq_len * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, seq_len * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, seq_len * d * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), seq_len * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), seq_len * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), seq_len * d * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "Launching CuTe FlashAttention kernel..." << std::endl;
    cute::flash_attention_cute_kernel<<<grid_size, block_size>>>(d_Q, d_K, d_V, d_O, seq_len, d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, seq_len * d * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Verifying result..." << std::endl;
    std::cout << "GPU output (first 2 rows, first 8 cols):" << std::endl;
    print_output(h_O, seq_len, d, 2, 8);
    std::cout << std::endl;
    
    if (verify_attention(h_O, h_O_ref, seq_len, d)) {
        std::cout << "\n[PASS] FlashAttention: Output matches reference" << std::endl;
    } else {
        std::cout << "\n[FAIL] FlashAttention: Result mismatch!" << std::endl;
        return EXIT_FAILURE;
    }
    
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    
    std::cout << "\n=== Solution Complete! ===" << std::endl;
    
    return EXIT_SUCCESS;
}
