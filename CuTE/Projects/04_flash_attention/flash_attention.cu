/**
 * Project 04: FlashAttention using CuTe
 * 
 * Objective: Implement tiled attention with online softmax
 * 
 * Instructions:
 * 1. Read the README.md for theory and guidance
 * 2. Complete all TODO sections in this file
 * 3. Build with: make project_04_flash_attention
 * 4. Run and verify correctness
 * 
 * Key CuTe Concepts:
 * - Tiled matrix operations
 * - Online softmax for numerical stability
 * - Shared memory for intermediate results
 * - Attention mechanism: softmax(QK^T/sqrt(d)) * V
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

// Configuration
struct FlashAttentionConfig {
    static constexpr int Br = 32;   // Row tile size (queries per block)
    static constexpr int Bc = 32;   // Column tile size (keys/values per tile)
    static constexpr int Bd = 8;    // Dimension tile (for loading Q, K, V)
};

/**
 * TODO: Implement FlashAttention kernel using CuTe
 * 
 * Algorithm (per output tile):
 * 
 * 1. Initialize running statistics:
 *    - m_i = -infinity (running max)
 *    - l_i = 0.0 (running sum)
 *    - o_i = 0.0 (output accumulator)
 * 
 * 2. For each K, V tile (k_start = 0 to seq_len step Bc):
 *    a. Load Q tile [Br x d], K tile [Bc x d], V tile [Bc x d]
 *    b. Compute attention scores: S = Q_tile × K_tile^T [Br x Bc]
 *    c. Scale: S = S / sqrt(d)
 *    d. Compute tile max: m_tile = max(S)
 *    e. Update running max: m_new = max(m_i, m_tile)
 *    f. Rescale accumulator: o_i = o_i * exp(m_i - m_new)
 *    g. Compute probabilities: P = exp(S - m_new)
 *    h. Update running sum: l_new = l_i * exp(m_i - m_new) + sum(P)
 *    i. Accumulate output: o_i = o_i + P × V_tile
 *    j. Update: m_i = m_new, l_i = l_new
 * 
 * 3. Normalize output: O = o_i / l_i
 * 
 * Hints:
 * - Use shared memory for tiles: __shared__ float Q_tile[Br][d];
 * - Use rsqrtf() for 1/sqrt(d)
 * - Use expf() for exponential, fmaxf() for max
 * - Handle edge cases where tile extends beyond sequence
 */
__global__ void flash_attention_cute_kernel(
    float* Q, float* K, float* V, float* O,
    int seq_len, int d) {
    
    using Config = FlashAttentionConfig;
    
    // Each block computes one row tile of output (Br queries)
    int tile_idx = blockIdx.x;
    int q_start = tile_idx * Config::Br;
    
    int tid = threadIdx.x;
    
    // TODO 1: Check bounds - skip if this tile is beyond sequence
    // if (q_start >= seq_len) return;
    
    // TODO 2: Create 2D layouts for input/output matrices
    // auto layout_QKV = make_layout(make_shape(seq_len, d), make_stride(d, 1));
    
    // TODO 3: Create tensors from pointers
    // auto tensor_Q = make_tensor(make_gmem_ptr(Q), layout_QKV);
    // auto tensor_K = make_tensor(make_gmem_ptr(K), layout_QKV);
    // auto tensor_V = make_tensor(make_gmem_ptr(V), layout_QKV);
    // auto tensor_O = make_tensor(make_gmem_ptr(O), layout_QKV);
    
    // TODO 4: Allocate shared memory for tiles
    // __shared__ float Q_tile[Config::Br * Config::Bd];
    // __shared__ float K_tile[Config::Bc * Config::Bd];
    // __shared__ float V_tile[Config::Bc * Config::Bd];
    
    // TODO 5: Initialize running statistics (per-thread private)
    // float m_i = -std::numeric_limits<float>::infinity();  // Running max
    // float l_i = 0.0f;  // Running sum
    // float o_i[Config::Br] = {0.0f};  // Output accumulator (simplified: 1 value per query)
    
    // Precompute scale factor
    // float scale = rsqrtf((float)d);
    
    // TODO 6: Main loop over K, V tiles
    // for (int k_start = 0; k_start < seq_len; k_start += Config::Bc) {
    //     // Phase 1: Load tiles from global to shared memory
    //     // Each thread loads assigned elements
    //     
    //     __syncthreads();
    //     
    //     // Phase 2: Compute attention scores S = Q × K^T
    //     // For simplicity, compute one score per thread
    //     // S[i, j] = sum_k(Q[i, k] * K[j, k])
    //     
    //     // Phase 3: Scale scores by 1/sqrt(d)
    //     
    //     // Phase 4: Find tile max (reduction across scores)
    //     
    //     // Phase 5: Update running max and rescale
    //     // m_new = max(m_i, m_tile)
    //     // o_i = o_i * exp(m_i - m_new)
    //     
    //     // Phase 6: Compute probabilities P = exp(S - m_new)
    //     
    //     // Phase 7: Update running sum and accumulate output
    //     // l_new = l_i * exp(m_i - m_new) + sum(P)
    //     // o_i = o_i + P × V
    //     
    //     // Update running statistics
    //     // m_i = m_new
    //     // l_i = l_new
    //     
    //     __syncthreads();
    // }
    
    // TODO 7: Normalize and write output
    // for (int i = 0; i < Config::Br; i++) {
    //     int row = q_start + i;
    //     if (row < seq_len && l_i > 0.0f) {
    //         // Normalize: O[row] = o_i / l_i
    //         // Note: This is simplified - full implementation outputs d-dimensional vectors
    //     }
    // }
    
    // Suppress unused parameter warnings (remove after implementing)
    (void)Q; (void)K; (void)V; (void)O;
    (void)seq_len; (void)d;
}

} // namespace cute

// ============================================================================
// Host Code - Setup, Launch, and Verification
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
 * Initialize matrices with random values
 */
void init_attention_input(std::vector<float>& Q, std::vector<float>& K, 
                          std::vector<float>& V, int seq_len, int d) {
    for (int i = 0; i < seq_len * d; ++i) {
        Q[i] = (float)(i % 100) / 100.0f - 0.5f;
        K[i] = (float)((i * 3) % 100) / 100.0f - 0.5f;
        V[i] = (float)((i * 7) % 100) / 100.0f - 0.5f;
    }
}

/**
 * Reference CPU attention for verification
 */
void attention_reference(const std::vector<float>& Q,
                         const std::vector<float>& K,
                         const std::vector<float>& V,
                         std::vector<float>& O,
                         int seq_len, int d) {
    float scale = 1.0f / sqrtf((float)d);
    
    for (int i = 0; i < seq_len; ++i) {
        // Compute attention scores for query i
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
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            sum_exp += expf(scores[j] - max_score);
        }
        
        // Apply attention to values
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

/**
 * Verify attention output
 */
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

/**
 * Print output sample
 */
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

int main(int argc, char** argv) {
    using Config = cute::FlashAttentionConfig;
    
    // Configuration - use small sizes for debugging
    const int seq_len = 128;
    const int d = 32;
    
    // Launch configuration
    const int block_size = 256;
    const int grid_size = (seq_len + Config::Br - 1) / Config::Br;
    
    std::cout << "=== Project 04: FlashAttention with CuTe ===" << std::endl;
    std::cout << "Sequence length: " << seq_len << std::endl;
    std::cout << "Dimension: " << d << std::endl;
    std::cout << "Tile size: " << Config::Br << " queries x " << Config::Bc << " keys" << std::endl;
    std::cout << "Launch config: " << grid_size << " blocks x " << block_size << " threads" << std::endl;
    std::cout << std::endl;
    
    // Allocate and initialize host memory
    std::vector<float> h_Q(seq_len * d), h_K(seq_len * d), h_V(seq_len * d);
    std::vector<float> h_O(seq_len * d, 0.0f);
    
    init_attention_input(h_Q, h_K, h_V, seq_len, d);
    
    std::cout << "Computing CPU reference..." << std::endl;
    std::vector<float> h_O_ref(seq_len * d);
    attention_reference(h_Q, h_K, h_V, h_O_ref, seq_len, d);
    std::cout << "CPU reference complete." << std::endl;
    std::cout << "Reference output (first 2 rows, first 8 cols):" << std::endl;
    print_output(h_O_ref, seq_len, d, 2, 8);
    std::cout << std::endl;
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, seq_len * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, seq_len * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, seq_len * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, seq_len * d * sizeof(float)));
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), seq_len * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), seq_len * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), seq_len * d * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch GPU kernel
    std::cout << "Launching CuTe FlashAttention kernel..." << std::endl;
    cute::flash_attention_cute_kernel<<<grid_size, block_size>>>(d_Q, d_K, d_V, d_O, seq_len, d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, seq_len * d * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify result
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
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    
    std::cout << "\n=== Project 04 Complete! ===" << std::endl;
    std::cout << "Next: Try implementing causal masking in the challenges." << std::endl;
    
    return EXIT_SUCCESS;
}
