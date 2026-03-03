/**
 * Project 05: FlashInfer-style PagedAttention using CuTe
 * 
 * Objective: Implement attention with variable sequence lengths using page tables
 * 
 * Instructions:
 * 1. Read the README.md for theory and guidance
 * 2. Complete all TODO sections in this file
 * 3. Build with: make project_05_flashinfer
 * 4. Run and verify correctness
 * 
 * Key CuTe Concepts:
 * - Paged memory access patterns
 * - Indirect tensor indexing via page tables
 * - Variable-length sequence handling
 * - Efficient KV cache utilization
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
struct FlashInferConfig {
    static constexpr int PAGE_SIZE = 16;    // Keys/values per page
    static constexpr int MAX_PAGES = 16;    // Max pages per request
    static constexpr int BLOCK_SIZE = 256;  // Threads per block
};

/**
 * Request information structure (mirrored on device)
 */
struct RequestInfo {
    int q_start;       // Query start index in Q tensor
    int seq_len;       // Actual sequence length
    int kv_page_start; // First page index in page table for this request
};

/**
 * TODO: Implement FlashInfer-style PagedAttention kernel
 * 
 * Algorithm (per request):
 * 
 * 1. Get request info from req_info[req_idx]
 * 2. For each query position in this request:
 *    a. Load Q vector
 *    b. Initialize running statistics (m, l, o)
 *    c. For each page in the request's KV cache:
 *       i. Look up physical page_id from page_table
 *       ii. Load K, V vectors from K_pages, V_pages
 *       iii. Compute attention score: s = Q · K / sqrt(d)
 *       iv. Update running max and rescale
 *       v. Compute probability: p = exp(s - m_new)
 *       vi. Update running sum and accumulate output
 *    d. Normalize output: O = o / l
 *    e. Write output for this query position
 * 
 * Hints:
 * - Page table lookup: page_table[req_idx * MAX_PAGES + block_idx]
 * - Physical page offset: offset_in_page = logical_pos % PAGE_SIZE
 * - Logical block: block_idx = logical_pos / PAGE_SIZE
 * - Use -1 in page_table to indicate invalid/empty pages
 */
__global__ void flashinfer_cute_kernel(
    float* Q, float* K_pages, float* V_pages, float* O,
    int* page_table, RequestInfo* req_info,
    int batch, int d, int num_pages) {
    
    using Config = FlashInferConfig;
    
    int req_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // TODO 1: Check bounds
    // if (req_idx >= batch) return;
    
    // TODO 2: Get request info
    // RequestInfo req = req_info[req_idx];
    // int q_start = req.q_start;
    // int seq_len = req.seq_len;
    // int kv_page_start = req.kv_page_start;
    
    // TODO 3: Create layouts for paged K/V cache
    // K_pages shape: [num_pages, PAGE_SIZE, d]
    // auto layout_KV = make_layout(make_shape(num_pages, Config::PAGE_SIZE, d), 
    //                               make_stride(Config::PAGE_SIZE * d, d, 1));
    
    // TODO 4: Create tensors
    // auto tensor_Q = make_tensor(make_gmem_ptr(Q), make_layout(batch * Config::MAX_PAGES, d));
    // auto tensor_K = make_tensor(make_gmem_ptr(K_pages), layout_KV);
    // auto tensor_V = make_tensor(make_gmem_ptr(V_pages), layout_KV);
    // auto tensor_O = make_tensor(make_gmem_ptr(O), make_layout(batch * Config::MAX_PAGES, d));
    
    // TODO 5: Precompute scale factor
    // float scale = rsqrtf((float)d);
    
    // TODO 6: Main computation loop
    // Each thread handles one or more query positions
    // for (int q_pos = tid; q_pos < seq_len; q_pos += blockDim.x) {
    //     // Load Q vector for this position
    //     float q_vec[Config::PAGE_SIZE];  // Simplified: assume d <= PAGE_SIZE
    //     int global_q_idx = q_start + q_pos;
    //     for (int k = 0; k < d; k++) {
    //         q_vec[k] = tensor_Q(global_q_idx, k);
    //     }
    //     
    //     // Initialize running statistics
    //     float m_i = -std::numeric_limits<float>::infinity();
    //     float l_i = 0.0f;
    //     float o_i[Config::PAGE_SIZE] = {0.0f};
    //     
    //     // Loop over pages
    //     int num_blocks = (seq_len + Config::PAGE_SIZE - 1) / Config::PAGE_SIZE;
    //     for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    //         // Look up physical page ID
    //         int page_id = page_table[kv_page_start * Config::MAX_PAGES + block_idx];
    //         if (page_id < 0 || page_id >= num_pages) continue;
    //         
    //         // Determine how many valid keys in this page
    //         int kv_start = block_idx * Config::PAGE_SIZE;
    //         int num_valid = min(Config::PAGE_SIZE, seq_len - kv_start);
    //         
    //         // Load K, V vectors from this page
    //         // Compute attention scores
    //         // Update running statistics
    //         // Accumulate output
    //     }
    //     
    //     // Normalize and write output
    //     if (l_i > 0.0f) {
    //         for (int k = 0; k < d; k++) {
    //             tensor_O(global_q_idx, k) = o_i[k] / l_i;
    //         }
    //     }
    // }
    
    // Suppress unused parameter warnings (remove after implementing)
    (void)Q; (void)K_pages; (void)V_pages; (void)O;
    (void)page_table; (void)req_info;
    (void)batch; (void)d; (void)num_pages;
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

using FlashInferConfig = cute::FlashInferConfig;
using RequestInfo = cute::RequestInfo;

/**
 * Initialize test data with variable sequence lengths
 */
void init_test_data(std::vector<float>& Q, std::vector<float>& K_pages,
                    std::vector<float>& V_pages, std::vector<int>& page_table,
                    std::vector<RequestInfo>& req_info,
                    int batch, int d, int num_pages,
                    const std::vector<int>& seq_lengths) {
    
    int page_size = FlashInferConfig::PAGE_SIZE;
    int max_pages = FlashInferConfig::MAX_PAGES;
    
    // Initialize Q, K, V with deterministic values
    for (size_t i = 0; i < Q.size(); ++i) {
        Q[i] = (float)(i % 100) / 100.0f - 0.5f;
    }
    
    for (size_t i = 0; i < K_pages.size(); ++i) {
        K_pages[i] = (float)((i * 3) % 100) / 100.0f - 0.5f;
        V_pages[i] = (float)((i * 7) % 100) / 100.0f - 0.5f;
    }
    
    // Build page table and request info
    int next_page = 0;
    int q_offset = 0;
    
    for (int b = 0; b < batch; ++b) {
        int seq_len = seq_lengths[b];
        int num_blocks = (seq_len + page_size - 1) / page_size;
        
        req_info[b].q_start = q_offset;
        req_info[b].seq_len = seq_len;
        req_info[b].kv_page_start = b * max_pages;
        
        // Assign pages to this request
        for (int block = 0; block < num_blocks; ++block) {
            page_table[b * max_pages + block] = next_page++;
        }
        
        // Mark unused pages as invalid
        for (int block = num_blocks; block < max_pages; ++block) {
            page_table[b * max_pages + block] = -1;
        }
        
        q_offset += seq_len;
    }
}

/**
 * Reference CPU PagedAttention for verification
 */
void paged_attention_reference(
    const std::vector<float>& Q,
    const std::vector<float>& K_pages,
    const std::vector<float>& V_pages,
    const std::vector<int>& page_table,
    const std::vector<RequestInfo>& req_info,
    std::vector<float>& O,
    int batch, int d, int num_pages) {
    
    int page_size = FlashInferConfig::PAGE_SIZE;
    int max_pages = FlashInferConfig::MAX_PAGES;
    float scale = 1.0f / sqrtf((float)d);
    
    for (int b = 0; b < batch; ++b) {
        int q_start = req_info[b].q_start;
        int seq_len = req_info[b].seq_len;
        int kv_page_start = req_info[b].kv_page_start;
        int num_blocks = (seq_len + page_size - 1) / page_size;
        
        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            int global_q_idx = q_start + q_pos;
            
            // Load Q vector
            std::vector<float> q_vec(d);
            for (int k = 0; k < d; ++k) {
                q_vec[k] = Q[global_q_idx * d + k];
            }
            
            // Compute attention scores with all KVs
            std::vector<float> scores(seq_len);
            float max_score = -std::numeric_limits<float>::infinity();
            
            for (int kv_pos = 0; kv_pos < seq_len; ++kv_pos) {
                // Look up physical location
                int block_idx = kv_pos / page_size;
                int offset_in_page = kv_pos % page_size;
                int page_id = page_table[kv_page_start + block_idx];
                
                if (page_id < 0 || page_id >= num_pages) continue;
                
                // Load K vector
                int base_idx = page_id * page_size * d + offset_in_page * d;
                float score = 0.0f;
                for (int k = 0; k < d; ++k) {
                    score += q_vec[k] * K_pages[base_idx + k];
                }
                score *= scale;
                scores[kv_pos] = score;
                max_score = fmaxf(max_score, score);
            }
            
            // Compute softmax
            float sum_exp = 0.0f;
            for (int kv_pos = 0; kv_pos < seq_len; ++kv_pos) {
                sum_exp += expf(scores[kv_pos] - max_score);
            }
            
            // Apply attention to values
            for (int k = 0; k < d; ++k) {
                float out_val = 0.0f;
                for (int kv_pos = 0; kv_pos < seq_len; ++kv_pos) {
                    float prob = expf(scores[kv_pos] - max_score) / sum_exp;
                    
                    int block_idx = kv_pos / page_size;
                    int offset_in_page = kv_pos % page_size;
                    int page_id = page_table[kv_page_start + block_idx];
                    
                    if (page_id >= 0 && page_id < num_pages) {
                        int base_idx = page_id * page_size * d + offset_in_page * d;
                        out_val += prob * V_pages[base_idx + k];
                    }
                }
                O[global_q_idx * d + k] = out_val;
            }
        }
    }
}

/**
 * Verify PagedAttention output
 */
bool verify_paged_attention(const std::vector<float>& O_gpu,
                            const std::vector<float>& O_cpu,
                            int total_queries, int d,
                            float tolerance = 1e-2f) {
    float max_error = 0.0f;
    
    for (int i = 0; i < total_queries * d; ++i) {
        float error = std::abs(O_gpu[i] - O_cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    
    if (max_error > tolerance) {
        std::cerr << "Max error: " << max_error << " exceeds tolerance " << tolerance << std::endl;
        return false;
    }
    
    std::cout << "Max error: " << max_error << " (tolerance: " << tolerance << ")" << std::endl;
    return true;
}

/**
 * Print output sample
 */
void print_paged_output(const std::vector<float>& O,
                        const std::vector<RequestInfo>& req_info,
                        int batch, int d) {
    for (int b = 0; b < batch; ++b) {
        int q_start = req_info[b].q_start;
        int seq_len = req_info[b].seq_len;
        
        std::cout << "  Request " << b << " (len=" << seq_len << "), first query:" << std::endl;
        std::cout << "    ";
        for (int k = 0; k < std::min(d, 8); ++k) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                      << O[(q_start) * d + k] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    using Config = FlashInferConfig;
    
    // Configuration
    const int batch = 4;
    const int d = 32;
    const int num_pages = 16;
    
    // Variable sequence lengths (no padding!)
    std::vector<int> seq_lengths = {12, 8, 15, 5};
    int total_queries = 0;
    for (int len : seq_lengths) total_queries += len;
    
    // Launch configuration
    const int block_size = Config::BLOCK_SIZE;
    const int grid_size = batch;
    
    std::cout << "=== Project 05: FlashInfer PagedAttention with CuTe ===" << std::endl;
    std::cout << "Batch size: " << batch << std::endl;
    std::cout << "Dimension: " << d << std::endl;
    std::cout << "Page size: " << Config::PAGE_SIZE << std::endl;
    std::cout << "Sequence lengths: ";
    for (int len : seq_lengths) std::cout << len << " ";
    std::cout << std::endl;
    std::cout << "Total queries: " << total_queries << std::endl;
    std::cout << std::endl;
    
    // Allocate host memory
    int max_pages_per_req = Config::MAX_PAGES;
    std::vector<float> h_Q(total_queries * d);
    std::vector<float> h_K_pages(num_pages * Config::PAGE_SIZE * d);
    std::vector<float> h_V_pages(num_pages * Config::PAGE_SIZE * d);
    std::vector<int> h_page_table(batch * max_pages_per_req);
    std::vector<RequestInfo> h_req_info(batch);
    std::vector<float> h_O(total_queries * d, 0.0f);
    
    // Initialize test data
    init_test_data(h_Q, h_K_pages, h_V_pages, h_page_table, h_req_info,
                   batch, d, num_pages, seq_lengths);
    
    std::cout << "Computing CPU reference..." << std::endl;
    std::vector<float> h_O_ref(total_queries * d);
    paged_attention_reference(h_Q, h_K_pages, h_V_pages, h_page_table, h_req_info,
                              h_O_ref, batch, d, num_pages);
    std::cout << "CPU reference complete." << std::endl;
    std::cout << "Reference output (first query of each request):" << std::endl;
    print_paged_output(h_O_ref, h_req_info, batch, d);
    std::cout << std::endl;
    
    // Allocate device memory
    float *d_Q, *d_K_pages, *d_V_pages, *d_O;
    int *d_page_table;
    RequestInfo *d_req_info;
    
    CUDA_CHECK(cudaMalloc(&d_Q, total_queries * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_pages, num_pages * Config::PAGE_SIZE * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_pages, num_pages * Config::PAGE_SIZE * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, total_queries * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_page_table, batch * max_pages_per_req * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_req_info, batch * sizeof(RequestInfo)));
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total_queries * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K_pages, h_K_pages.data(), num_pages * Config::PAGE_SIZE * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_pages, h_V_pages.data(), num_pages * Config::PAGE_SIZE * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_page_table, h_page_table.data(), batch * max_pages_per_req * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_req_info, h_req_info.data(), batch * sizeof(RequestInfo), cudaMemcpyHostToDevice));
    
    // Launch GPU kernel
    std::cout << "Launching CuTe FlashInfer kernel..." << std::endl;
    cute::flashinfer_cute_kernel<<<grid_size, block_size>>>(
        d_Q, d_K_pages, d_V_pages, d_O,
        d_page_table, d_req_info,
        batch, d, num_pages);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, total_queries * d * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify result
    std::cout << "Verifying result..." << std::endl;
    std::cout << "GPU output (first query of each request):" << std::endl;
    print_paged_output(h_O, h_req_info, batch, d);
    std::cout << std::endl;
    
    bool all_pass = true;
    for (int b = 0; b < batch; ++b) {
        int q_start = h_req_info[b].q_start;
        int seq_len = h_req_info[b].seq_len;
        
        bool pass = verify_paged_attention(
            h_O.data() + q_start * d,
            h_O_ref.data() + q_start * d,
            seq_len, d);
        
        std::cout << "Request " << b << " (len=" << seq_len << "): " 
                  << (pass ? "PASS" : "FAIL") << std::endl;
        if (!pass) all_pass = false;
    }
    
    if (all_pass) {
        std::cout << "\n[PASS] FlashInfer: All requests match reference" << std::endl;
    } else {
        std::cout << "\n[FAIL] FlashInfer: Some requests mismatch!" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K_pages));
    CUDA_CHECK(cudaFree(d_V_pages));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_page_table));
    CUDA_CHECK(cudaFree(d_req_info));
    
    std::cout << "\n=== Project 05 Complete! ===" << std::endl;
    std::cout << "Congratulations! You've completed all CuTe projects!" << std::endl;
    
    return EXIT_SUCCESS;
}
