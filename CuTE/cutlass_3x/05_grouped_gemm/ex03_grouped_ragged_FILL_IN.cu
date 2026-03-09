/*
 * Module 05 — Grouped GEMM
 * Exercise 03 — Ragged Batch GEMM
 *
 * CUTLASS LAYER: Grouped GEMM with ragged batches
 *
 * WHAT YOU'RE BUILDING:
 *   Ragged batch GEMM for variable sequence length attention.
 *   Critical for LLM inference with dynamic sequence lengths.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Your grouped GEMM knowledge, applied to ragged tensor pattern.
 *
 * OBJECTIVE:
 *   - Handle ragged batch dimensions
 *   - Process variable sequence lengths efficiently
 *   - Understand pointer array management
 */

// PREDICT BEFORE COMPILING
// Q1: What's the padding waste in naive ragged batch handling?
// Q2: How does ragged batching help LLM inference throughput?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "benchmark.cuh"
#include "reference.cuh"

using namespace cutlass;

// ============================================================================
// RAGGED BATCH CONFIGURATION
// ============================================================================

using ArchTag = cutlass::arch::Sm80;
using ElementAccumulator = float;
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementD = cutlass::half_t;

// Simulated ragged batch (variable sequence lengths)
struct RaggedBatchConfig {
    int batch_size;
    int hidden_dim;
    int output_dim;
    std::vector<int> seq_lengths;  // Variable per sample
};

// ============================================================================
// SIMULATE RAGGED BATCH
// ============================================================================

RaggedBatchConfig create_ragged_batch(int batch_size, int hidden_dim, int output_dim,
                                       int min_seq = 64, int max_seq = 2048) {
    std::vector<int> seq_lengths(batch_size);
    
    // Simulate realistic sequence length distribution
    // (most samples are short, some are long)
    for (int i = 0; i < batch_size; ++i) {
        // Exponential-like distribution
        float r = float(rand()) / RAND_MAX;
        seq_lengths[i] = min_seq + int(r * r * (max_seq - min_seq));
    }
    
    return {batch_size, hidden_dim, output_dim, seq_lengths};
}

// ============================================================================
// PADDING WASTE ANALYSIS
// ============================================================================

void analyze_padding_waste(const RaggedBatchConfig& config, int pad_to = 128) {
    int total_tokens = 0;
    int total_padded = 0;
    
    for (int seq_len : config.seq_lengths) {
        int padded_len = ((seq_len + pad_to - 1) / pad_to) * pad_to;
        total_tokens += seq_len;
        total_padded += padded_len;
    }
    
    float waste_pct = 100.0f * (total_padded - total_tokens) / total_padded;
    
    std::cout << "Padding analysis (pad to " << pad_to << "):" << std::endl;
    std::cout << "  Total tokens: " << total_tokens << std::endl;
    std::cout << "  Total padded: " << total_padded << std::endl;
    std::cout << "  Padding waste: " << waste_pct << "%" << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 05, Exercise 03: Ragged Batch GEMM ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Create ragged batch config
    constexpr int BATCH_SIZE = 32;
    constexpr int HIDDEN_DIM = 4096;
    constexpr int OUTPUT_DIM = 4096;
    
    RaggedBatchConfig config = create_ragged_batch(BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM, 64, 2048);
    
    std::cout << "Ragged batch configuration:" << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Hidden dim: " << HIDDEN_DIM << std::endl;
    std::cout << "  Output dim: " << OUTPUT_DIM << std::endl;
    std::cout << std::endl;
    
    std::cout << "Sequence lengths:" << std::endl;
    int min_seq = config.seq_lengths[0], max_seq = config.seq_lengths[0];
    int sum_seq = 0;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        if (i < 8 || i >= BATCH_SIZE - 4) {
            std::cout << "  Sample " << i << ": " << config.seq_lengths[i] << std::endl;
        } else if (i == 8) {
            std::cout << "  ..." << std::endl;
        }
        min_seq = min(min_seq, config.seq_lengths[i]);
        max_seq = max(max_seq, config.seq_lengths[i]);
        sum_seq += config.seq_lengths[i];
    }
    std::cout << std::endl;
    std::cout << "  Min sequence: " << min_seq << std::endl;
    std::cout << "  Max sequence: " << max_seq << std::endl;
    std::cout << "  Total tokens: " << sum_seq << std::endl;
    std::cout << "  Avg sequence: " << (sum_seq / BATCH_SIZE) << std::endl;
    std::cout << std::endl;
    
    // Analyze padding waste
    analyze_padding_waste(config, 128);
    std::cout << std::endl;
    
    // Allocate device memory (ragged)
    std::vector<ElementA*> d_A(BATCH_SIZE);
    std::vector<ElementB*> d_B(BATCH_SIZE);
    std::vector<ElementD*> d_D(BATCH_SIZE);
    
    for (int i = 0; i < BATCH_SIZE; ++i) {
        int M = config.seq_lengths[i];
        size_t bytes_in = M * HIDDEN_DIM * sizeof(ElementA);
        size_t bytes_out = M * OUTPUT_DIM * sizeof(ElementD);
        
        cudaMalloc(&d_A[i], bytes_in);
        cudaMalloc(&d_B[i], HIDDEN_DIM * OUTPUT_DIM * sizeof(ElementB));
        cudaMalloc(&d_D[i], bytes_out);
        
        cutlass_ref::init_matrix_random(d_A[i], M * HIDDEN_DIM);
        cutlass_ref::init_matrix_random(d_B[i], HIDDEN_DIM * OUTPUT_DIM);
    }
    
    // ========================================================================
    // GROUPED GEMM FOR RAGGED BATCH
    // ========================================================================
    
    std::cout << "=== GROUPED GEMM FOR RAGGED BATCH ===" << std::endl;
    std::cout << "Processing " << BATCH_SIZE << " variable-length sequences..." << std::endl;
    
    // TODO [HARD]: Implement grouped GEMM for ragged batch
    // HINT: Each sequence is a separate GEMM with its own M dimension
    
    /*
    // Build grouped GEMM arguments
    using GroupedGemmKernel = cutlass::gemm::device::GemmGrouped<...>;
    
    GroupedGemmKernel::Arguments args;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        args.problem_sizes.push_back({config.seq_lengths[i], OUTPUT_DIM, HIDDEN_DIM});
        args.ptr_A.push_back(d_A[i]);
        args.ptr_B.push_back(d_B[i]);
        args.ptr_D.push_back(d_D[i]);
        args.lda.push_back(HIDDEN_DIM);
        args.ldb.push_back(OUTPUT_DIM);
        args.ldd.push_back(OUTPUT_DIM);
    }
    
    // Launch grouped GEMM
    GroupedGemmKernel grouped_op;
    // ... (same pattern as previous exercises)
    */
    
    std::cout << "  (Implementation left as exercise)" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // ANALYSIS
    // ========================================================================
    
    std::cout << "=== ANALYSIS ===" << std::endl;
    std::cout << "Ragged batch GEMM benefits:" << std::endl;
    std::cout << "  1. Zero padding waste" << std::endl;
    std::cout << "  2. Single kernel launch for all sequences" << std::endl;
    std::cout << "  3. Natural fit for LLM inference (variable seq len)" << std::endl;
    std::cout << std::endl;
    std::cout << "Use cases:" << std::endl;
    std::cout << "  - LLM prefill (variable input lengths)" << std::endl;
    std::cout << "  - LLM decode (batched generation)" << std::endl;
    std::cout << "  - Encoder-decoder (different seq lens)" << std::endl;
    std::cout << std::endl;
    std::cout << "Expected speedup vs padded batch:" << std::endl;
    std::cout << "  - High variance seq lens: 2-5×" << std::endl;
    std::cout << "  - Low variance seq lens: 1.2-2×" << std::endl;
    
    // Cleanup
    for (int i = 0; i < BATCH_SIZE; ++i) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_D[i]);
    }
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: What's the padding waste for your use case?" << std::endl;
    std::cout << "C2: How does ragged batching help LLM inference?" << std::endl;
    std::cout << "C3: When would padded batch be acceptable?" << std::endl;
    
    return 0;
}
