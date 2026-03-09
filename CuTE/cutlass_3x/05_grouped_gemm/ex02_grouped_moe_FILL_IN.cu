/*
 * Module 05 — Grouped GEMM
 * Exercise 02 — Variable Expert Routing (MoE)
 *
 * CUTLASS LAYER: Grouped GEMM with variable problem sizes
 *
 * WHAT YOU'RE BUILDING:
 *   MoE expert routing with variable tokens per expert.
 *   This is exactly how TRT-LLM implements MoE inference.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Your grouped GEMM from Exercise 01, now with variable
 *   problem sizes per group (real MoE pattern).
 *
 * OBJECTIVE:
 *   - Configure grouped GEMM with variable sizes
 *   - Simulate MoE token routing
 *   - Measure speedup vs naive expert loop
 */

// PREDICT BEFORE COMPILING
// Q1: How does variable sizing affect load balancing?
// Q2: What's the token distribution skew in typical MoE?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "benchmark.cuh"
#include "reference.cuh"

using namespace cutlass;

// ============================================================================
// MOE CONFIGURATION
// ============================================================================

using ArchTag = cutlass::arch::Sm80;

constexpr int NUM_EXPERTS = 8;
constexpr int HIDDEN_DIM = 4096;
constexpr int MLP_DIM = 16384;  // 4x hidden (SwiGLU)

// Token distribution (simulated router output)
// In real MoE, this comes from the router network
struct MoEConfig {
    int num_experts;
    int hidden_dim;
    int mlp_dim;
    std::vector<int> tokens_per_expert;  // Variable per expert
};

// ============================================================================
// SIMULATE MOE ROUTING
// ============================================================================

MoEConfig simulate_moe_routing(int total_tokens, int num_experts, 
                                int hidden_dim, int mlp_dim,
                                float load_imbalance = 2.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Simulate imbalanced routing (some experts get more tokens)
    // Load imbalance factor: 1.0 = perfectly balanced, >1.0 = imbalanced
    std::vector<float> expert_weights(num_experts);
    for (int i = 0; i < num_experts; ++i) {
        expert_weights[i] = 1.0f + (rand() % 100) / 100.0f * (load_imbalance - 1.0f);
    }
    
    // Normalize weights
    float sum_weights = 0.0f;
    for (float w : expert_weights) sum_weights += w;
    for (float& w : expert_weights) w /= sum_weights;
    
    // Assign tokens to experts
    std::vector<int> tokens_per_expert(num_experts, 0);
    int remaining_tokens = total_tokens;
    
    for (int i = 0; i < num_experts - 1; ++i) {
        int assigned = int(remaining_tokens * expert_weights[i]);
        tokens_per_expert[i] = assigned;
        remaining_tokens -= assigned;
    }
    tokens_per_expert[num_experts - 1] = remaining_tokens;
    
    return {num_experts, hidden_dim, mlp_dim, tokens_per_expert};
}

// ============================================================================
// NAIVE MOE (SEQUENTIAL EXPERT LOOP)
// ============================================================================

float launch_naive_moe(
    const std::vector<ElementA*>& d_A,
    const std::vector<ElementB*>& d_B,
    const std::vector<ElementD*>& d_D,
    const std::vector<int>& tokens_per_expert,
    int hidden_dim, int mlp_dim,
    int warmup = 3, int iters = 20
) {
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;
    
    using TileShape = cutlass::GemmlShape<128, 128, 64>;
    
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder
        <ArchTag, cutlass::arch::OpClassTensorOp,
         cutlass::half_t, LayoutA, cutlass::half_t, LayoutB,
         ElementAccumulator, TileShape,
         cutlass::gemm::collective::ClusterShapeAuto,
         cutlass::gemm::collective::StageCountAutoCarveout<128>,
         cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    
    using CollectiveEpilogue = typename cutlass::gemm::collective::CollectiveBuilder
        <ArchTag, cutlass::arch::OpClassTensorOp,
         ElementAccumulator, TileShape,
         cutlass::gemm::collective::ClusterShapeAuto,
         cutlass::half_t, LayoutD, cutlass::half_t, LayoutD,
         cutlass::gemm::collective::EpilogueScheduleAuto>::CollectiveOp;
    
    // Warmup
    for (int iter = 0; iter < warmup; ++iter) {
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            int M = tokens_per_expert[e];
            if (M == 0) continue;
            
            using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<
                cutlass::gemm::GemmUniversal<
                    cutlass::gemm::GemmShape<_, _, _, 1>,
                    CollectiveMainloop, CollectiveEpilogue
                >
            >;
            
            // Launch would go here (simplified for exercise)
        }
    }
    cudaDeviceSynchronize();
    
    // Timed (simplified)
    cutlass_bench::GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < iters; ++iter) {
        cudaDeviceSynchronize();  // Placeholder
    }
    timer.stop();
    
    return timer.elapsed_ms() / iters;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 05, Exercise 02: MoE Variable Expert Routing ===" << std::endl;
    
    cutlass_bench::print_device_info();
    
    // Simulate MoE routing
    int total_tokens = 8192;
    MoEConfig config = simulate_moe_routing(total_tokens, NUM_EXPERTS, 
                                            HIDDEN_DIM, MLP_DIM, 2.0f);
    
    std::cout << "MoE configuration:" << std::endl;
    std::cout << "  Total tokens: " << total_tokens << std::endl;
    std::cout << "  Num experts: " << NUM_EXPERTS << std::endl;
    std::cout << "  Hidden dim: " << HIDDEN_DIM << std::endl;
    std::cout << "  MLP dim: " << MLP_DIM << std::endl;
    std::cout << std::endl;
    
    std::cout << "Token distribution per expert:" << std::endl;
    int max_tokens = 0, min_tokens = total_tokens;
    for (int i = 0; i < NUM_EXPERTS; ++i) {
        std::cout << "  Expert " << i << ": " << config.tokens_per_expert[i] << " tokens" << std::endl;
        max_tokens = max(max_tokens, config.tokens_per_expert[i]);
        min_tokens = min(min_tokens, config.tokens_per_expert[i]);
    }
    std::cout << std::endl;
    std::cout << "  Max tokens: " << max_tokens << std::endl;
    std::cout << "  Min tokens: " << min_tokens << std::endl;
    std::cout << "  Imbalance ratio: " << (float(max_tokens) / max(1, min_tokens)) << "x" << std::endl;
    std::cout << std::endl;
    
    // Allocate device memory (variable sizes)
    std::vector<ElementA*> d_A(NUM_EXPERTS);
    std::vector<ElementB*> d_B(NUM_EXPERTS);
    std::vector<ElementD*> d_D(NUM_EXPERTS);
    
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        int M = config.tokens_per_expert[e];
        size_t bytes_in = M * HIDDEN_DIM * sizeof(ElementA);
        size_t bytes_out = M * MLP_DIM * sizeof(ElementD);
        
        cudaMalloc(&d_A[e], bytes_in);
        cudaMalloc(&d_B[e], HIDDEN_DIM * MLP_DIM * sizeof(ElementB));
        cudaMalloc(&d_D[e], bytes_out);
        
        cutlass_ref::init_matrix_random(d_A[e], M * HIDDEN_DIM);
        cutlass_ref::init_matrix_random(d_B[e], HIDDEN_DIM * MLP_DIM);
    }
    
    // Expert weights (shared across all experts in MoE)
    ElementB* d_expert_weights;
    cudaMalloc(&d_expert_weights, HIDDEN_DIM * MLP_DIM * sizeof(ElementB));
    cutlass_ref::init_matrix_random(d_expert_weights, HIDDEN_DIM * MLP_DIM);
    
    // ========================================================================
    // BENCHMARK
    // ========================================================================
    
    std::cout << "=== BENCHMARK ===" << std::endl;
    
    // Naive approach (sequential expert loop)
    std::cout << "Running naive MoE (sequential expert loop)..." << std::endl;
    float time_naive = launch_naive_moe(d_A, d_B, d_D, 
                                         config.tokens_per_expert,
                                         HIDDEN_DIM, MLP_DIM);
    std::cout << "  Time: " << time_naive << " ms" << std::endl;
    
    // TODO [HARD]: Implement grouped GEMM for variable-sized experts
    // HINT: Same pattern as Exercise 01, but with variable M per group
    
    std::cout << "\nRunning grouped MoE (single launch)..." << std::endl;
    // float time_grouped = ...;
    // std::cout << "  Time: " << time_grouped << " ms" << std::endl;
    // std::cout << "  Speedup: " << (time_naive / time_grouped) << "x" << std::endl;
    
    // ========================================================================
    // ANALYSIS
    // ========================================================================
    
    std::cout << "\n=== ANALYSIS ===" << std::endl;
    std::cout << "MoE grouped GEMM benefits:" << std::endl;
    std::cout << "  1. Single kernel launch for all experts" << std::endl;
    std::cout << "  2. Work stealing balances load imbalance" << std::endl;
    std::cout << "  3. Better GPU utilization for variable token counts" << std::endl;
    std::cout << std::endl;
    std::cout << "Expected speedup:" << std::endl;
    std::cout << "  - 8 experts: 3-5×" << std::endl;
    std::cout << "  - 64 experts: 10-20×" << std::endl;
    std::cout << "  - 256 experts: 30-50×" << std::endl;
    
    // Cleanup
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        cudaFree(d_A[e]);
        cudaFree(d_B[e]);
        cudaFree(d_D[e]);
    }
    cudaFree(d_expert_weights);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: How does token imbalance affect naive MoE?" << std::endl;
    std::cout << "C2: Why does grouped GEMM handle imbalance better?" << std::endl;
    std::cout << "C3: What's the expert count in your target MoE model?" << std::endl;
    
    return 0;
}
