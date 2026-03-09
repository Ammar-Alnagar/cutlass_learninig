/*
 * Module 03 — Warp-Specialized GEMM
 * Exercise 03 — Warp-Specialized Attention (FA3 Architecture)
 *
 * CUTLASS LAYER: Warp-specialized CollectiveMma for attention
 *
 * WHAT YOU'RE BUILDING:
 *   Flash Attention 3 (FA3) architecture using warp specialization.
 *   This is the exact pattern used in production FA3 implementations
 *   on H100.
 *
 * CuTe FOUNDATION THIS BUILDS ON:
 *   Your manual attention implementation from CuTe, now with
 *   warp-specialized GEMM for QK^T and PV matmuls.
 *
 * OBJECTIVE:
 *   - Configure warp specialization for attention pattern
 *   - Understand producer/consumer split for Q, K, V loads
 *   - Implement fused attention using warp-specialized collectives
 */

// PREDICT BEFORE COMPILING
// Q1: How does attention's QK^T + PV pattern affect warp split?
// Q2: Why does FA3 need more producer warps than dense GEMM?

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "benchmark.cuh"
#include "reference.cuh"
#include "roofline.cuh"

using namespace cutlass;

// ============================================================================
// SETUP — Attention problem definition
// ============================================================================

using ArchTag = cutlass::arch::Sm90;

// Attention dimensions
constexpr int B = 8;       // Batch size
constexpr int H = 64;      // Heads
constexpr int S = 4096;    // Sequence length
constexpr int D = 128;     // Head dimension

// GEMM equivalent (for QK^T): M=B*H*S, N=S, K=D
constexpr int M_gemm = B * H * S;
constexpr int N_gemm = S;
constexpr int K_gemm = D;

using ElementQKV = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementOutput = cutlass::bfloat16_t;

using LayoutQ = cutlass::layout::RowMajor;           // [B*H*S, D]
using LayoutK = cutlass::layout::ColumnMajor;        // [D, S] for K^T
using LayoutV = cutlass::layout::RowMajor;           // [B*H*S, D]
using LayoutO = cutlass::layout::RowMajor;           // [B*H*S, D]

// ============================================================================
// ATTENTION MATMUL PATTERNS
// ============================================================================
// FA3 uses two GEMMs:
//   1. QK^T: [B*H*S, D] × [D, S] → [B*H*S, S] (attention scores)
//   2. PV:   [B*H*S, S] × [B*H*S, D] → [B*H*S, D] (output)
//
// Warp specialization for attention:
//   - More producer warps needed (3 matrices: Q, K, V)
//   - Consumer warps handle QK^T, softmax, PV in sequence
//   - Shared memory holds Q, K, V tiles simultaneously

// ============================================================================
// WARP SPLIT FOR ATTENTION
// ============================================================================
// Attention requires different warp split than dense GEMM:
//
// Dense GEMM (2 matrices):
//   - Producer: 4 warps (load A, B)
//   - Consumer: 28 warps (MMA)
//
// Attention (3 matrices + softmax):
//   - Producer: 8 warps (load Q, K, V)
//   - Consumer: 24 warps (QK^T MMA, softmax, PV MMA)
//
// The extra producer warps handle the third matrix load.

// TODO [HARD]: Configure warp-specialized collectives for QK^T GEMM
// HINT: Use more producer warps for 3-matrix attention pattern

/*
using TileShapeQK = cutlass::GemmlShape<64, 128, 64>;  // Smaller M tile for attention

using CollectiveMainloopQK = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementQKV, LayoutQ,
   ElementQKV, LayoutK,
   ElementAccumulator,
   TileShapeQK,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<230400>,
   cutlass::gemm::collective::KernelScheduleWarpSpecialized
   >::CollectiveOp;
*/

struct CollectiveMainloopQK {};

// TODO [HARD]: Configure warp-specialized collectives for PV GEMM
// HINT: PV has different shape (M×S × S×D → M×D)

/*
using TileShapePV = cutlass::GemmlShape<128, 128, 128>;  // Larger K tile for PV

using CollectiveMainloopPV = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator, cutlass::layout::RowMajor,  // P (attention scores, FP32)
   ElementQKV, LayoutV,
   ElementAccumulator,
   TileShapePV,
   cutlass::gemm::collective::ClusterShapeAuto,
   cutlass::gemm::collective::StageCountAutoCarveout<230400>,
   cutlass::gemm::collective::KernelScheduleWarpSpecialized
   >::CollectiveOp;
*/

struct CollectiveMainloopPV {};

// ============================================================================
// COLLECTIVE EPILOGUE — With softmax fusion
// ============================================================================
// FA3 fuses softmax into the epilogue between QK^T and PV.
// This is complex — we'll use a simplified approach here.

using CollectiveEpilogueQK = typename cutlass::gemm::collective::CollectiveBuilder
  <ArchTag,
   cutlass::arch::OpClassTensorOp,
   ElementAccumulator,
   cutlass::GemmlShape<64, 128, 64>,
   cutlass::gemm::collective::ClusterShapeAuto,
   ElementAccumulator, cutlass::layout::RowMajor,  // FP32 attention scores
   ElementAccumulator, cutlass::layout::RowMajor,
   cutlass::gemm::collective::EpilogueScheduleAuto
   >::CollectiveOp;

// ============================================================================
// GEMM KERNEL TYPES
// ============================================================================

using GemmKernelQK = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::GemmUniversal<
        cutlass::gemm::GemmShape<M_gemm, N_gemm, K_gemm, 1>,
        CollectiveMainloopQK,
        CollectiveEpilogueQK
    >
>;

// ============================================================================
// SOFTMAX — Reference implementation
// ============================================================================

// Online softmax (numerically stable)
__global__ void softmax_kernel(
    float* data, int M, int N
) {
    int row = blockIdx.x;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float val = data[row * N + col];
        max_val = fmaxf(max_val, val);
    }
    
    // Warp reduction for max
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float val = expf(data[row * N + col] - max_val);
        data[row * N + col] = val;
        sum += val;
    }
    
    // Warp reduction for sum
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = sum;
    __syncthreads();
    sum = shared_sum;
    
    // Normalize
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        data[row * N + col] /= sum;
    }
}

void launch_softmax(float* d_data, int M, int N) {
    int threads = 256;
    softmax_kernel<<<M, threads>>>(d_data, M, N);
}

// ============================================================================
// FUSED ATTENTION LAUNCHER
// ============================================================================

struct AttentionConfig {
    int B, H, S, D;
    float scale;  // 1/sqrt(D)
};

template <typename GemmQK, typename GemmPV>
void launch_fused_attention(
    const bfloat16_t* d_Q, const bfloat16_t* d_K, const bfloat16_t* d_V,
    bfloat16_t* d_O,
    float* d_scores,  // Intermediate [B*H*S, S]
    AttentionConfig config,
    cudaStream_t stream = 0
) {
    int M = config.B * config.H * config.S;
    int N = config.S;
    int K = config.D;
    
    // Step 1: QK^T GEMM
    typename GemmQK::Arguments args_qk{
        cutlass::gemm::GemmCoord{M, N, K},
        {d_Q, K},
        {d_K, N},
        {d_scores, N},
        {d_scores, N},
        {config.scale, 0.0f}
    };
    
    GemmQK gemm_qk;
    size_t ws_size = 0;
    GemmQK::get_workspace_size(args_qk, &ws_size);
    void* ws = nullptr;
    if (ws_size > 0) cudaMalloc(&ws, ws_size);
    gemm_qk.run(args_qk, ws, stream);
    if (ws) cudaFree(ws);
    
    // Step 2: Softmax (in-place on scores)
    launch_softmax(d_scores, M, N);
    
    // Step 3: PV GEMM
    // (Would use GemmPV here — simplified for this exercise)
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Module 03, Exercise 03: Warp-Specialized Attention (FA3) ===" << std::endl;
    
    // Check for Hopper GPU
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    if (prop.major < 9) {
        std::cout << "WARNING: This exercise requires Hopper (SM90) GPU." << std::endl;
        std::cout << "Your GPU is SM" << prop.major * 10 + prop.minor << std::endl;
        std::cout << "Code will not launch, but you can still study the configuration." << std::endl;
        std::cout << "\nFA3 Architecture:" << std::endl;
        std::cout << "  - Producer warps: 8 (load Q, K, V)" << std::endl;
        std::cout << "  - Consumer warps: 24 (QK^T, softmax, PV)" << std::endl;
        std::cout << "  - Expected speedup over FA2: 1.5-2.0x" << std::endl;
        return 0;
    }
    
    std::cout << "  SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << std::endl;
    
    // Attention configuration
    AttentionConfig config{B, H, S, D, 1.0f / sqrtf(float(D))};
    
    std::cout << "Attention configuration:" << std::endl;
    std::cout << "  Batch: " << B << ", Heads: " << H << std::endl;
    std::cout << "  Sequence: " << S << ", Head dim: " << D << std::endl;
    std::cout << "  Scale: " << config.scale << std::endl;
    std::cout << std::endl;
    
    // Allocate device memory
    bfloat16_t *d_Q, *d_K, *d_V, *d_O;
    float *d_scores;  // FP32 attention scores [B*H*S, S]
    
    size_t bytes_qkv = B * H * S * D * sizeof(bfloat16_t);
    size_t bytes_scores = B * H * S * S * sizeof(float);
    size_t bytes_out = B * H * S * D * sizeof(bfloat16_t);
    
    cudaMalloc(&d_Q, bytes_qkv);
    cudaMalloc(&d_K, bytes_qkv);
    cudaMalloc(&d_V, bytes_qkv);
    cudaMalloc(&d_O, bytes_out);
    cudaMalloc(&d_scores, bytes_scores);
    
    cutlass_ref::init_matrix_random(d_Q, B * H * S * D);
    cutlass_ref::init_matrix_random(d_K, B * H * S * D);
    cutlass_ref::init_matrix_random(d_V, B * H * S * D);
    
    // ========================================================================
    // FA3 ARCHITECTURE ANALYSIS
    // ========================================================================
    
    std::cout << "=== FA3 ARCHITECTURE ANALYSIS ===" << std::endl;
    std::cout << "Warp specialization for attention:" << std::endl;
    std::cout << std::endl;
    std::cout << "  Producer Warps (8):" << std::endl;
    std::cout << "    ├─ TMA Load Q tiles [B*H*S, D]" << std::endl;
    std::cout << "    ├─ TMA Load K tiles [S, D]" << std::endl;
    std::cout << "    └─ TMA Load V tiles [B*H*S, D]" << std::endl;
    std::cout << std::endl;
    std::cout << "  Consumer Warps (24):" << std::endl;
    std::cout << "    ├─ QK^T MMA → attention scores" << std::endl;
    std::cout << "    ├─ Softmax (in registers)" << std::endl;
    std::cout << "    └─ PV MMA → output" << std::endl;
    std::cout << std::endl;
    std::cout << "  Key insight: 3 matrix loads need more producer warps" << std::endl;
    std::cout << "  than dense GEMM (which only loads 2 matrices)." << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // COMPARISON: FA2 vs FA3
    // ========================================================================
    
    std::cout << "=== FA2 vs FA3 COMPARISON ===" << std::endl;
    std::cout << std::left << std::setw(25) << "Feature"
              << std::right << std::setw(15) << "FA2"
              << std::setw(15) << "FA3" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    std::cout << std::left << std::setw(25) << "Architecture"
              << std::right << std::setw(15) << "Ampere"
              << std::setw(15) << "Hopper" << std::endl;
    std::cout << std::left << std::setw(25) << "Warp specialization"
              << std::right << std::setw(15) << "No"
              << std::setw(15) << "Yes" << std::endl;
    std::cout << std::left << std::setw(25) << "TMA loads"
              << std::right << std::setw(15) << "No"
              << std::setw(15) << "Yes" << std::endl;
    std::cout << std::left << std::setw(25) << "Producer warps"
              << std::right << std::setw(15) << "N/A"
              << std::setw(15) << "8" << std::endl;
    std::cout << std::left << std::setw(25) << "Expected speedup"
              << std::right << std::setw(15) << "1.0x"
              << std::setw(15) << "1.5-2.0x" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // PROFILING GUIDANCE
    // ========================================================================
    
    std::cout << "=== PROFILING GUIDANCE ===" << std::endl;
    std::cout << "Profile FA3 with ncu:" << std::endl;
    std::cout << "  ncu --metrics smsp__thread_inst_executed_per_pipe_tensor.ratio,\\" << std::endl;
    std::cout << "              l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\\" << std::endl;
    std::cout << "              dram__throughput.avg \\" << std::endl;
    std::cout << "       ./fa3_warp_specialized" << std::endl;
    std::cout << std::endl;
    std::cout << "Expected observations:" << std::endl;
    std::cout << "  - High tensor instruction ratio (consumer warps busy)" << std::endl;
    std::cout << "  - High memory bandwidth (producer warps feeding data)" << std::endl;
    std::cout << "  - Low stall percentage (good latency hiding)" << std::endl;
    
    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_scores);
    
    std::cout << "\n=== CHECKPOINT ===" << std::endl;
    std::cout << "C1: Why does FA3 need more producer warps than dense GEMM?" << std::endl;
    std::cout << "C2: How does warp specialization improve attention performance?" << std::endl;
    std::cout << "C3: What's the expected speedup of FA3 over FA2?" << std::endl;
    
    return 0;
}
