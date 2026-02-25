/**
 * Exercise 07: Complete GEMM Mainloop
 * 
 * Objective: Integrate all concepts into a complete GEMM mainloop
 * 
 * Tasks:
 * 1. Review all components
 * 2. Understand integration points
 * 3. See complete kernel structure
 * 4. Practice with full example
 * 
 * Key Concepts:
 * - Integration: All concepts working together
 * - Complete Kernel: Production-ready structure
 * - Optimization: All techniques combined
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 07: Complete GEMM Mainloop ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Review components
    std::cout << "Task 1 - Component Review:" << std::endl;
    std::cout << "Complete GEMM uses concepts from all modules:" << std::endl;
    std::cout << std::endl;

    std::cout << "Module 01 - Layout Algebra:" << std::endl;
    std::cout << "  - Define matrix layouts" << std::endl;
    std::cout << "  - Tiled organization" << std::endl;
    std::cout << std::endl;

    std::cout << "Module 02 - CuTe Tensors:" << std::endl;
    std::cout << "  - Wrap pointers with layouts" << std::endl;
    std::cout << "  - Access elements safely" << std::endl;
    std::cout << std::endl;

    std::cout << "Module 03 - Tiled Copy:" << std::endl;
    std::cout << "  - Global to shared memory transfer" << std::endl;
    std::cout << "  - Vectorized loads" << std::endl;
    std::cout << std::endl;

    std::cout << "Module 04 - MMA Atoms:" << std::endl;
    std::cout << "  - Tensor Core operations" << std::endl;
    std::cout << "  - Mixed precision compute" << std::endl;
    std::cout << std::endl;

    std::cout << "Module 05 - Shared Memory Swizzling:" << std::endl;
    std::cout << "  - Bank conflict avoidance" << std::endl;
    std::cout << "  - Optimal layouts" << std::endl;
    std::cout << std::endl;

    std::cout << "Module 06 - Collective Mainloops:" << std::endl;
    std::cout << "  - Producer-consumer pipeline" << std::endl;
    std::cout << "  - Double buffering" << std::endl;
    std::cout << std::endl;

    // TASK 2: Complete kernel structure
    std::cout << "Task 2 - Complete Kernel Structure:" << std::endl;
    std::cout << std::endl;

    std::cout << "1. Kernel Entry:" << std::endl;
    std::cout << "   - Calculate thread/block indices" << std::endl;
    std::cout << "   - Initialize accumulators" << std::endl;
    std::cout << std::endl;

    std::cout << "2. Prologue:" << std::endl;
    std::cout << "   - Load first tiles to shared memory" << std::endl;
    std::cout << "   - Issue async copy fences" << std::endl;
    std::cout << std::endl;

    std::cout << "3. Mainloop:" << std::endl;
    std::cout << "   - Wait for async copy" << std::endl;
    std::cout << "   - MMA compute on loaded data" << std::endl;
    std::cout << "   - Load next tiles (async)" << std::endl;
    std::cout << "   - Accumulate results" << std::endl;
    std::cout << std::endl;

    std::cout << "4. Epilogue:" << std::endl;
    std::cout << "   - Wait for final async copy" << std::endl;
    std::cout << "   - Final MMA compute" << std::endl;
    std::cout << "   - Apply activation/bias (optional)" << std::endl;
    std::cout << "   - Store results to global memory" << std::endl;
    std::cout << std::endl;

    // TASK 3: Data flow
    std::cout << "Task 3 - Data Flow:" << std::endl;
    std::cout << std::endl;

    std::cout << "Global Memory (A, B)" << std::endl;
    std::cout << "       |" << std::endl;
    std::cout << "       v (cp.async)" << std::endl;
    std::cout << "Shared Memory (tiles)" << std::endl;
    std::cout << "       |" << std::endl;
    std::cout << "       v (ldmatrix)" << std::endl;
    std::cout << "Registers (fragments)" << std::endl;
    std::cout << "       |" << std::endl;
    std::cout << "       v (mma.sync)" << std::endl;
    std::cout << "Accumulator (results)" << std::endl;
    std::cout << "       |" << std::endl;
    std::cout << "       v (st.global)" << std::endl;
    std::cout << "Global Memory (C)" << std::endl;
    std::cout << std::endl;

    // TASK 4: Performance considerations
    std::cout << "Task 4 - Performance Considerations:" << std::endl;
    std::cout << std::endl;

    std::cout << "Memory Bound Regime:" << std::endl;
    std::cout << "  - Small K, large M, N" << std::endl;
    std::cout << "  - Focus: Memory bandwidth" << std::endl;
    std::cout << "  - Optimize: Coalescing, vectorization" << std::endl;
    std::cout << std::endl;

    std::cout << "Compute Bound Regime:" << std::endl;
    std::cout << "  - Large K" << std::endl;
    std::cout << "  - Focus: Tensor Core utilization" << std::endl;
    std::cout << "  - Optimize: Pipeline depth, occupancy" << std::endl;
    std::cout << std::endl;

    // COMPLETE GEMM KERNEL
    std::cout << "=== Complete GEMM Kernel ===" << std::endl;
    std::cout << R"(
template<typename TileShape, typename KTiles>
__global__ void complete_gemm(
    ElementA* A, ElementB* B, ElementC* C,
    ProblemShape<M, N, K> problem,
    TileShape tile_shape,
    KTiles k_tiles
) {
    // Thread and block indices
    auto block_coord = make_coord(blockIdx.x, blockIdx.y);
    auto thread_coord = make_coord(threadIdx.x, threadIdx.y);
    
    // Allocate shared memory
    extern __shared__ uint8_t smem_base[];
    Tensor sA = make_smem_tensor_A(smem_base);
    Tensor sB = make_smem_tensor_B(smem_base);
    
    // Allocate registers for accumulation
    Tensor accum = make_accum_tensor();
    clear(accum);
    
    // Create TiledCopy and TiledMMA
    auto copy_A = make_tiled_copy_A();
    auto copy_B = make_tiled_copy_B();
    auto mma = make_tiled_mma();
    
    // Prologue: Load first tile
    copy_A(A, sA, 0);
    copy_B(B, sB, 0);
    cp_async_fence();
    
    // Mainloop
    CUTE_UNROLL
    for (int k = 1; k < size(k_tiles); ++k) {
        // Wait for previous load
        cp_async_wait<0>();
        
        // MMA compute on loaded data
        mma(accum, sA, sB);
        
        // Load next tile (async)
        copy_A(A, sA, k);
        copy_B(B, sB, k);
        cp_async_fence();
    }
    
    // Epilogue: Final compute
    cp_async_wait<0>();
    mma(accum, sA, sB);
    
    // Store results
    store(C, accum);
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Complete GEMM integrates all CuTe concepts" << std::endl;
    std::cout << "2. Pipeline structure: prologue, mainloop, epilogue" << std::endl;
    std::cout << "3. Data flows: gmem -> smem -> regs -> accum -> gmem" << std::endl;
    std::cout << "4. Optimize for memory or compute bound regime" << std::endl;

    return 0;
}
