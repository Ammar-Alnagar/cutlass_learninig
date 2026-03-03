/*
 * EXERCISE: Bank Conflicts Demo - Fill in the Gaps
 * 
 * WHAT THIS TEACHES:
 *   - Understand shared memory bank conflicts
 *   - See how consecutive thread access causes conflicts
 *   - Measure the performance impact of bank conflicts
 *
 * INSTRUCTIONS:
 *   1. Read the comments explaining each concept
 *   2. Find the // TODO: markers
 *   3. Fill in the missing code to complete the exercise
 *   4. Compile and run to verify your solution
 *
 * MENTAL MODEL:
 *   Ada has 32 shared memory banks (bank 0-31)
 *   Consecutive 4-byte words are in consecutive banks
 *   If 32 threads all access bank 0, requests serialize (32x slower)
 *   Swizzle XORs the address so threads access different banks
 */

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/cute.hpp>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

using namespace cute;

// ============================================================================
// KERNEL: Bank Conflict Demonstration (WITHOUT swizzle)
// ============================================================================
__global__ void bank_conflict_kernel(float* gmem_out, bool use_conflicting_access) {
    // MENTAL MODEL: Shared memory tile [32, 32] = 1024 floats
    // 32 threads access column 0 vs. row 0
    __shared__ float smem[32 * 32];

    int tid = threadIdx.x;

    if (use_conflicting_access) {
        // MENTAL MODEL: CONFLICTING ACCESS
        // All 32 threads access smem[0], smem[32], smem[64], ...
        // These are all in bank 0! (addresses 0, 128, 256, ... bytes)
        // Result: 32-way bank conflict, accesses serialize

        // TODO 1: Create column-major access pattern that causes bank conflicts
        // Each thread accesses a different row, same column
        // Formula: idx = row * 32 + tid (where tid is the column)
        for (int i = 0; i < 32; i++) {
            int idx = /* YOUR CODE HERE */;
            smem[idx] = static_cast<float>(tid);
        }
    } else {
        // MENTAL MODEL: COALESCED ACCESS
        // Thread t accesses smem[t], smem[t+1], smem[t+2], ...
        // Each thread accesses a different bank
        // Result: No bank conflicts, full bandwidth

        // TODO 2: Create row-major access pattern (conflict-free)
        // Each thread accesses a different column, same row
        // Formula: idx = tid * 32 + i
        for (int i = 0; i < 32; i++) {
            int idx = /* YOUR CODE HERE */;
            smem[idx] = static_cast<float>(tid);
        }
    }

    __syncthreads();

    // Read back
    if (use_conflicting_access) {
        for (int i = 0; i < 32; i++) {
            // TODO 3: Read back with same conflicting pattern
            int idx = /* YOUR CODE HERE */;
            gmem_out[tid * 32 + i] = smem[idx];
        }
    } else {
        for (int i = 0; i < 32; i++) {
            // TODO 4: Read back with same conflict-free pattern
            int idx = tid * 32 + i;
            gmem_out[tid * 32 + i] = smem[idx];
        }
    }

    // Thread 0 prints info
    if (tid == 0) {
        printf("=== Bank Conflict Demo ===\n");
        printf("Shared memory: 32x32 floats (1024 elements, 4 KB)\n");
        printf("Banks: 32 (4-byte interleaving)\n\n");

        printf("Conflicting access pattern:\n");
        printf("  Thread t accesses: smem[0*32+t], smem[1*32+t], ...\n");
        printf("  All threads access bank %d (32-way conflict!)\n", 0);
        printf("  Expected: ~32x slower than conflict-free\n\n");

        printf("Conflict-free access pattern:\n");
        printf("  Thread t accesses: smem[t*32+0], smem[t*32+1], ...\n");
        printf("  Each thread accesses different banks\n");
        printf("  Expected: Full smem bandwidth\n");
    }
}

// ============================================================================
// MAIN: Compare conflicting vs. conflict-free performance
// ============================================================================
int main() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("=== Bank Conflicts Demo Exercise ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Shared memory banks: 32\n");
    printf("smem per SM: %d KB\n\n", prop.sharedMemPerMultiprocessor);

    // Allocate output [32, 32]
    constexpr int SIZE = 32 * 32;
    float* d_out;
    cudaMalloc(&d_out, SIZE * sizeof(float));

    std::cout << "--- Kernel Output ---\n\n";

    // Warmup
    bank_conflict_kernel<<<1, 32>>>(d_out, true);
    bank_conflict_kernel<<<1, 32>>>(d_out, false);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Time conflicting access
    printf("=== Timing Conflicting Access ===\n");
    nvtxRangePush("conflicting");
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        bank_conflict_kernel<<<1, 32>>>(d_out, true);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    nvtxRangePop();

    float elapsed_conflict;
    cudaEventElapsedTime(&elapsed_conflict, start);
    cudaEventElapsedTime(&elapsed_conflict, stop);
    elapsed_conflict /= 100.0f;

    // Time conflict-free access
    printf("=== Timing Conflict-Free Access ===\n");
    nvtxRangePush("conflict_free");
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        bank_conflict_kernel<<<1, 32>>>(d_out, false);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    nvtxRangePop();

    float elapsed_free;
    cudaEventElapsedTime(&elapsed_free, start);
    cudaEventElapsedTime(&elapsed_free, stop);
    elapsed_free /= 100.0f;

    float slowdown = elapsed_conflict / elapsed_free;

    printf("\n=== Results ===\n");
    printf("Conflicting access:  %.3f ms per kernel\n", elapsed_conflict);
    printf("Conflict-free access: %.3f ms per kernel\n", elapsed_free);
    printf("Slowdown factor: %.1fx\n", slowdown);
    printf("Expected: ~32x (theoretical max for 32-way conflict)\n");

    // Verify correctness
    float h_out[SIZE];
    cudaMemcpy(h_out, d_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < SIZE; i++) {
        if (h_out[i] < 0.0f || h_out[i] > 31.0f) {
            pass = false;
            break;
        }
    }

    printf("\n[%s] Bank conflict demo verified\n", pass ? "PASS" : "FAIL");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);

    return pass ? 0 : 1;
}

/*
 * CHECKPOINT: Answer before moving to ex02
 * 
 * Q1: What causes a 32-way bank conflict?
 *     Answer: _______________
 * 
 * Q2: For row-major [32, 32] smem, which access pattern conflicts?
 *     Answer: _______________
 * 
 * Q3: How does swizzle eliminate bank conflicts?
 *     Answer: _______________
 */
