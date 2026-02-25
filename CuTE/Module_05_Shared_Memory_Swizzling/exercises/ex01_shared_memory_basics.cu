/**
 * Exercise 01: Shared Memory Basics
 * 
 * Objective: Understand shared memory fundamentals and its role
 *            in CUDA kernel optimization
 * 
 * Tasks:
 * 1. Learn shared memory characteristics
 * 2. Understand bank structure
 * 3. Practice with shared memory allocation
 * 4. Identify use cases
 * 
 * Key Concepts:
 * - Shared Memory: Fast, on-chip, block-scoped memory
 * - Banks: 32 memory banks on sm_80+
 * - Latency: ~100x faster than global memory
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 01: Shared Memory Basics ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Shared memory characteristics
    std::cout << "Task 1 - Shared Memory Characteristics:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "| Property        | Shared Memory | Global Memory |" << std::endl;
    std::cout << "|-----------------|---------------|---------------|" << std::endl;
    std::cout << "| Location        | On-chip       | Off-chip      |" << std::endl;
    std::cout << "| Latency         | ~20 cycles    | ~400 cycles   |" << std::endl;
    std::cout << "| Bandwidth       | Very High     | Lower         |" << std::endl;
    std::cout << "| Scope           | Thread Block  | All Threads   |" << std::endl;
    std::cout << "| Persistence     | Block lifetime| Kernel lifetime|" << std::endl;
    std::cout << "| Size (A100)     | 192 KB/SM     | 40-80 GB      |" << std::endl;
    std::cout << std::endl;

    // TASK 2: Bank structure
    std::cout << "Task 2 - Bank Structure:" << std::endl;
    std::cout << "sm_80+ has 32 banks" << std::endl;
    std::cout << "Each bank services one 4-byte access per cycle" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Bank assignment for 32-bit words:" << std::endl;
    std::cout << "  Address N maps to bank (N / 4) % 32" << std::endl;
    std::cout << std::endl;

    // Show bank mapping for first 64 bytes
    std::cout << "Bank mapping for first 64 bytes:" << std::endl;
    for (int addr = 0; addr < 64; addr += 4) {
        int bank = (addr / 4) % 32;
        std::cout << "  Bytes " << addr << "-" << (addr + 3) << " -> Bank " << bank << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: Simulate shared memory allocation
    std::cout << "Task 3 - Shared Memory Allocation:" << std::endl;
    
    // Simulate shared memory buffer
    float shared_buffer[256];  // 1 KB shared memory simulation
    
    auto smem_layout = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
    auto smem_tensor = make_tensor(make_smem_ptr(shared_buffer), smem_layout);
    
    std::cout << "Allocated 16x16 float matrix in shared memory" << std::endl;
    std::cout << "Total size: " << 16 * 16 * 4 << " bytes" << std::endl;
    std::cout << std::endl;

    // Initialize shared memory
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            smem_tensor(i, j) = static_cast<float>(i * 16 + j);
        }
    }

    std::cout << "Shared memory contents (first 4 rows):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 16; ++j) {
            printf("%3d ", static_cast<int>(smem_tensor(i, j)));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // TASK 4: Common use cases
    std::cout << "Task 4 - Common Use Cases:" << std::endl;
    std::cout << "1. Tiled matrix multiplication (GEMM)" << std::endl;
    std::cout << "   - Load tiles from global to shared" << std::endl;
    std::cout << "   - Reuse data multiple times" << std::endl;
    std::cout << std::endl;
    
    std::cout << "2. Reduction operations" << std::endl;
    std::cout << "   - Block-wide sum, max, min" << std::endl;
    std::cout << "   - Tree-based reduction in shared memory" << std::endl;
    std::cout << std::endl;
    
    std::cout << "3. Data reorganization" << std::endl;
    std::cout << "   - Transpose matrices" << std::endl;
    std::cout << "   - Convert data layouts" << std::endl;
    std::cout << std::endl;
    
    std::cout << "4. Communication" << std::endl;
    std::cout << "   - Thread cooperation within block" << std::endl;
    std::cout << "   - Data sharing between threads" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate shared memory requirements
    std::cout << "=== Challenge: Shared Memory Requirements ===" << std::endl;
    std::cout << "For a GEMM kernel with 128x128 thread blocks:" << std::endl;
    std::cout << "  Tile A: 128x128 FP16 = " << 128 * 128 * 2 << " bytes" << std::endl;
    std::cout << "  Tile B: 128x128 FP16 = " << 128 * 128 * 2 << " bytes" << std::endl;
    std::cout << "  Total shared memory: " << 2 * 128 * 128 * 2 << " bytes" << std::endl;
    std::cout << std::endl;

    // SHARED MEMORY DECLARATION
    std::cout << "=== Shared Memory Declaration ===" << std::endl;
    std::cout << R"(
// Static shared memory (known at compile time)
__shared__ float smem[256];

// Dynamic shared memory (specified at launch)
extern __shared__ float smem[];

// Launch with dynamic shared memory
kernel<<<blocks, threads, shared_mem_size>>>(...);
// shared_mem_size in bytes
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Shared memory is fast, on-chip memory" << std::endl;
    std::cout << "2. 32 banks service concurrent accesses" << std::endl;
    std::cout << "3. Shared memory enables data reuse" << std::endl;
    std::cout << "4. Proper usage is critical for performance" << std::endl;

    return 0;
}
