/**
 * Exercise 05: Copy Atom and Tiled Copy
 * 
 * Objective: Understand CuTe's copy atom abstraction for defining
 *            how threads copy data
 * 
 * Tasks:
 * 1. Understand what a copy atom is
 * 2. See how atoms define thread cooperation
 * 3. Practice with different atom configurations
 * 4. Connect atoms to tiled copy
 * 
 * Key Concepts:
 * - Copy Atom: Defines how a group of threads copies data
 * - Thread Mma: How threads are organized for the operation
 * - Instruction: The actual hardware instruction (cp.async, etc.)
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 05: Copy Atom and Tiled Copy ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Understand copy atom concept
    std::cout << "Task 1 - Copy Atom Concept:" << std::endl;
    std::cout << "A copy atom defines:" << std::endl;
    std::cout << "  - How many threads cooperate" << std::endl;
    std::cout << "  - How data is divided among threads" << std::endl;
    std::cout << "  - What instruction to use (load, store, async)" << std::endl;
    std::cout << std::endl;

    std::cout << "Common copy atoms for sm_89:" << std::endl;
    std::cout << "  - SM75_U32x4_LDSM_N: 4 threads, 32-bit elements, LDSM" << std::endl;
    std::cout << "  - SM75_U16x8_LDSM_N: 8 threads, 16-bit elements, LDSM" << std::endl;
    std::cout << "  - Universal: Works across architectures" << std::endl;
    std::cout << std::endl;

    // TASK 2: Simulate a simple copy atom (4 threads, 4 elements each)
    std::cout << "Task 2 - Simple Copy Atom Simulation:" << std::endl;
    float src_data[32];
    float dst_data[32];
    
    for (int i = 0; i < 32; ++i) {
        src_data[i] = static_cast<float>(i);
        dst_data[i] = 0.0f;
    }

    auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}), GenRowMajor{});
    auto src_tensor = make_tensor(make_gmem_ptr(src_data), layout);
    auto dst_tensor = make_tensor(make_gmem_ptr(dst_data), layout);

    std::cout << "Configuration: 4 threads, each copies 8 elements" << std::endl;
    std::cout << "Total data: 32 elements" << std::endl;
    std::cout << std::endl;

    // Simulate each thread's work
    for (int thread_id = 0; thread_id < 4; ++thread_id) {
        std::cout << "Thread " << thread_id << " copies:" << std::endl;
        int start_row = thread_id;
        for (int j = 0; j < 8; ++j) {
            dst_tensor(start_row, j) = src_tensor(start_row, j);
            std::cout << "  src(" << start_row << "," << j << ") = " 
                      << src_tensor(start_row, j) << std::endl;
        }
        std::cout << std::endl;
    }

    // TASK 3: Different atom configurations
    std::cout << "Task 3 - Different Atom Configurations:" << std::endl;
    std::cout << std::endl;

    struct AtomConfig {
        const char* name;
        int threads;
        int elements_per_thread;
        int total_elements;
    };

    AtomConfig configs[] = {
        {"4 threads × 8 elements", 4, 8, 32},
        {"8 threads × 4 elements", 8, 4, 32},
        {"16 threads × 4 elements", 16, 4, 64},
        {"32 threads × 4 elements", 32, 4, 128}
    };

    for (auto& cfg : configs) {
        std::cout << cfg.name << ":" << std::endl;
        std::cout << "  Total elements: " << cfg.total_elements << std::endl;
        std::cout << "  Elements per thread: " << cfg.elements_per_thread << std::endl;
        std::cout << std::endl;
    }

    // TASK 4: Atom layout calculation
    std::cout << "Task 4 - Atom Layout Calculation:" << std::endl;
    std::cout << "For a 4-thread atom copying a 4x8 tile:" << std::endl;
    std::cout << "  Each thread handles 1 row (8 elements)" << std::endl;
    std::cout << "  Thread 0 -> Row 0" << std::endl;
    std::cout << "  Thread 1 -> Row 1" << std::endl;
    std::cout << "  Thread 2 -> Row 2" << std::endl;
    std::cout << "  Thread 3 -> Row 3" << std::endl;
    std::cout << std::endl;

    // TASK 5: Vectorized atom (128-bit loads)
    std::cout << "Task 5 - Vectorized Copy Atom:" << std::endl;
    std::cout << "With 128-bit loads (4 floats per load):" << std::endl;
    std::cout << "  Each thread loads 2 vectorized elements per row" << std::endl;
    std::cout << "  Row of 8 elements = 2 vectorized loads" << std::endl;
    std::cout << "  4 threads × 2 loads × 4 elements = 32 elements" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Calculate atom efficiency
    std::cout << "=== Challenge: Atom Efficiency ===" << std::endl;
    std::cout << "Which configuration is most efficient?" << std::endl;
    std::cout << std::endl;
    std::cout << "Answer: Depends on the hardware and memory pattern!" << std::endl;
    std::cout << "  - More threads = more parallelism" << std::endl;
    std::cout << "  - Vectorized loads = better bandwidth" << std::endl;
    std::cout << "  - Match atom to memory layout for coalescing" << std::endl;
    std::cout << std::endl;

    // TILED COPY WITH ATOMS
    std::cout << "=== Tiled Copy with Atoms ===" << std::endl;
    std::cout << R"(
// Conceptual CuTe tiled copy with atom
template<typename CopyAtom>
__global__ void tiled_copy_kernel(float* src, float* dst, ...) {
    // Define the copy atom
    CopyAtom atom;
    
    // Source and destination tensors
    auto src_tensor = make_tensor(make_gmem_ptr(src), src_layout);
    auto dst_tensor = make_tensor(make_smem_ptr(dst), dst_layout);
    
    // Thread's copy operation
    // Atom handles the details of how threads cooperate
    atom.copy(src_tensor, dst_tensor, thread_idx);
}
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Copy atom defines thread cooperation pattern" << std::endl;
    std::cout << "2. Atoms specify instruction type and data division" << std::endl;
    std::cout << "3. Different atoms for different architectures" << std::endl;
    std::cout << "4. Atoms enable portable, efficient copy operations" << std::endl;

    return 0;
}
