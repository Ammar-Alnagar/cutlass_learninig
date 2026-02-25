/**
 * Exercise 02: Collective Copy Operations
 * 
 * Objective: Understand collective copy operations where threads
 *            cooperate to move data efficiently
 * 
 * Tasks:
 * 1. Learn collective operation concepts
 * 2. Understand thread cooperation
 * 3. Practice with TiledCopy
 * 4. Analyze efficiency
 * 
 * Key Concepts:
 * - Collective: All threads participate
 * - Cooperation: Threads work together
 * - Efficiency: Better than individual copies
 */

#include <iostream>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

using namespace cute;

int main() {
    std::cout << "=== Exercise 02: Collective Copy Operations ===" << std::endl;
    std::cout << std::endl;

    // TASK 1: Collective copy concept
    std::cout << "Task 1 - Collective Copy Concept:" << std::endl;
    std::cout << "Collective copy involves all threads:" << std::endl;
    std::cout << "  Each thread copies a portion of data" << std::endl;
    std::cout << "  Cooperation ensures complete coverage" << std::endl;
    std::cout << "  More efficient than individual copies" << std::endl;
    std::cout << std::endl;

    // TASK 2: Simulate collective copy
    std::cout << "Task 2 - Collective Copy Simulation:" << std::endl;
    
    float src[256];
    float dst[256];
    
    for (int i = 0; i < 256; ++i) {
        src[i] = static_cast<float>(i);
        dst[i] = 0.0f;
    }

    auto layout = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
    auto src_tensor = make_tensor(make_gmem_ptr(src), layout);
    auto dst_tensor = make_tensor(make_gmem_ptr(dst), layout);

    std::cout << "16x16 matrix copy with 16 threads:" << std::endl;
    std::cout << "  Each thread copies 16 elements" << std::endl;
    std::cout << std::endl;

    // Simulate thread work distribution
    std::cout << "Thread work distribution:" << std::endl;
    for (int t = 0; t < 16; ++t) {
        int start_element = t * 16;
        std::cout << "  Thread " << t << ": Elements " << start_element 
                  << "-" << (start_element + 15) << std::endl;
    }
    std::cout << std::endl;

    // TASK 3: TiledCopy pattern
    std::cout << "Task 3 - TiledCopy Pattern:" << std::endl;
    std::cout << "CuTe's TiledCopy abstraction:" << std::endl;
    std::cout << "  Defines copy atom (thread cooperation pattern)" << std::endl;
    std::cout << "  Specifies source and destination layouts" << std::endl;
    std::cout << "  Handles thread indexing automatically" << std::endl;
    std::cout << std::endl;

    std::cout << "TiledCopy components:" << std::endl;
    std::cout << "  Copy Atom: How threads cooperate" << std::endl;
    std::cout << "  Src Layout: Source memory organization" << std::endl;
    std::cout << "  Dst Layout: Destination memory organization" << std::endl;
    std::cout << "  Tiler: How work is divided" << std::endl;
    std::cout << std::endl;

    // TASK 4: Compare individual vs collective
    std::cout << "Task 4 - Individual vs Collective:" << std::endl;
    
    std::cout << std::endl;
    std::cout << "| Aspect        | Individual    | Collective    |" << std::endl;
    std::cout << "|---------------|---------------|---------------|" << std::endl;
    std::cout << "| Coordination  | None          | Synchronized  |" << std::endl;
    std::cout << "| Coverage      | May miss data | Complete      |" << std::endl;
    std::cout << "| Efficiency    | Lower         | Higher        |" << std::endl;
    std::cout << "| Vectorization | Difficult     | Easy          |" << std::endl;
    std::cout << std::endl;

    // TASK 5: Collective copy efficiency
    std::cout << "Task 5 - Efficiency Analysis:" << std::endl;
    
    int total_elements = 256;
    int threads = 16;
    int elements_per_thread = total_elements / threads;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Total elements: " << total_elements << std::endl;
    std::cout << "  Threads: " << threads << std::endl;
    std::cout << "  Elements per thread: " << elements_per_thread << std::endl;
    std::cout << std::endl;

    std::cout << "With vectorized loads (4 elements per load):" << std::endl;
    int loads_per_thread = elements_per_thread / 4;
    int total_loads = threads * loads_per_thread;
    std::cout << "  Loads per thread: " << loads_per_thread << std::endl;
    std::cout << "  Total loads: " << total_loads << std::endl;
    std::cout << "  vs scalar: " << total_elements << " loads" << std::endl;
    std::cout << "  Reduction: " << (float)total_loads / total_elements * 100 << "%" << std::endl;
    std::cout << std::endl;

    // CHALLENGE: Design collective copy
    std::cout << "=== Challenge: Design Collective Copy ===" << std::endl;
    std::cout << "For 64x64 matrix with 128 threads:" << std::endl;
    std::cout << "  Total elements: " << 64 * 64 << std::endl;
    std::cout << "  Elements per thread: " << (64 * 64) / 128 << std::endl;
    std::cout << "  With 128-bit loads: " << (64 * 64) / 128 / 4 << " loads per thread" << std::endl;
    std::cout << std::endl;

    // COLLECTIVE COPY PATTERN
    std::cout << "=== Collective Copy Pattern ===" << std::endl;
    std::cout << R"(
// CuTe TiledCopy pattern
auto copy_atom = make_copy_atom(...);
auto src_layout = make_layout(...);
auto dst_layout = make_layout(...);
auto tiler = make_tiler(...);

auto tiled_copy = make_tiled_copy(copy_atom, src_layout, dst_layout, tiler);

// In kernel:
auto thread_copy = tiled_copy.get_thread_copy(thread_idx);
thread_copy(src_tensor, dst_tensor);
)" << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Key Learnings:" << std::endl;
    std::cout << "1. Collective copy involves all threads" << std::endl;
    std::cout << "2. Cooperation ensures complete coverage" << std::endl;
    std::cout << "3. TiledCopy abstracts the complexity" << std::endl;
    std::cout << "4. Vectorization improves efficiency" << std::endl;

    return 0;
}
