// SOLUTION: ex03_memory_order_layers
// Demonstrates acquire/release memory ordering

#include <iostream>
#include <atomic>
#include <thread>
#include <vector>

// ==================== The Two Layers of Atomics ====================

// Layer 1: ATOMICITY
//   - std::atomic<T> prevents tearing
//   - Read/write is indivisible

// Layer 2: MEMORY ORDER
//   - memory_order controls WHEN changes become visible
//   - acquire/release create a "synchronizes-with" relationship

// ==================== Example: Flag + Data Pattern ====================

std::atomic<int> data{0};
std::atomic<bool> ready{false};

// Writer: write data, then publish with release semantics
void writer_thread() {
    data.store(42, std::memory_order_relaxed);  // Write data (no ordering needed yet)
    // Release semantics: all writes BEFORE this point are visible
    // to threads that do acquire load of 'ready' AFTER this store
    ready.store(true, std::memory_order_release);
    std::cout << "Writer: data=42, ready=true (release)\n";
}

// Reader: wait for flag, then read data with acquire semantics
void reader_thread() {
    // Acquire semantics: all reads AFTER this point see writes
    // that happened before the matching release store
    while (!ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    // Because we used acquire on ready, we're guaranteed to see
    // all writes that happened before the release store to ready
    std::cout << "Reader: ready=true, data=" << data.load(std::memory_order_relaxed) << "\n";
}

// ==================== Wrong Pattern: Relaxed on Both ====================

std::atomic<int> data_relaxed{0};
std::atomic<bool> ready_relaxed{false};

void writer_relaxed() {
    data_relaxed.store(42, std::memory_order_relaxed);
    ready_relaxed.store(true, std::memory_order_relaxed);
    // BUG: With relaxed, stores can be reordered
    // ready might become visible before data on some architectures
}

void reader_relaxed() {
    while (!ready_relaxed.load(std::memory_order_relaxed)) {
        std::this_thread::yield();
    }
    // BUG: No guarantee we see the data write
    // May see ready=true but data=0 (stale)
    std::cout << "Reader (relaxed): data=" << data_relaxed.load(std::memory_order_relaxed) << "\n";
}

// ==================== Demonstration ====================

void demonstrate_acquire_release() {
    std::cout << "=== Acquire/Release Pattern ===\n";
    
    data = 0;
    ready = false;
    
    std::thread writer(writer_thread);
    std::thread reader(reader_thread);
    
    writer.join();
    reader.join();
    
    std::cout << "(Reader should see data=42)\n\n";
}

void demonstrate_relaxed_problem() {
    std::cout << "=== Relaxed Pattern (Problematic) ===\n";
    
    data_relaxed = 0;
    ready_relaxed = false;
    
    std::thread writer(writer_relaxed);
    std::thread reader(reader_relaxed);
    
    writer.join();
    reader.join();
    
    std::cout << "(Reader MAY see data=0 even though ready=true)\n";
    std::cout << "(On x86, this often works due to strong hardware ordering,\n";
    std::cout << " but on ARM/Power it can fail. The code is still buggy.)\n\n";
}

// ==================== Release Sequence Pattern ====================

void demonstrate_release_sequence() {
    std::cout << "=== Release Sequence (Multiple Writers) ===\n";
    
    std::atomic<int> counter{0};
    
    // Multiple threads increment counter with release
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&counter]() {
            for (int j = 0; j < 100; ++j) {
                counter.fetch_add(1, std::memory_order_release);
            }
        });
    }
    
    for (auto& t : writers) t.join();
    
    // Reader uses acquire to see all increments
    int final_count = counter.load(std::memory_order_acquire);
    std::cout << "Final counter: " << final_count << " (expected: 400)\n\n";
}

int main() {
    demonstrate_acquire_release();
    demonstrate_relaxed_problem();
    demonstrate_release_sequence();
    
    std::cout << "=== KEY LEARNING ===\n";
    std::cout << "Layer 1 (Atomicity): std::atomic prevents tearing\n";
    std::cout << "Layer 2 (Ordering): memory_order controls visibility\n";
    std::cout << "\nAcquire/Release Pattern:\n";
    std::cout << "  Writer: data.store(42); ready.store(true, release);\n";
    std::cout << "  Reader: while(!ready.load(acquire)); read data;\n";
    std::cout << "  Result: Reader sees data=42 (release synchronizes with acquire)\n";
    std::cout << "\nMemory Order Summary:\n";
    std::cout << "  relaxed: atomicity only (fastest)\n";
    std::cout << "  acquire: loads can't move before (reader)\n";
    std::cout << "  release: stores can't move after (writer)\n";
    std::cout << "  acq_rel: both (read-modify-write)\n";
    std::cout << "  seq_cst: all threads see same order (default, slowest)\n";
    
    return 0;
}

// KEY_INSIGHT:
// Two layers of std::atomic:
//
// Layer 1 (ATOMICITY): Prevents tearing
//   - std::atomic<T> ensures indivisible read/write
//   - Covered in ex01, ex02
//
// Layer 2 (ORDERING): Controls visibility timing
//   - memory_order_release: "I'm done writing, make it visible"
//   - memory_order_acquire: "I'm reading, make sure I see everything"
//   - Release synchronizes-with Acquire → visibility guarantee
//
// Release-Acquire Pattern:
//   Thread 1: data = X; flag.store(true, release);
//   Thread 2: while(!flag.load(acquire)); use(data);
//   Result: Thread 2 sees data=X (guaranteed by release-acquire)
//
// CUDA mapping:
//   - __threadfence() ≈ memory_order_seq_cst (global visibility)
//   - __threadfence_block() ≈ memory_order_release (block visibility)
//   - atomicAdd with memory_order_relaxed is common (atomicity only)
